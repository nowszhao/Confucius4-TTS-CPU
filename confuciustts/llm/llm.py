import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2Config, GPT2Model
from transformers.modeling_utils import PreTrainedModel
from transformers.generation import GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions
from transformers.configuration_utils import PretrainedConfig
from dataclasses import dataclass
from typing import Optional, Tuple

from .text_encoder import TextEmbeddingProjector
from .speaker_encoder import Qwen3TTSSpeakerEncoder, Qwen3TTSSpeakerEncoderConfig
from .position_embeddings import DummyPositionEmbedding, LearnedPositionalEmbedding

@dataclass
class Text2SemanticConfig(PretrainedConfig):
    """Configuration for the Text2Semantic model (speaker condition + text → semantic tokens).

    Attributes:
        num_layers: Number of transformer layers.
        model_dim: Hidden dimension of the transformer.
        num_heads: Number of attention heads.
        max_text_seq_lens: Maximum number of text tokens.
        max_semantic_seq_lens: Maximum number of semantic tokens.
        vocab_size: Text vocabulary size.
        semantic_vocab_size: Semantic token vocabulary size (includes BOS/EOS).
        text_embedding_dim: Input dimension of the text embedding table.
        speaker_embedding_dim: Input mel feature dimension for the speaker encoder.
        start_semantic_token: BOS token index for the semantic sequence.
        stop_semantic_token: EOS token index for the semantic sequence.
    """

    model_type = "text2semantic"

    num_layers: int = 24
    model_dim: int = 1280
    num_heads: int = 20
    max_text_seq_lens: int = 520
    max_semantic_seq_lens: int = 1520
    vocab_size: int = 32000
    semantic_vocab_size: int = 8194
    text_embedding_dim: int = 4096
    speaker_embedding_dim: int = 1024
    start_semantic_token: int = 8192
    stop_semantic_token: int = 8193

    def __init__(
        self,
        num_layers: int = 24,
        model_dim: int = 1280,
        num_heads: int = 20,
        max_text_seq_lens: int = 520,
        max_semantic_seq_lens: int = 1520,
        vocab_size: int = 32000,
        semantic_vocab_size: int = 8194,
        text_embedding_dim: int = 4096,
        speaker_embedding_dim: int = 1024,
        start_semantic_token: int = 8192,
        stop_semantic_token: int = 8193,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.num_layers = num_layers
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.max_text_seq_lens = max_text_seq_lens
        self.max_semantic_seq_lens = max_semantic_seq_lens
        self.vocab_size = vocab_size
        self.semantic_vocab_size = semantic_vocab_size
        self.text_embedding_dim = text_embedding_dim
        self.speaker_embedding_dim = speaker_embedding_dim
        self.start_semantic_token = start_semantic_token
        self.stop_semantic_token = stop_semantic_token


class Text2Semantic(PreTrainedModel, GenerationMixin):

    config_class = Text2SemanticConfig

    def __init__(self, config: Text2SemanticConfig):
        super().__init__(config)

        self.config = config
        self.max_seq_len = config.max_text_seq_lens + config.max_semantic_seq_lens + 1

        self.text_projector = TextEmbeddingProjector(
            vocab_size=config.vocab_size,
            embed_dim=config.text_embedding_dim,
            output_size=config.model_dim,
        )

        self.semantic_embedding = nn.Embedding(
            config.semantic_vocab_size, config.model_dim
        )

        self.text_position_embedding = LearnedPositionalEmbedding(
            config.max_text_seq_lens, config.model_dim
        )
        self.semantic_position_embedding = LearnedPositionalEmbedding(
            config.max_semantic_seq_lens, config.model_dim
        )

        gpt_config = GPT2Config(
            vocab_size=config.semantic_vocab_size,
            n_positions=self.max_seq_len,
            n_ctx=self.max_seq_len,
            n_embd=config.model_dim,
            n_layer=config.num_layers,
            n_head=config.num_heads,
            gradient_checkpointing=False,
            use_cache=True,
        )
        self.transformer = GPT2Model(gpt_config)

        del self.transformer.wpe
        self.transformer.wpe = DummyPositionEmbedding(config.model_dim)

        del self.transformer.wte

        del self.transformer.ln_f
        self.transformer.ln_f = nn.Identity()

        self.final_norm = nn.LayerNorm(config.model_dim)
        self.semantic_head = nn.Linear(config.model_dim, config.semantic_vocab_size)

        speaker_config = Qwen3TTSSpeakerEncoderConfig(
            mel_dim=config.speaker_embedding_dim,
            enc_dim=config.model_dim,
        )
        self.speaker_encoder = Qwen3TTSSpeakerEncoder(speaker_config)

        self.cached_condition_emb = None
        self.cached_text_emb = None

        self.post_init()


    def _prepare_embed_inputs(
        self,
        text_inputs: Optional[torch.Tensor] = None,
        semantic_codes: Optional[torch.Tensor] = None,
        condition_vector: Optional[torch.Tensor] = None,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Build the inputs_embeds tensor that will be fed into the transformer.

        Args:
            text_inputs: (batch, text_len) text token ids.
            semantic_codes: (batch, sem_len) semantic token ids used in training.
            condition_vector: (batch, mel_frames, speaker_dim) mel features for the speaker encoder.
            input_ids: (batch, seq_len) token ids used during inference; the first
            prefix_len positions are placeholder values replaced by cached embeddings.
            attention_mask: (batch, total_len).

        Returns:
            inputs_embeds: (batch, seq_len, model_dim) embedding tensor.
        """
        if text_inputs is not None:
            # Training: build the full sequence [condition | text | semantic]
            text_emb = self.text_projector(text_inputs)
            text_emb = self.text_position_embedding(text_emb)

            semantic_emb = self.semantic_embedding(semantic_codes)
            semantic_emb = self.semantic_position_embedding(semantic_emb)

            condition_emb = self.speaker_encoder(condition_vector).unsqueeze(1)

            return torch.cat([condition_emb, text_emb, semantic_emb], dim=1)

        else:
            assert self.cached_condition_emb is not None, "Call store_conditioning() first"
            assert self.cached_text_emb is not None, "Call store_conditioning() first"

            condition_len = self.cached_condition_emb.shape[1]
            text_len = self.cached_text_emb.shape[1]
            prefix_len = condition_len + text_len

            if input_ids.shape[1] != 1:
                semantic_inputs = input_ids[:, prefix_len:]
                semantic_emb = self.semantic_embedding(semantic_inputs)
                semantic_emb = self.semantic_position_embedding(semantic_emb)

                if self.cached_condition_emb.shape[0] != semantic_emb.shape[0]:
                    condition_emb = self.cached_condition_emb.repeat_interleave(
                        semantic_emb.shape[0] // self.cached_condition_emb.shape[0], 0
                    )
                    text_emb = self.cached_text_emb.repeat_interleave(
                        semantic_emb.shape[0] // self.cached_text_emb.shape[0], 0
                    )
                else:
                    condition_emb = self.cached_condition_emb
                    text_emb = self.cached_text_emb

                return torch.cat([condition_emb, text_emb, semantic_emb], dim=1)

            else:
                semantic_emb = self.semantic_embedding(input_ids)
                assert attention_mask is not None, (
                    "attention_mask is required for single-token KV-cache decoding"
                )
                semantic_pos = attention_mask.shape[1] - prefix_len - 1
                pos_emb = self.semantic_position_embedding.get_fixed_embedding(
                    semantic_pos, input_ids.device
                )
                return semantic_emb + pos_emb

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        text_inputs: Optional[torch.Tensor] = None,
        text_lengths: Optional[torch.Tensor] = None,
        semantic_codes: Optional[torch.Tensor] = None,
        semantic_lengths: Optional[torch.Tensor] = None,
        condition_vector: Optional[torch.Tensor] = None,
        return_latent: bool = False,
    ):
        """
        Compute semantic token logits or latent hidden states for training and inference(return_latent=True).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if text_inputs is not None:
            inputs_embeds = self._prepare_embed_inputs(
                text_inputs=text_inputs,
                semantic_codes=semantic_codes,
                condition_vector=condition_vector,
            )

            if attention_mask is None:
                batch_size = text_inputs.shape[0]
                device = text_inputs.device
                cond_mask = torch.ones(batch_size, 1, dtype=torch.bool, device=device)
                text_mask = torch.arange(text_inputs.shape[1], device=device).unsqueeze(0) < text_lengths.unsqueeze(1)
                semantic_mask = torch.arange(semantic_codes.shape[1], device=device).unsqueeze(0) < (semantic_lengths + 2).unsqueeze(1)
                attention_mask = torch.cat([cond_mask, text_mask, semantic_mask], dim=1)

        else:
            inputs_embeds = self._prepare_embed_inputs(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

        transformer_outputs = self.transformer(
            inputs_embeds=inputs_embeds,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = transformer_outputs.last_hidden_state

        if return_latent:
            # Strip the condition token (index 0) and the BOS/EOS semantic tokens
            # (last two positions) to return only the interior semantic hidden states.
            return hidden_states[:, 1 + text_inputs.shape[1]:-2]

        if text_inputs is not None:
            # Skip the condition token ([:, 1:]) then skip text tokens to get only
            # the semantic portion before projecting to logits.
            hidden_states = self.final_norm(hidden_states[:, 1:])[:, text_inputs.shape[1]:]

        else:
            hidden_states = self.final_norm(hidden_states)

        logits = self.semantic_head(hidden_states)

        loss = None
        if labels is not None:
            logits_for_loss = logits.permute(0, 2, 1)
            loss = F.cross_entropy(logits_for_loss, labels, ignore_index=-100)

        if not return_dict:
            output = (logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )

    def store_conditioning(
        self,
        condition_vector: torch.Tensor,
        text_inputs: torch.Tensor
    ):
        """
        Pre-compute and cache speaker and text embeddings for reuse across decoding steps.
        """
        with torch.no_grad():
            condition_emb = self.speaker_encoder(condition_vector).unsqueeze(1)
            text_emb = self.text_projector(text_inputs)
            text_emb = self.text_position_embedding(text_emb)

            self.cached_condition_emb = condition_emb
            self.cached_text_emb = text_emb

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        **kwargs
    ):
        """
        Prepare model inputs for each generation step (HuggingFace GenerationMixin hook).
        """
        attention_mask = kwargs.get("attention_mask", None)

        if past_key_values:
            input_ids = input_ids[:, -1:]

        return {
            "input_ids": input_ids,
            "past_key_values": past_key_values,
            "use_cache": kwargs.get("use_cache", True),
            "attention_mask": attention_mask,
        }

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        return tuple(
            tuple(
                past_state.index_select(0, beam_idx.to(past_state.device))
                for past_state in layer_past
            )
            for layer_past in past_key_values
        )

    @torch.inference_mode()
    def generate(
        self,
        text_inputs: torch.Tensor,
        condition_vector: torch.Tensor,
        max_length: int = 500,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9,
        do_sample: bool = True,
        eos_token_id: Optional[int] = None,
        **kwargs
    ) -> torch.Tensor:
        """Generate semantic token sequences from text and speaker condition.

        Args:
            text_inputs: (batch, text_len) text token ids.
            condition_vector: (batch, mel_frames, speaker_dim) mel features for the speaker.
            max_length: Maximum total sequence length passed to HuggingFace generate.
            temperature: Sampling temperature.
            top_k: Top-k sampling parameter.
            top_p: Nucleus sampling parameter.
            do_sample: If False, uses greedy decoding.
            eos_token_id: EOS token to stop generation; defaults to stop_semantic_token.
            **kwargs: Additional arguments forwarded to HuggingFace generate.

        Returns:
            (batch, generated_len) tensor of semantic token indices, with the
            placeholder prefix stripped from the HuggingFace generate output.
        """
        self.store_conditioning(condition_vector, text_inputs)

        batch_size = text_inputs.shape[0]
        device = text_inputs.device
        
        prefix_len = 1 + text_inputs.shape[1]

        bos = self.config.start_semantic_token
        eos = eos_token_id if eos_token_id is not None else self.config.stop_semantic_token
        start_tokens = torch.full(
            (batch_size, prefix_len + 1),
            fill_value=bos,
            dtype=torch.long,
            device=device,
        )
        attention_mask = torch.ones(batch_size, prefix_len + 1, dtype=torch.long, device=device)

        generated = super().generate(
            start_tokens,
            attention_mask=attention_mask,
            max_length=max_length,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            do_sample=do_sample,
            eos_token_id=eos,
            pad_token_id=self.config.stop_semantic_token,
            **kwargs
        )

        return generated[:, prefix_len + 1:]
