import torch
import torch.nn as nn
import sys
import os
from typing import Optional, Tuple
import torchaudio

AMPHION_PATH = os.path.join(os.path.dirname(__file__), "../../external/Amphion")
if AMPHION_PATH not in sys.path:
    sys.path.insert(0, AMPHION_PATH)


class SemanticExtractor(nn.Module):
    """
    Wrapper for Amphion MaskGCT semantic extraction.

    Extracts semantic tokens from audio using the w2v-BERT model
    from Amphion's MaskGCT implementation.
    """

    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        sample_rate: int = 16000,
    ):
        """
        Initialize semantic extractor.

        Args:
            model_path: Path to w2v-BERT checkpoint
            device: Device to load model on
            sample_rate: Expected audio sample rate
        """
        super().__init__()

        try:
            from transformers import SeamlessM4TFeatureExtractor, Wav2Vec2BertModel
        except ImportError:
            raise ImportError(
                f"Cannot import Amphion MaskGCT. Make sure it's available at {AMPHION_PATH}"
            )

        self.device = device
        self.sample_rate = sample_rate

        hf_name = model_path if model_path else "facebook/w2v-bert-2.0"
        self.processor = SeamlessM4TFeatureExtractor.from_pretrained(hf_name)
        self.model = Wav2Vec2BertModel.from_pretrained(hf_name)
        self.model.eval()
        self.model.to(device)
        self.semantic_mean = None
        self.semantic_std = None

    @torch.no_grad()
    def extract(
        self,
        audio: torch.Tensor,
        audio_sr: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Extract semantic features from audio.

        Args:
            audio: Audio waveform (batch, samples) or (samples,)
            audio_sr: Audio sample rate (if different from model's expected rate)

        Returns:
            Semantic features of shape (batch, feature_dim, time)
        """
        # Ensure audio is 2D
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)

        # Resample if needed
        if audio_sr is not None and audio_sr != self.sample_rate:
            audio = torchaudio.functional.resample(
                audio,
                orig_freq=audio_sr,
                new_freq=self.sample_rate,
            )

        audio = audio.to(self.device)

        inputs = self.processor(audio.squeeze(0).cpu().numpy(), sampling_rate=self.sample_rate, return_tensors="pt")
        input_features = inputs["input_features"][0].to(self.device)
        attention_mask = inputs["attention_mask"][0].to(self.device)

        vq_emb = self.model(input_features=input_features, attention_mask=attention_mask, output_hidden_states=True)
        feat = vq_emb.hidden_states[17]  # (B, T, C)
        if self.semantic_mean is not None and self.semantic_std is not None:
            feat = (feat - self.semantic_mean.to(feat)) / self.semantic_std.to(feat)
        return feat

    @torch.no_grad()
    def extract_from_file(
        self,
        audio_path: str,
    ) -> torch.Tensor:
        """
        Extract semantic features from audio file.

        Args:
            audio_path: Path to audio file

        Returns:
            Semantic features of shape (1, feature_dim, time)
        """
        # Load audio
        audio, sr = torchaudio.load(audio_path)

        # Convert to mono if stereo
        if audio.shape[0] > 1:
            audio = audio.mean(dim=0, keepdim=True)

        # Extract features
        return self.extract(audio.squeeze(0), audio_sr=sr)


class SemanticCodec(nn.Module):
    """
    Semantic codec for extracting and quantizing semantic tokens.
    """

    def __init__(
        self,
        semantic_model_path: str,
        codec_model_path: str,
        device: str = "cuda",
        sample_rate: int = 16000,
    ):
        """
        Initialize semantic codec.

        Args:
            semantic_model_path: Path to w2v-BERT checkpoint
            codec_model_path: Path to codec checkpoint
            device: Device to load models on
            sample_rate: Expected audio sample rate
        """
        super().__init__()

        try:
            from models.tts.maskgct.maskgct_utils import build_semantic_codec
        except ImportError:
            raise ImportError(
                f"Cannot import Amphion MaskGCT codec. Make sure it's available at {AMPHION_PATH}"
            )

        self.device = device
        self.sample_rate = sample_rate

        self.semantic_extractor = SemanticExtractor(
            semantic_model_path,
            device=device,
            sample_rate=sample_rate,
        )

        cfg = torch.load(codec_model_path, map_location=device)
        self.codec_model = build_semantic_codec(cfg["cfg"], device)
        self.codec_model.load_state_dict(cfg["model"])
        self.codec_model.eval()

    @torch.no_grad()
    def encode(
        self,
        audio: torch.Tensor,
        audio_sr: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode audio to semantic codes.

        Args:
            audio: Audio waveform (batch, samples) or (samples,)
            audio_sr: Audio sample rate

        Returns:
            Tuple of (semantic_codes, semantic_features)
            - semantic_codes: Discrete codes (batch, time)
            - semantic_features: Continuous features (batch, feature_dim, time)
        """
        # Extract semantic features
        semantic_features = self.semantic_extractor.extract(audio, audio_sr)

        # Quantize to discrete codes
        semantic_codes = self.codec_model.encode(semantic_features)

        return semantic_codes, semantic_features

    @torch.no_grad()
    def encode_from_file(
        self,
        audio_path: str,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode audio file to semantic codes.

        Args:
            audio_path: Path to audio file

        Returns:
            Tuple of (semantic_codes, semantic_features)
        """
        # Load audio
        audio, sr = torchaudio.load(audio_path)

        # Convert to mono if stereo
        if audio.shape[0] > 1:
            audio = audio.mean(dim=0, keepdim=True)

        # Encode
        return self.encode(audio.squeeze(0), audio_sr=sr)


def load_semantic_extractor(
    model_path: str,
    device: str = "cuda",
    sample_rate: int = 16000,
) -> SemanticExtractor:
    """
    Load semantic extractor from checkpoint.

    Args:
        model_path: Path to w2v-BERT checkpoint
        device: Device to load on
        sample_rate: Expected audio sample rate

    Returns:
        Loaded semantic extractor
    """
    return SemanticExtractor(model_path, device, sample_rate)


def load_semantic_codec(
    semantic_model_path: str,
    codec_model_path: str,
    device: str = "cuda",
    sample_rate: int = 16000,
) -> SemanticCodec:
    """
    Load semantic codec from checkpoints.

    Args:
        semantic_model_path: Path to w2v-BERT checkpoint
        codec_model_path: Path to codec checkpoint
        device: Device to load on
        sample_rate: Expected audio sample rate

    Returns:
        Loaded semantic codec
    """
    return SemanticCodec(
        semantic_model_path,
        codec_model_path,
        device,
        sample_rate,
    )
