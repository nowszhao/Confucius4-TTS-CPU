"""
This code is modified from https://github.com/QwenLM/Qwen3-TTS.
"""

import torch
import torch.nn as nn


class TextEmbeddingProjector(nn.Module):
    """
    MLP for resizing text embedding dimension.
    Reference: Qwen3TTSTalkerResizeMLP from https://github.com/QwenLM/Qwen3-TTS.
    
    Structure: Embedding -> Linear(input_size, intermediate_size) -> Act -> Linear(intermediate_size, output_size)
    """
    def __init__(
        self, 
        vocab_size: int, 
        embed_dim: int, 
        output_size: int, 
        hidden_act: str = "silu", 
        bias: bool = True,
    ):
        super().__init__()
        
        # Initialize embedding layer and load pretrained weights
        self.embed = nn.Embedding(vocab_size, embed_dim)
        
        # Freeze embedding weights
        self.embed.weight.requires_grad = False
        self.embed.eval()
        
        # Text projection MLP (following Qwen3TTSTalkerResizeMLP structure)
        self.text_projection_fc1 = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.text_projection_fc2 = nn.Linear(embed_dim, output_size, bias=bias)
        
        # Activation function
        if hidden_act == "silu":
            self.act_fn = nn.SiLU()
        elif hidden_act == "gelu":
            self.act_fn = nn.GELU()
        elif hidden_act == "relu":
            self.act_fn = nn.ReLU()
        else:
            self.act_fn = nn.SiLU()  # Default to SiLU
    
        # Initialize projection layers
        nn.init.normal_(self.text_projection_fc1.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.text_projection_fc2.weight, mean=0.0, std=0.02)
        if bias:
            nn.init.zeros_(self.text_projection_fc1.bias)
            nn.init.zeros_(self.text_projection_fc2.bias)

    def forward(self, text_ids: torch.Tensor) -> torch.Tensor:
        """Resize text embeddings through MLP projection."""
        with torch.no_grad():
            text_embeds = self.embed(text_ids)
        # MLP projection: fc1 -> act -> fc2
        return self.text_projection_fc2(self.act_fn(self.text_projection_fc1(text_embeds)))
    
    def load_pretrained_embeddings(self, pretrained_weights: torch.Tensor):
        """Load pretrained embedding weights."""
        self.embed.weight.data.copy_(pretrained_weights)
        self.embed.weight.requires_grad = False
