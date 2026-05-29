"""Pad-mask helpers for sequence models."""

from typing import Optional

import torch


def make_pad_mask(lengths: torch.Tensor, max_len: Optional[int] = None) -> torch.Tensor:
    """Return a boolean tensor where True marks padded positions.

    Args:
        lengths: 1-D long tensor of valid sequence lengths, shape (B,).
        max_len: Optional explicit max length. Defaults to ``lengths.max()``.

    Returns:
        Bool tensor of shape (B, max_len) with True on padded positions.
    """
    if lengths.dim() != 1:
        raise ValueError(f"lengths must be 1-D, got shape {tuple(lengths.shape)}")
    batch_size = lengths.size(0)
    if max_len is None:
        max_len = int(lengths.max().item()) if batch_size > 0 else 0
    seq_range = torch.arange(0, max_len, device=lengths.device, dtype=lengths.dtype)
    seq_range = seq_range.unsqueeze(0).expand(batch_size, -1)
    return seq_range >= lengths.unsqueeze(1)
