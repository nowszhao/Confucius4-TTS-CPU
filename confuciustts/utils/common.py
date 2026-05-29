"""
Common utility functions for configuration and model management.

Provides utilities for loading configurations, managing checkpoints,
and handling model initialization.
"""

import torch
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, field, asdict
import safetensors.torch


def load_yaml_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to YAML config file

    Returns:
        Configuration dictionary
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def save_yaml_config(config: Dict[str, Any], config_path: str):
    """
    Save configuration to YAML file.

    Args:
        config: Configuration dictionary
        config_path: Path to save YAML file
    """
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)


def load_checkpoint(
    checkpoint_path: str,
    device: str = "cuda",
    use_safetensors: bool = True,
) -> Dict[str, torch.Tensor]:
    """
    Load model checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file
        device: Device to load checkpoint on
        use_safetensors: Whether to use safetensors format

    Returns:
        State dictionary
    """
    checkpoint_path = Path(checkpoint_path)

    if use_safetensors and checkpoint_path.suffix == ".safetensors":
        # Load safetensors
        state_dict = safetensors.torch.load_file(
            str(checkpoint_path),
            device=device,
        )
    else:
        # Load PyTorch checkpoint
        state_dict = torch.load(
            checkpoint_path,
            map_location=device,
        )

        # Extract state_dict if it's a full checkpoint
        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]

    return state_dict


def save_checkpoint(
    state_dict: Dict[str, torch.Tensor],
    checkpoint_path: str,
    use_safetensors: bool = True,
):
    """
    Save model checkpoint.

    Args:
        state_dict: Model state dictionary
        checkpoint_path: Path to save checkpoint
        use_safetensors: Whether to use safetensors format
    """
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    if use_safetensors:
        # Save as safetensors
        if not str(checkpoint_path).endswith(".safetensors"):
            checkpoint_path = checkpoint_path.with_suffix(".safetensors")

        safetensors.torch.save_file(state_dict, str(checkpoint_path))
    else:
        # Save as PyTorch checkpoint
        torch.save({"state_dict": state_dict}, checkpoint_path)


def get_device(use_cuda: bool = True) -> torch.device:
    """
    Get torch device.

    Args:
        use_cuda: Whether to use CUDA if available

    Returns:
        Torch device
    """
    if use_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def count_parameters(model: torch.nn.Module) -> int:
    """
    Count trainable parameters in model.

    Args:
        model: PyTorch model

    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
