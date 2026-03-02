"""Utility functions for device management and deterministic behavior."""

import os
import random
from typing import Optional, Union

import numpy as np
import torch
import torchaudio


def get_device(device: Optional[str] = None) -> torch.device:
    """
    Get the appropriate device for computation.
    
    Args:
        device: Device specification. If None, auto-detect based on availability.
        
    Returns:
        torch.device: The device to use for computation.
    """
    if device is None or device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    else:
        return torch.device(device)


def set_seed(seed: int) -> None:
    """
    Set random seed for reproducibility.
    
    Args:
        seed: Random seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Set deterministic behavior for PyTorch
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Set deterministic behavior for torchaudio
    torchaudio.set_audio_backend("sox_io")


def get_audio_backend() -> str:
    """
    Get the appropriate audio backend for the current platform.
    
    Returns:
        str: Audio backend name.
    """
    if torch.cuda.is_available():
        return "sox_io"  # Most reliable for CUDA
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "sox_io"  # Works well with MPS
    else:
        return "sox_io"  # Default fallback


def setup_environment(seed: int = 42, device: Optional[str] = None) -> torch.device:
    """
    Setup the environment for reproducible training.
    
    Args:
        seed: Random seed for reproducibility.
        device: Device specification.
        
    Returns:
        torch.device: The device to use.
    """
    # Set random seed
    set_seed(seed)
    
    # Get device
    device = get_device(device)
    
    # Set audio backend
    torchaudio.set_audio_backend(get_audio_backend())
    
    # Set environment variables for reproducibility
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    
    return device


def count_parameters(model: torch.nn.Module) -> int:
    """
    Count the number of trainable parameters in a model.
    
    Args:
        model: PyTorch model.
        
    Returns:
        int: Number of trainable parameters.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model_size(model: torch.nn.Module) -> str:
    """
    Get human-readable model size.
    
    Args:
        model: PyTorch model.
        
    Returns:
        str: Model size in MB.
    """
    param_size = 0
    buffer_size = 0
    
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    size_all_mb = (param_size + buffer_size) / 1024**2
    return f"{size_all_mb:.2f} MB"


def format_time(seconds: float) -> str:
    """
    Format time in seconds to human-readable string.
    
    Args:
        seconds: Time in seconds.
        
    Returns:
        str: Formatted time string.
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}m {secs:.1f}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        return f"{hours}h {minutes}m {secs:.1f}s"
