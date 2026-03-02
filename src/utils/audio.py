"""Audio processing utilities for speech recognition."""

import logging
from typing import Dict, List, Optional, Tuple, Union

import librosa
import numpy as np
import soundfile as sf
import torch
import torchaudio
from torch import Tensor

logger = logging.getLogger(__name__)


def load_audio(
    file_path: str,
    sample_rate: int = 16000,
    normalize: bool = True,
    mono: bool = True
) -> Tuple[Tensor, int]:
    """
    Load audio file and return waveform tensor.
    
    Args:
        file_path: Path to audio file.
        sample_rate: Target sample rate.
        normalize: Whether to normalize audio.
        mono: Whether to convert to mono.
        
    Returns:
        Tuple of (waveform, actual_sample_rate).
    """
    try:
        waveform, sr = torchaudio.load(file_path)
        
        # Convert to mono if needed
        if mono and waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # Resample if needed
        if sr != sample_rate:
            resampler = torchaudio.transforms.Resample(sr, sample_rate)
            waveform = resampler(waveform)
            sr = sample_rate
        
        # Normalize if requested
        if normalize:
            waveform = normalize_audio(waveform)
        
        return waveform, sr
        
    except Exception as e:
        logger.error(f"Error loading audio file {file_path}: {e}")
        raise


def normalize_audio(waveform: Tensor) -> Tensor:
    """
    Normalize audio waveform to [-1, 1] range.
    
    Args:
        waveform: Input audio waveform.
        
    Returns:
        Normalized waveform.
    """
    if waveform.numel() == 0:
        return waveform
    
    max_val = torch.max(torch.abs(waveform))
    if max_val > 0:
        waveform = waveform / max_val
    
    return waveform


def preemphasis(waveform: Tensor, coeff: float = 0.97) -> Tensor:
    """
    Apply pre-emphasis filter to waveform.
    
    Args:
        waveform: Input audio waveform.
        coeff: Pre-emphasis coefficient.
        
    Returns:
        Pre-emphasized waveform.
    """
    if coeff == 0.0:
        return waveform
    
    # Apply pre-emphasis: y[n] = x[n] - coeff * x[n-1]
    emphasized = torch.zeros_like(waveform)
    emphasized[..., 0] = waveform[..., 0]
    emphasized[..., 1:] = waveform[..., 1:] - coeff * waveform[..., :-1]
    
    return emphasized


def extract_log_mel_spectrogram(
    waveform: Tensor,
    sample_rate: int = 16000,
    n_fft: int = 1024,
    hop_length: int = 320,
    win_length: int = 1024,
    n_mels: int = 80,
    f_min: float = 0.0,
    f_max: Optional[float] = None,
    preemphasis_coeff: float = 0.97
) -> Tensor:
    """
    Extract log mel spectrogram from waveform.
    
    Args:
        waveform: Input audio waveform.
        sample_rate: Sample rate of audio.
        n_fft: FFT window size.
        hop_length: Hop length for STFT.
        win_length: Window length for STFT.
        n_mels: Number of mel filter banks.
        f_min: Minimum frequency.
        f_max: Maximum frequency.
        preemphasis_coeff: Pre-emphasis coefficient.
        
    Returns:
        Log mel spectrogram tensor.
    """
    if f_max is None:
        f_max = sample_rate // 2
    
    # Apply pre-emphasis
    if preemphasis_coeff > 0:
        waveform = preemphasis(waveform, preemphasis_coeff)
    
    # Extract mel spectrogram
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        n_mels=n_mels,
        f_min=f_min,
        f_max=f_max,
        power=2.0
    )
    
    mel_spec = mel_transform(waveform)
    
    # Convert to log scale
    log_mel_spec = torch.log(mel_spec + 1e-8)
    
    return log_mel_spec


def apply_spec_augment(
    spectrogram: Tensor,
    time_mask_param: int = 27,
    freq_mask_param: int = 12,
    time_mask_num: int = 2,
    freq_mask_num: int = 2
) -> Tensor:
    """
    Apply SpecAugment to spectrogram.
    
    Args:
        spectrogram: Input spectrogram.
        time_mask_param: Maximum time mask length.
        freq_mask_param: Maximum frequency mask length.
        time_mask_num: Number of time masks.
        freq_mask_num: Number of frequency masks.
        
    Returns:
        Augmented spectrogram.
    """
    augmented = spectrogram.clone()
    
    # Apply time masking
    for _ in range(time_mask_num):
        if augmented.shape[-1] > time_mask_param:
            t = torch.randint(0, time_mask_param, (1,)).item()
            t0 = torch.randint(0, augmented.shape[-1] - t, (1,)).item()
            augmented[..., t0:t0+t] = 0
    
    # Apply frequency masking
    for _ in range(freq_mask_num):
        if augmented.shape[-2] > freq_mask_param:
            f = torch.randint(0, freq_mask_param, (1,)).item()
            f0 = torch.randint(0, augmented.shape[-2] - f, (1,)).item()
            augmented[..., f0:f0+f, :] = 0
    
    return augmented


def speed_perturbation(
    waveform: Tensor,
    sample_rate: int,
    min_speed: float = 0.9,
    max_speed: float = 1.1
) -> Tuple[Tensor, int]:
    """
    Apply speed perturbation to waveform.
    
    Args:
        waveform: Input audio waveform.
        sample_rate: Sample rate of audio.
        min_speed: Minimum speed factor.
        max_speed: Maximum speed factor.
        
    Returns:
        Tuple of (perturbed_waveform, new_sample_rate).
    """
    speed_factor = torch.rand(1).item() * (max_speed - min_speed) + min_speed
    
    # Convert to numpy for librosa
    if waveform.dim() > 1:
        waveform_np = waveform.squeeze().numpy()
    else:
        waveform_np = waveform.numpy()
    
    # Apply speed perturbation
    perturbed_np = librosa.effects.time_stretch(waveform_np, rate=speed_factor)
    
    # Convert back to tensor
    perturbed = torch.from_numpy(perturbed_np).float()
    
    # Adjust sample rate
    new_sample_rate = int(sample_rate * speed_factor)
    
    return perturbed, new_sample_rate


def pad_sequence(
    sequences: List[Tensor],
    batch_first: bool = True,
    padding_value: float = 0.0
) -> Tensor:
    """
    Pad a list of variable length sequences.
    
    Args:
        sequences: List of sequences to pad.
        batch_first: Whether batch dimension is first.
        padding_value: Value to use for padding.
        
    Returns:
        Padded tensor.
    """
    return torch.nn.utils.rnn.pad_sequence(
        sequences, batch_first=batch_first, padding_value=padding_value
    )


def compute_rtf(
    audio_duration: float,
    processing_time: float
) -> float:
    """
    Compute Real-Time Factor (RTF).
    
    Args:
        audio_duration: Duration of audio in seconds.
        processing_time: Processing time in seconds.
        
    Returns:
        Real-time factor.
    """
    if audio_duration == 0:
        return float('inf')
    return processing_time / audio_duration


def validate_audio_file(file_path: str) -> bool:
    """
    Validate that audio file can be loaded.
    
    Args:
        file_path: Path to audio file.
        
    Returns:
        True if file is valid, False otherwise.
    """
    try:
        waveform, _ = load_audio(file_path)
        return waveform.numel() > 0
    except Exception:
        return False
