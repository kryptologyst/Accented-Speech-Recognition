"""Data loading and preprocessing for accented speech recognition."""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import pandas as pd
import torch
from datasets import Dataset, load_dataset
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Dataset as TorchDataset

from src.utils.audio import load_audio, normalize_audio, preemphasis

logger = logging.getLogger(__name__)


class AccentDataset(TorchDataset):
    """Dataset class for accented speech recognition."""
    
    def __init__(
        self,
        config: DictConfig,
        split: str = "train",
        transform: Optional[callable] = None
    ):
        """
        Initialize the dataset.
        
        Args:
            config: Dataset configuration.
            split: Dataset split (train, val, test).
            transform: Optional transform to apply.
        """
        self.config = config
        self.split = split
        self.transform = transform
        
        # Load dataset
        self.data = self._load_dataset()
        
        # Filter by accent if specified
        if config.accent.focus_accent:
            self.data = self.data.filter(
                lambda x: x["accent"] == config.accent.focus_accent
            )
        
        logger.info(f"Loaded {len(self.data)} samples for {split} split")
    
    def _load_dataset(self) -> Dataset:
        """Load the dataset based on configuration."""
        if self.config.dataset.name == "common_voice":
            return self._load_common_voice()
        elif self.config.dataset.name == "synthetic":
            return self._load_synthetic_dataset()
        else:
            raise ValueError(f"Unknown dataset: {self.config.dataset.name}")
    
    def _load_common_voice(self) -> Dataset:
        """Load Common Voice dataset."""
        try:
            dataset = load_dataset(
                "mozilla-foundation/common_voice_11_0",
                self.config.dataset.language,
                split=self.split,
                cache_dir=self.config.dataset.cache_dir
            )
            
            # Add accent information (simplified for demo)
            def add_accent_info(example):
                # In a real implementation, you would use actual accent labels
                # For demo purposes, we'll simulate accent distribution
                accents = ["american", "british", "australian", "canadian", "irish"]
                example["accent"] = accents[hash(example["client_id"]) % len(accents)]
                return example
            
            dataset = dataset.map(add_accent_info)
            return dataset
            
        except Exception as e:
            logger.warning(f"Failed to load Common Voice: {e}")
            logger.info("Falling back to synthetic dataset")
            return self._load_synthetic_dataset()
    
    def _load_synthetic_dataset(self) -> Dataset:
        """Load or generate synthetic dataset for demo purposes."""
        # This is a placeholder for synthetic data generation
        # In a real implementation, you would generate synthetic accented speech
        synthetic_data = {
            "audio": [{"path": "synthetic_audio_1.wav"} for _ in range(100)],
            "sentence": ["Hello, how are you today?"] * 100,
            "accent": ["american"] * 25 + ["british"] * 25 + ["australian"] * 25 + ["canadian"] * 25,
            "client_id": [f"spk_{i:03d}" for i in range(100)]
        }
        
        return Dataset.from_dict(synthetic_data)
    
    def __len__(self) -> int:
        """Return dataset length."""
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single sample from the dataset."""
        example = self.data[idx]
        
        # Load audio
        if "path" in example["audio"]:
            waveform, sample_rate = load_audio(
                example["audio"]["path"],
                sample_rate=self.config.audio.sample_rate,
                normalize=self.config.audio.normalize
            )
        else:
            # Handle array-based audio data
            waveform = torch.tensor(example["audio"]["array"], dtype=torch.float32)
            sample_rate = example["audio"]["sampling_rate"]
        
        # Apply pre-emphasis if configured
        if hasattr(self.config.audio, "preemphasis") and self.config.audio.preemphasis > 0:
            waveform = preemphasis(waveform, self.config.audio.preemphasis)
        
        # Apply transforms
        if self.transform:
            waveform = self.transform(waveform)
        
        # Prepare sample
        sample = {
            "input_values": waveform.squeeze(),
            "labels": example["sentence"],
            "accent": example["accent"],
            "speaker_id": example["client_id"],
            "sample_rate": sample_rate
        }
        
        return sample


class AccentDataModule:
    """Data module for managing accented speech datasets."""
    
    def __init__(self, config: DictConfig):
        """
        Initialize the data module.
        
        Args:
            config: Data configuration.
        """
        self.config = config
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
    
    def setup(self) -> None:
        """Setup datasets for all splits."""
        logger.info("Setting up datasets...")
        
        self.train_dataset = AccentDataset(
            self.config,
            split="train",
            transform=self._get_train_transform()
        )
        
        self.val_dataset = AccentDataset(
            self.config,
            split="validation",
            transform=self._get_val_transform()
        )
        
        self.test_dataset = AccentDataset(
            self.config,
            split="test",
            transform=self._get_val_transform()
        )
        
        logger.info("Dataset setup complete")
    
    def _get_train_transform(self) -> Optional[callable]:
        """Get training transforms."""
        if not self.config.augmentation.enabled:
            return None
        
        def transform(waveform):
            # Apply speed perturbation
            if self.config.augmentation.speed_perturbation.enabled:
                from src.utils.audio import speed_perturbation
                waveform, _ = speed_perturbation(
                    waveform,
                    self.config.audio.sample_rate,
                    self.config.augmentation.speed_perturbation.min_speed,
                    self.config.augmentation.speed_perturbation.max_speed
                )
            
            return waveform
        
        return transform
    
    def _get_val_transform(self) -> Optional[callable]:
        """Get validation transforms (no augmentation)."""
        return None
    
    def train_dataloader(self) -> DataLoader:
        """Get training dataloader."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.config.loading.batch_size,
            shuffle=self.config.loading.shuffle,
            num_workers=self.config.loading.num_workers,
            pin_memory=self.config.loading.pin_memory,
            collate_fn=self._collate_fn
        )
    
    def val_dataloader(self) -> DataLoader:
        """Get validation dataloader."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.config.loading.batch_size,
            shuffle=False,
            num_workers=self.config.loading.num_workers,
            pin_memory=self.config.loading.pin_memory,
            collate_fn=self._collate_fn
        )
    
    def test_dataloader(self) -> DataLoader:
        """Get test dataloader."""
        return DataLoader(
            self.test_dataset,
            batch_size=self.config.loading.batch_size,
            shuffle=False,
            num_workers=self.config.loading.num_workers,
            pin_memory=self.config.loading.pin_memory,
            collate_fn=self._collate_fn
        )
    
    def _collate_fn(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """Collate function for batching."""
        # Pad sequences
        input_values = [sample["input_values"] for sample in batch]
        input_values = torch.nn.utils.rnn.pad_sequence(
            input_values, batch_first=True, padding_value=0.0
        )
        
        # Extract other fields
        labels = [sample["labels"] for sample in batch]
        accents = [sample["accent"] for sample in batch]
        speaker_ids = [sample["speaker_id"] for sample in batch]
        
        return {
            "input_values": input_values,
            "labels": labels,
            "accents": accents,
            "speaker_ids": speaker_ids
        }
    
    def get_accent_distribution(self) -> Dict[str, int]:
        """Get distribution of accents in the dataset."""
        if self.train_dataset is None:
            self.setup()
        
        accent_counts = {}
        for sample in self.train_dataset:
            accent = sample["accent"]
            accent_counts[accent] = accent_counts.get(accent, 0) + 1
        
        return accent_counts
