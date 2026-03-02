"""Wav2Vec2-based ASR model for accented speech recognition."""

import logging
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers import (
    Wav2Vec2ForCTC,
    Wav2Vec2Processor,
    Wav2Vec2CTCTokenizer,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2Config
)
from omegaconf import DictConfig

from src.utils.device import get_device

logger = logging.getLogger(__name__)


class Wav2Vec2ASRModel(nn.Module):
    """Wav2Vec2-based ASR model with accent robustness."""
    
    def __init__(self, config: DictConfig):
        """
        Initialize the Wav2Vec2 ASR model.
        
        Args:
            config: Model configuration.
        """
        super().__init__()
        self.config = config
        self.device = get_device()
        
        # Load pre-trained model and processor
        self.processor = Wav2Vec2Processor.from_pretrained(
            config.architecture.model_name
        )
        
        self.model = Wav2Vec2ForCTC.from_pretrained(
            config.architecture.model_name,
            ctc_loss_reduction="mean",
            pad_token_id=self.processor.tokenizer.pad_token_id,
            vocab_size=len(self.processor.tokenizer)
        )
        
        # Configure model based on config
        self._configure_model()
        
        logger.info(f"Initialized Wav2Vec2 model: {config.architecture.model_name}")
        logger.info(f"Vocabulary size: {len(self.processor.tokenizer)}")
    
    def _configure_model(self) -> None:
        """Configure model based on configuration."""
        arch_config = self.config.architecture
        
        # Freeze components if specified
        if arch_config.freeze_feature_extractor:
            self.model.freeze_feature_extractor()
            logger.info("Frozen feature extractor")
        
        if arch_config.freeze_feature_encoder:
            self.model.freeze_feature_encoder()
            logger.info("Frozen feature encoder")
        
        # Configure SpecAugment
        if arch_config.apply_spec_augment:
            self.model.config.apply_spec_augment = True
            self.model.config.mask_time_prob = arch_config.mask_time_prob
            self.model.config.mask_time_length = arch_config.mask_time_length
            self.model.config.mask_feature_prob = arch_config.mask_feature_prob
            self.model.config.mask_feature_length = arch_config.mask_feature_length
            logger.info("Applied SpecAugment configuration")
    
    def forward(
        self,
        input_values: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the model.
        
        Args:
            input_values: Input audio features.
            attention_mask: Attention mask.
            labels: Target labels for training.
            
        Returns:
            Model outputs dictionary.
        """
        outputs = self.model(
            input_values=input_values,
            attention_mask=attention_mask,
            labels=labels
        )
        
        return outputs
    
    def encode(self, input_values: torch.Tensor) -> torch.Tensor:
        """
        Encode input audio to hidden states.
        
        Args:
            input_values: Input audio features.
            
        Returns:
            Hidden states.
        """
        with torch.no_grad():
            outputs = self.model.wav2vec2(input_values)
            hidden_states = outputs.last_hidden_state
        
        return hidden_states
    
    def decode(
        self,
        logits: torch.Tensor,
        beam_size: int = 1,
        length_penalty: float = 1.0,
        early_stopping: bool = True
    ) -> List[str]:
        """
        Decode logits to text using beam search or greedy decoding.
        
        Args:
            logits: Model output logits.
            beam_size: Beam size for beam search.
            length_penalty: Length penalty for beam search.
            early_stopping: Whether to use early stopping.
            
        Returns:
            List of decoded transcriptions.
        """
        if beam_size == 1:
            # Greedy decoding
            predicted_ids = torch.argmax(logits, dim=-1)
            transcriptions = self.processor.batch_decode(
                predicted_ids,
                skip_special_tokens=True
            )
        else:
            # Beam search decoding
            transcriptions = self.processor.batch_decode(
                logits,
                beam_size=beam_size,
                length_penalty=length_penalty,
                early_stopping=early_stopping,
                skip_special_tokens=True
            )
        
        return transcriptions
    
    def transcribe(
        self,
        audio: Union[torch.Tensor, str],
        sample_rate: int = 16000
    ) -> str:
        """
        Transcribe audio to text.
        
        Args:
            audio: Audio tensor or file path.
            sample_rate: Sample rate of audio.
            
        Returns:
            Transcribed text.
        """
        # Process audio input
        if isinstance(audio, str):
            # Load audio file
            from src.utils.audio import load_audio
            waveform, sr = load_audio(audio, sample_rate=sample_rate)
        else:
            waveform = audio
            sr = sample_rate
        
        # Ensure correct sample rate
        if sr != 16000:
            from torchaudio.transforms import Resample
            resampler = Resample(sr, 16000)
            waveform = resampler(waveform)
        
        # Process with model
        inputs = self.processor(
            waveform,
            sampling_rate=16000,
            return_tensors="pt",
            padding=True
        )
        
        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Get model outputs
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
        
        # Decode
        transcription = self.decode(logits)[0]
        
        return transcription
    
    def get_accent_embeddings(
        self,
        input_values: torch.Tensor,
        accent_labels: Optional[List[str]] = None
    ) -> torch.Tensor:
        """
        Extract accent-aware embeddings from the model.
        
        Args:
            input_values: Input audio features.
            accent_labels: Optional accent labels for supervised learning.
            
        Returns:
            Accent-aware embeddings.
        """
        # Get hidden states from Wav2Vec2
        hidden_states = self.encode(input_values)
        
        # Pool over time dimension (mean pooling)
        embeddings = torch.mean(hidden_states, dim=1)
        
        return embeddings
    
    def compute_accent_loss(
        self,
        embeddings: torch.Tensor,
        accent_labels: List[str]
    ) -> torch.Tensor:
        """
        Compute accent classification loss for multi-task learning.
        
        Args:
            embeddings: Accent embeddings.
            accent_labels: Accent labels.
            
        Returns:
            Accent classification loss.
        """
        # This is a placeholder for accent classification
        # In a real implementation, you would add an accent classifier head
        # and compute cross-entropy loss
        
        # For now, return zero loss
        return torch.tensor(0.0, device=embeddings.device)
    
    def save_model(self, path: str) -> None:
        """
        Save model and processor.
        
        Args:
            path: Path to save the model.
        """
        self.model.save_pretrained(path)
        self.processor.save_pretrained(path)
        logger.info(f"Model saved to {path}")
    
    @classmethod
    def from_pretrained(cls, path: str, config: Optional[DictConfig] = None) -> "Wav2Vec2ASRModel":
        """
        Load model from pretrained checkpoint.
        
        Args:
            path: Path to pretrained model.
            config: Optional configuration.
            
        Returns:
            Loaded model instance.
        """
        if config is None:
            # Create default config
            from omegaconf import OmegaConf
            config = OmegaConf.create({
                "architecture": {
                    "model_name": path,
                    "freeze_feature_extractor": False,
                    "freeze_feature_encoder": False,
                    "apply_spec_augment": True,
                    "mask_time_prob": 0.05,
                    "mask_time_length": 10,
                    "mask_feature_prob": 0.0,
                    "mask_feature_length": 64
                }
            })
        
        model = cls(config)
        
        # Load pretrained weights
        model.model = Wav2Vec2ForCTC.from_pretrained(path)
        model.processor = Wav2Vec2Processor.from_pretrained(path)
        
        logger.info(f"Loaded pretrained model from {path}")
        return model
    
    def get_model_info(self) -> Dict[str, Union[int, str]]:
        """
        Get model information.
        
        Returns:
            Dictionary with model information.
        """
        from src.utils.device import count_parameters, get_model_size
        
        return {
            "model_name": self.config.architecture.model_name,
            "vocab_size": len(self.processor.tokenizer),
            "parameters": count_parameters(self.model),
            "model_size": get_model_size(self.model),
            "device": str(self.device)
        }
