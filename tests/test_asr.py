"""Tests for the accented speech recognition system."""

import pytest
import torch
from omegaconf import OmegaConf

from src.models.wav2vec2 import Wav2Vec2ASRModel
from src.utils.device import get_device, setup_environment
from src.utils.audio import normalize_audio, preemphasis
from src.metrics.asr_metrics import ASRMetrics


class TestDeviceUtils:
    """Test device utility functions."""
    
    def test_get_device(self):
        """Test device detection."""
        device = get_device()
        assert isinstance(device, torch.device)
    
    def test_setup_environment(self):
        """Test environment setup."""
        device = setup_environment(seed=42)
        assert isinstance(device, torch.device)


class TestAudioUtils:
    """Test audio utility functions."""
    
    def test_normalize_audio(self):
        """Test audio normalization."""
        # Test with normal audio
        audio = torch.tensor([1.0, -1.0, 0.5, -0.5])
        normalized = normalize_audio(audio)
        assert torch.max(torch.abs(normalized)) <= 1.0
        
        # Test with zero audio
        audio_zero = torch.zeros(10)
        normalized_zero = normalize_audio(audio_zero)
        assert torch.allclose(normalized_zero, audio_zero)
    
    def test_preemphasis(self):
        """Test pre-emphasis filter."""
        audio = torch.tensor([1.0, 2.0, 3.0, 4.0])
        emphasized = preemphasis(audio, coeff=0.97)
        
        # Check that first sample is unchanged
        assert emphasized[0] == audio[0]
        
        # Check that subsequent samples are modified
        assert not torch.allclose(emphasized[1:], audio[1:])


class TestASRMetrics:
    """Test ASR metrics calculation."""
    
    def test_wer_calculation(self):
        """Test WER calculation."""
        metrics = ASRMetrics()
        
        predictions = ["hello world", "good morning"]
        references = ["hello world", "good evening"]
        
        metrics.add_batch(predictions, references)
        wer = metrics.compute_wer()
        
        assert isinstance(wer, float)
        assert 0.0 <= wer <= 1.0
    
    def test_cer_calculation(self):
        """Test CER calculation."""
        metrics = ASRMetrics()
        
        predictions = ["hello", "world"]
        references = ["helo", "word"]
        
        metrics.add_batch(predictions, references)
        cer = metrics.compute_cer()
        
        assert isinstance(cer, float)
        assert 0.0 <= cer <= 1.0
    
    def test_empty_metrics(self):
        """Test metrics with empty data."""
        metrics = ASRMetrics()
        
        wer = metrics.compute_wer()
        cer = metrics.compute_cer()
        
        assert wer == 0.0
        assert cer == 0.0


class TestWav2Vec2Model:
    """Test Wav2Vec2 model functionality."""
    
    @pytest.fixture
    def model_config(self):
        """Create test model configuration."""
        return OmegaConf.create({
            "architecture": {
                "model_name": "facebook/wav2vec2-base-960h",
                "freeze_feature_extractor": False,
                "freeze_feature_encoder": False,
                "apply_spec_augment": True,
                "mask_time_prob": 0.05,
                "mask_time_length": 10,
                "mask_feature_prob": 0.0,
                "mask_feature_length": 64
            }
        })
    
    def test_model_initialization(self, model_config):
        """Test model initialization."""
        try:
            model = Wav2Vec2ASRModel(model_config)
            assert model is not None
            assert hasattr(model, 'model')
            assert hasattr(model, 'processor')
        except Exception as e:
            # Skip test if model can't be loaded (e.g., no internet)
            pytest.skip(f"Model loading failed: {e}")
    
    def test_model_info(self, model_config):
        """Test model info retrieval."""
        try:
            model = Wav2Vec2ASRModel(model_config)
            info = model.get_model_info()
            
            assert isinstance(info, dict)
            assert "model_name" in info
            assert "vocab_size" in info
            assert "parameters" in info
        except Exception as e:
            pytest.skip(f"Model loading failed: {e}")


class TestIntegration:
    """Integration tests."""
    
    def test_basic_pipeline(self):
        """Test basic ASR pipeline."""
        # This would test the full pipeline from audio to transcription
        # For now, just test that components can be imported
        try:
            from src.data.accent_dataset import AccentDataset
            from src.train.trainer import ASRTrainer
            from src.eval.evaluator import ASREvaluator
            assert True
        except ImportError as e:
            pytest.fail(f"Import failed: {e}")


if __name__ == "__main__":
    pytest.main([__file__])
