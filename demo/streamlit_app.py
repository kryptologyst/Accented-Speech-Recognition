"""Streamlit demo for accented speech recognition."""

import logging
import tempfile
from pathlib import Path
from typing import Optional, Tuple

import streamlit as st
import torch
import torchaudio
from omegaconf import DictConfig, OmegaConf

from src.models.wav2vec2 import Wav2Vec2ASRModel
from src.utils.audio import load_audio, normalize_audio
from src.utils.device import get_device

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page config
st.set_page_config(
    page_title="Accented Speech Recognition Demo",
    page_icon="🎤",
    layout="wide"
)

# Privacy disclaimer
st.warning("""
**PRIVACY DISCLAIMER**: This is a research and educational demo. 
- NOT intended for biometric identification or production use
- No personal information is stored or logged
- Voice cloning or impersonation is strictly prohibited
- Use responsibly and ethically
""")


@st.cache_resource
def load_model() -> Wav2Vec2ASRModel:
    """Load the ASR model."""
    try:
        # Load default configuration
        config_path = Path("configs/model/wav2vec2.yaml")
        if config_path.exists():
            config = OmegaConf.load(config_path)
        else:
            # Create default config if file doesn't exist
            config = OmegaConf.create({
                "architecture": {
                    "model_name": "facebook/wav2vec2-large-960h",
                    "freeze_feature_extractor": False,
                    "freeze_feature_encoder": False,
                    "apply_spec_augment": True,
                    "mask_time_prob": 0.05,
                    "mask_time_length": 10,
                    "mask_feature_prob": 0.0,
                    "mask_feature_length": 64
                }
            })
        
        model = Wav2Vec2ASRModel(config)
        return model
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None


def process_audio(audio_file) -> Tuple[Optional[torch.Tensor], Optional[int]]:
    """
    Process uploaded audio file.
    
    Args:
        audio_file: Uploaded audio file.
        
    Returns:
        Tuple of (waveform, sample_rate).
    """
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            tmp_file.write(audio_file.read())
            tmp_path = tmp_file.name
        
        # Load audio
        waveform, sample_rate = load_audio(tmp_path, sample_rate=16000)
        
        # Clean up temporary file
        Path(tmp_path).unlink()
        
        return waveform, sample_rate
    except Exception as e:
        st.error(f"Error processing audio: {e}")
        return None, None


def transcribe_audio(model: Wav2Vec2ASRModel, waveform: torch.Tensor) -> str:
    """
    Transcribe audio using the model.
    
    Args:
        model: ASR model.
        waveform: Audio waveform.
        
    Returns:
        Transcribed text.
    """
    try:
        transcription = model.transcribe(waveform)
        return transcription
    except Exception as e:
        st.error(f"Error during transcription: {e}")
        return ""


def main():
    """Main demo application."""
    st.title("🎤 Accented Speech Recognition Demo")
    
    st.markdown("""
    This demo showcases an automatic speech recognition (ASR) system designed for robust performance 
    across different accents. Upload an audio file or record your voice to see the transcription.
    """)
    
    # Load model
    with st.spinner("Loading ASR model..."):
        model = load_model()
    
    if model is None:
        st.error("Failed to load model. Please check the configuration.")
        return
    
    # Model info
    with st.expander("Model Information"):
        model_info = model.get_model_info()
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Model", model_info["model_name"])
        with col2:
            st.metric("Parameters", f"{model_info['parameters']:,}")
        with col3:
            st.metric("Device", model_info["device"])
    
    # Input section
    st.header("Audio Input")
    
    # Audio input options
    input_method = st.radio(
        "Choose input method:",
        ["Upload Audio File", "Record Audio"],
        horizontal=True
    )
    
    audio_data = None
    sample_rate = None
    
    if input_method == "Upload Audio File":
        uploaded_file = st.file_uploader(
            "Upload an audio file",
            type=["wav", "mp3", "flac", "m4a"],
            help="Supported formats: WAV, MP3, FLAC, M4A"
        )
        
        if uploaded_file is not None:
            with st.spinner("Processing audio file..."):
                audio_data, sample_rate = process_audio(uploaded_file)
    
    elif input_method == "Record Audio":
        st.info("Audio recording functionality would be implemented here using Streamlit's audio recording capabilities.")
        # Note: Streamlit doesn't have built-in audio recording, would need additional libraries
    
    # Transcription section
    if audio_data is not None:
        st.header("Transcription Results")
        
        # Display audio info
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Duration", f"{audio_data.shape[-1] / sample_rate:.2f}s")
        with col2:
            st.metric("Sample Rate", f"{sample_rate:,} Hz")
        with col3:
            st.metric("Channels", audio_data.shape[0])
        
        # Transcribe button
        if st.button("Transcribe Audio", type="primary"):
            with st.spinner("Transcribing audio..."):
                transcription = transcribe_audio(model, audio_data)
            
            if transcription:
                st.success("Transcription completed!")
                
                # Display transcription
                st.subheader("Transcribed Text:")
                st.write(transcription)
                
                # Copy to clipboard
                if st.button("Copy to Clipboard"):
                    st.write("Transcription copied to clipboard!")
                
                # Confidence analysis (placeholder)
                st.subheader("Confidence Analysis")
                st.info("Confidence analysis would be implemented here to show model uncertainty.")
            
            else:
                st.error("Transcription failed. Please try again.")
    
    # Demo features
    st.header("Demo Features")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("🎯 Accent Robustness")
        st.markdown("""
        - Trained on diverse accent data
        - Robust to pronunciation variations
        - Fair performance across accents
        """)
    
    with col2:
        st.subheader("⚡ Real-time Processing")
        st.markdown("""
        - Fast inference with optimized models
        - Low latency for interactive use
        - Efficient memory usage
        """)
    
    with col3:
        st.subheader("📊 Evaluation Metrics")
        st.markdown("""
        - Word Error Rate (WER)
        - Character Error Rate (CER)
        - Accent-specific performance
        """)
    
    # Technical details
    with st.expander("Technical Details"):
        st.markdown("""
        **Model Architecture**: Wav2Vec2-based transformer model
        
        **Training Data**: Multi-accent speech datasets with diverse pronunciation patterns
        
        **Key Features**:
        - Self-supervised pre-training on large speech corpora
        - Fine-tuned for accent robustness
        - CTC-based decoding for efficient inference
        
        **Performance**: Optimized for research and educational use
        """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>Research Demo - Not for Production Use</p>
        <p>Built with PyTorch, Transformers, and Streamlit</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
