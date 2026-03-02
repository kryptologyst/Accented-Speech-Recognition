# Accented Speech Recognition

Research-focused automatic speech recognition (ASR) system designed for robust performance across diverse accents. This project provides state-of-the-art models, comprehensive evaluation metrics, and interactive demos for accented speech recognition.

## ⚠️ PRIVACY DISCLAIMER

**This is a research and educational project. It is NOT intended for biometric identification or production use.**

- **Research Only**: This system is designed for academic research and educational purposes
- **No Biometric Use**: Do not use for voice identification, verification, or any biometric applications
- **Privacy Preserving**: No raw personal information is logged or stored
- **Voice Cloning Prohibited**: Misuse for voice cloning or impersonation is strictly prohibited
- **Ethical Use**: Use responsibly and in accordance with applicable laws and ethical guidelines

## Features

- **Modern ASR Models**: Wav2Vec2, Conformer, CTC/Attention hybrid architectures
- **Accent Robustness**: Specialized training and evaluation for diverse accents
- **Comprehensive Metrics**: WER, CER, accent-specific performance analysis
- **Interactive Demo**: Streamlit/Gradio interface for real-time testing
- **Reproducible Research**: Hydra configuration system and deterministic training
- **Production Ready**: Clean code, type hints, comprehensive testing

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/kryptologyst/Accented-Speech-Recognition.git
cd Accented-Speech-Recognition

# Install dependencies
pip install -e .

# For development
pip install -e ".[dev]"
```

### Basic Usage

```python
from src.models import AccentedASRModel
from src.data import AccentDataset

# Load model
model = AccentedASRModel.from_pretrained("facebook/wav2vec2-large-960h")

# Load dataset
dataset = AccentDataset("data/processed/accented_speech")

# Train model
model.train(dataset, epochs=5)

# Evaluate
results = model.evaluate(dataset.test_split)
print(f"WER: {results['wer']:.2%}")
```

### Interactive Demo

```bash
# Launch Streamlit demo
streamlit run demo/app.py

# Or launch Gradio demo
python demo/gradio_app.py
```

## Project Structure

```
accented-speech-recognition/
├── src/                    # Source code
│   ├── models/            # ASR model implementations
│   ├── data/              # Data loading and preprocessing
│   ├── features/          # Feature extraction
│   ├── losses/            # Loss functions
│   ├── metrics/           # Evaluation metrics
│   ├── decoding/          # Decoding algorithms
│   ├── train/             # Training scripts
│   ├── eval/              # Evaluation scripts
│   └── utils/             # Utility functions
├── configs/               # Hydra configuration files
├── data/                  # Data directory
│   ├── raw/              # Raw audio files
│   └── processed/        # Processed datasets
├── scripts/              # Training and evaluation scripts
├── notebooks/            # Jupyter notebooks for analysis
├── tests/                # Unit tests
├── assets/               # Generated artifacts (plots, audio)
├── demo/                 # Interactive demos
└── docs/                 # Documentation
```

## Dataset Schema

The system expects audio data organized as follows:

```
data/
├── raw/
│   ├── accent_1/
│   │   ├── speaker_1/
│   │   │   ├── audio_1.wav
│   │   │   └── audio_2.wav
│   │   └── speaker_2/
│   └── accent_2/
└── processed/
    ├── meta.csv          # Metadata: id, path, accent, speaker, text, split
    └── annotations.json  # Optional: detailed annotations
```

### Metadata Format (meta.csv)

| Column | Description | Example |
|--------|-------------|---------|
| id | Unique identifier | "accent1_spk1_001" |
| path | Relative path to audio | "raw/accent_1/speaker_1/audio_1.wav" |
| accent | Accent label | "british", "american", "australian" |
| speaker | Speaker identifier | "spk_001" |
| text | Transcription | "Hello, how are you today?" |
| split | Data split | "train", "val", "test" |
| duration | Audio duration (seconds) | 3.45 |

## Training

### Configuration

Training is configured using Hydra configs in `configs/`:

```bash
# Train with default config
python scripts/train.py

# Train with custom config
python scripts/train.py model=conformer data.batch_size=32

# Train with specific accent focus
python scripts/train.py data.accent_filter=british
```

### Available Models

- **Wav2Vec2**: Pre-trained transformer-based ASR
- **Conformer**: Convolution-augmented transformer
- **CTC/Attention Hybrid**: Traditional ASR architecture

### Training Features

- **SpecAugment**: Time and frequency masking
- **Speed Perturbation**: Audio speed variation
- **Accent-Aware Sampling**: Balanced training across accents
- **Gradient Accumulation**: Efficient training on limited hardware

## Evaluation

### Metrics

- **WER (Word Error Rate)**: Primary ASR metric
- **CER (Character Error Rate)**: Character-level accuracy
- **Accent-Specific Performance**: Per-accent breakdown
- **Confidence Calibration**: Model uncertainty analysis

### Running Evaluation

```bash
# Evaluate on test set
python scripts/evaluate.py model.checkpoint=best_model.pt

# Evaluate specific accents
python scripts/evaluate.py data.accent_filter=british

# Generate detailed report
python scripts/evaluate.py output.detailed_report=true
```

## Demo Applications

### Streamlit Demo

Interactive web interface with:
- Audio upload/recording
- Real-time transcription
- Accent detection
- Confidence visualization
- Error analysis

### Gradio Demo

Simple interface for quick testing:
- Drag-and-drop audio
- Instant transcription
- Multiple model comparison

## API Usage

### FastAPI Server

```bash
# Start API server
python scripts/serve.py

# API endpoints:
# POST /transcribe - Transcribe audio
# GET /models - List available models
# POST /evaluate - Evaluate model performance
```

### Python API

```python
from src.api import ASRClient

client = ASRClient("http://localhost:8000")
result = client.transcribe("path/to/audio.wav")
print(result.transcription)
```

## Development

### Code Quality

```bash
# Format code
black src/ tests/

# Lint code
ruff check src/ tests/

# Type checking
mypy src/

# Run tests
pytest tests/
```

### Pre-commit Hooks

```bash
# Install pre-commit hooks
pre-commit install

# Run hooks manually
pre-commit run --all-files
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass
6. Submit a pull request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this project in your research, please cite:

```bibtex
@software{accented_speech_recognition,
  title={Accented Speech Recognition},
  author={Kryptologyst},
  year={2026},
  url={https://github.com/kryptologyst/Accented-Speech-Recognition}
}
```

## Acknowledgments

- Hugging Face Transformers for pre-trained models
- Mozilla Common Voice for open datasets
- The ASR research community for foundational work

## Limitations

- **Research Demo**: Not suitable for production use
- **Limited Accents**: Performance may vary across accent types
- **Computational Requirements**: Requires GPU for optimal performance
- **Data Dependency**: Performance depends on training data quality

## Support

For questions and support:
- Create an issue on GitHub
- Check the documentation in `docs/`
- Review the example notebooks in `notebooks/`
# Accented-Speech-Recognition
