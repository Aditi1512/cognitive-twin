# Personal Cognitive Digital Twin

A real-time mental state monitoring system that uses computer vision and Generative AI to provide personalized cognitive feedback.

<p align="center">
  <img src="docs/assets/banner.png" alt="Cognitive Digital Twin Banner" width="800">
</p>

## Overview

The **Personal Cognitive Digital Twin** is an innovative system that continuously monitors and explains a user's mental state in real-time. Using only a standard webcam, it analyzes facial expressions, eye movements, and head posture to estimate focus levels, fatigue, and cognitive load—then provides natural language feedback powered by Generative AI.

### Key Features

- **Non-Intrusive Monitoring**: Uses only your laptop's webcam—no wearables or special equipment
- **Real-Time Analysis**: Continuous tracking at 15-30 FPS with instant feedback
- **Personalized Learning**: Builds a digital twin that learns your unique cognitive patterns
- **Natural Language Feedback**: GenAI-powered explanations that are easy to understand
- **Privacy-First Design**: All processing happens locally by default
- **Cross-Platform**: Works on Windows, macOS, and Linux

## Quick Start

### Prerequisites

- Python 3.10 or higher
- Webcam (built-in or USB)
- 8GB RAM minimum
- For local GenAI: [Ollama](https://ollama.ai/) installed

### Installation

```bash
# Clone the repository
git clone https://github.com/your-username/cognitive-twin.git
cd cognitive-twin

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Running the Application

```bash
# Start the application
python -m src.main

# Or with Streamlit UI
streamlit run src/ui/dashboard.py
```

### First Run

1. **Grant Camera Permission**: Allow webcam access when prompted
2. **Consent**: Review and accept the privacy terms
3. **Calibration**: Follow the brief calibration procedure (optional)
4. **Start Monitoring**: Click "Start Session" to begin

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     User Interface Layer                        │
├─────────────────────────────────────────────────────────────────┤
│                    Explanation Layer (GenAI)                    │
├─────────────────────────────────────────────────────────────────┤
│                   Mental State Estimation Layer                 │
├─────────────────────────────────────────────────────────────────┤
│                   Cognitive Modeling Layer                      │
├─────────────────────────────────────────────────────────────────┤
│                   Feature Extraction Layer                      │
├─────────────────────────────────────────────────────────────────┤
│                   Data Acquisition Layer                        │
└─────────────────────────────────────────────────────────────────┘
```

The system processes data through six layers:

1. **Data Acquisition**: Captures webcam video with user consent
2. **Feature Extraction**: Extracts facial landmarks, eye gaze, head pose using MediaPipe
3. **Cognitive Modeling**: Learns temporal patterns using LSTM/GRU networks
4. **Mental State Estimation**: Classifies focus, fatigue, and cognitive load
5. **Explanation Generation**: Creates natural language feedback via LLM
6. **User Interface**: Displays real-time information and notifications

## Project Structure

```
cognitive-twin/
├── src/
│   ├── main.py                 # Application entry point
│   ├── config.py               # Configuration management
│   ├── acquisition/            # Camera and consent
│   ├── features/               # Feature extraction
│   ├── cognitive/              # Digital twin core
│   ├── estimation/             # Mental state classifiers
│   ├── explanation/            # GenAI integration
│   ├── ui/                     # User interface
│   ├── storage/                # Data persistence
│   └── utils/                  # Utilities
├── models/                     # Pre-trained models
├── config/                     # Configuration files
├── data/                       # User data (gitignored)
├── docs/                       # Documentation
│   ├── PRD.md                  # Product Requirements
│   └── architecture/           # Technical docs
├── tests/                      # Test suites
├── requirements.txt
└── README.md
```

## Configuration

Configuration is managed through `config/default.json`:

```json
{
  "acquisition": {
    "camera_id": 0,
    "frame_width": 640,
    "frame_height": 480,
    "target_fps": 30
  },
  "genai": {
    "provider": "openrouter",
    "model": "meta-llama/llama-3.1-8b-instruct:free",
    "api_key": "your-openrouter-api-key",
    "base_url": "https://openrouter.ai/api/v1"
  },
  "privacy": {
    "store_raw_video": false,
    "data_retention_days": 30,
    "local_processing_only": false
  }
}
```

### GenAI Providers

**Option 1: OpenRouter (Free) - Recommended for Getting Started**

OpenRouter provides free access to high-quality models. No credit card required.

1. Sign up at [openrouter.ai](https://openrouter.ai)
2. Get your free API key from the dashboard

Set in config:
```json
{
  "genai": {
    "provider": "openrouter",
    "model": "meta-llama/llama-3.1-8b-instruct:free",
    "api_key": "your-openrouter-api-key",
    "base_url": "https://openrouter.ai/api/v1"
  }
}
```

**Free models available on OpenRouter:**
| Model | Best For |
|-------|----------|
| `meta-llama/llama-3.1-8b-instruct:free` | Balanced quality/speed (recommended) |
| `meta-llama/llama-3.2-3b-instruct:free` | Fastest responses |
| `google/gemma-2-9b-it:free` | Good reasoning |
| `mistralai/mistral-7b-instruct:free` | Efficient, high quality |

**Option 2: Local (Ollama) - Maximum Privacy**

```bash
# Install Ollama and pull a model
ollama pull llama3
```

Set in config:
```json
{
  "genai": {
    "provider": "ollama",
    "model": "llama3"
  }
}
```

**Option 3: Cloud (OpenAI) - Premium Quality**

Set in config:
```json
{
  "genai": {
    "provider": "openai",
    "model": "gpt-4",
    "api_key": "your-api-key-here"
  }
}
```

## Mental States Tracked

| State | Description | Indicators |
|-------|-------------|------------|
| **Focus Level** | Attention and concentration | Gaze stability, head position, blink patterns |
| **Fatigue** | Tiredness and drowsiness | Blink rate, eye closure duration, yawning |
| **Cognitive Load** | Mental effort and strain | Facial tension, micro-expressions, posture |

## Privacy & Data

### What We Collect

| Data Type | Stored | Transmitted |
|-----------|--------|-------------|
| Raw video | Never | Never |
| Facial landmarks | Transient | Never |
| Mental state scores | Locally | Cloud GenAI only (optional) |
| Session history | Locally | Never |

### Your Rights

- **Access**: View all your data in the app
- **Export**: Download your data as JSON/CSV
- **Delete**: Remove all data at any time
- **Control**: Choose local-only or cloud processing

## Development

### Setup Development Environment

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Install pre-commit hooks
pre-commit install
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test file
pytest tests/unit/test_features.py
```

### Code Style

```bash
# Format code
black src tests

# Check linting
ruff check src tests

# Type checking
mypy src
```

## Technology Stack

| Component | Technology |
|-----------|------------|
| Language | Python 3.10+ |
| Computer Vision | OpenCV, MediaPipe |
| Deep Learning | PyTorch |
| Temporal Modeling | LSTM/GRU |
| GenAI (Local) | Ollama |
| GenAI (Free Cloud) | OpenRouter |
| GenAI (Premium Cloud) | OpenAI API |
| UI | Streamlit / PyQt6 |
| Database | SQLite |

## Roadmap

- [x] Core feature extraction
- [x] Temporal cognitive modeling
- [x] Mental state estimation
- [x] GenAI explanation integration
- [x] Streamlit dashboard
- [ ] PyQt desktop application
- [ ] System tray notifications
- [ ] Advanced personalization
- [ ] Team/organization features
- [ ] Mobile companion app

## Research Background

This project is based on extensive literature review across:

- Cognitive load assessment and monitoring
- Vision-based mental state detection
- Temporal deep learning models
- Human digital twin systems
- Explainable and Generative AI

For detailed research context, see the [PRD](docs/PRD.md).

## Contributing

We welcome contributions! Please see our contributing guidelines:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [MediaPipe](https://mediapipe.dev/) for face mesh detection
- [PyTorch](https://pytorch.org/) for deep learning framework
- [Ollama](https://ollama.ai/) for local LLM support
- Research community for foundational work in cognitive monitoring

## Citation

If you use this project in your research, please cite:

```bibtex
@software{cognitive_twin_2026,
  title = {Personal Cognitive Digital Twin},
  author = {Your Name},
  year = {2026},
  url = {https://github.com/your-username/cognitive-twin}
}
```

---

<p align="center">
  Built with care for cognitive wellness
</p>
