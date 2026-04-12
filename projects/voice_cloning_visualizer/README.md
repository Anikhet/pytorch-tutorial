# Voice Cloning Feature Visualizer

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)
![License](https://img.shields.io/badge/License-MIT-green)
![GPU](https://img.shields.io/badge/GPU-Optional-yellow)

Interactive visualization of audio features used in voice cloning and speech synthesis systems.

## Learning Objectives

By completing this tutorial, you will learn:

- **Mel-Spectrograms**: Understand the primary representation for neural TTS models
- **MFCCs**: Learn cepstral coefficients for speaker identification
- **Pitch Extraction (F0)**: Extract fundamental frequency for prosody analysis
- **Audio Feature Engineering**: Compute spectral centroid, bandwidth, and rolloff
- **Time-Frequency Analysis**: Master STFT and its applications
- **Voice Characteristics**: Understand what makes voices unique and how to capture it

## Overview

This project visualizes the key features extracted from speech audio that are used in modern voice cloning systems like:
- **Tacotron** - Text-to-speech with mel-spectrograms
- **VITS** - End-to-end speech synthesis
- **YourTTS** - Multi-speaker voice cloning
- **Tortoise TTS** - High-quality voice cloning

## Features Visualized

| Feature | Description | Use in Voice Cloning |
|---------|-------------|---------------------|
| **Mel-Spectrogram** | Time-frequency representation on mel scale | Primary target for neural TTS models |
| **MFCCs** | Spectral envelope coefficients | Speaker identification, verification |
| **Pitch (F0)** | Fundamental frequency contour | Prosody transfer, intonation |
| **Energy** | RMS amplitude envelope | Speech activity, emphasis |
| **Spectral Centroid** | "Brightness" of sound | Voice timbre characteristics |
| **Spectral Bandwidth** | Frequency spread | Voice quality features |
| **Spectral Rolloff** | High-frequency content | Voice characteristics |

## Installation

```bash
cd voice_cloning_visualizer
pip install -r requirements.txt
```

### Requirements
- Python 3.8+
- librosa (audio processing)
- pygame (real-time visualization)
- matplotlib (static plots)
- numpy, scipy

## Usage

### Real-time Visualization (Pygame)

```bash
# Visualize synthetic speech-like audio
python demo.py

# Visualize your own audio file
python demo.py --file your_voice.wav

# Adjust duration of synthetic audio
python demo.py --duration 5.0
```

**Controls:**
- `SPACE` - Play/Pause animation
- `R` - Reset to beginning
- `Q` or `ESC` - Quit

### Static Visualization (Matplotlib)

```bash
# Generate static plot
python demo.py --static

# Save to specific file
python demo.py --static --output my_analysis.png
```

### Learn About Features

```bash
# Print educational explanation
python demo.py --explain
```

### Generate Sample Files

```bash
# Create sample WAV files
python demo.py --generate
```

## Project Structure

```
voice_cloning_visualizer/
├── README.md              # This file
├── requirements.txt       # Python dependencies
├── demo.py               # Main application
├── audio_processor.py    # Audio loading, mel-spectrogram computation
├── feature_extractor.py  # Voice feature extraction
├── visualizer.py         # Pygame real-time & Matplotlib static visualization
└── samples/              # Generated sample audio files
```

## How It Works

### 1. Audio Processing ([audio_processor.py](audio_processor.py))

Handles audio loading and basic transformations:
- Load WAV/MP3 files with resampling
- Compute Short-Time Fourier Transform (STFT)
- Generate mel-filterbank spectrograms
- Convert to log scale (dB)

### 2. Feature Extraction ([feature_extractor.py](feature_extractor.py))

Extracts voice-specific features:
- **MFCCs**: Using DCT on log mel-spectrogram
- **Pitch (F0)**: Using pYIN algorithm (librosa) or autocorrelation fallback
- **Energy**: Frame-wise RMS computation
- **Spectral features**: Centroid, bandwidth, rolloff, ZCR

### 3. Visualization ([visualizer.py](visualizer.py))

Two visualization modes:

**Real-time (Pygame)**:
- Scrolling spectrogram display
- Animated playback position
- Interactive controls

**Static (Matplotlib)**:
- Publication-quality multi-panel figure
- All features in one view
- Save as PNG/PDF

## Understanding the Features

### Mel-Spectrogram

The mel-spectrogram is the most important feature for neural voice synthesis:

```
Audio → STFT → Power Spectrum → Mel Filterbank → Log Scale → Mel-Spectrogram
```

Modern TTS systems (Tacotron, VITS) predict mel-spectrograms from text, then use a vocoder (HiFi-GAN, WaveGlow) to convert back to audio.

### MFCCs

MFCCs capture the spectral envelope shape:

```
Log Mel-Spectrogram → DCT → First 13 Coefficients → MFCCs
```

- **C0**: Related to overall energy
- **C1-C12**: Capture spectral shape at different scales
- **Delta/Delta-Delta**: Temporal dynamics

### Pitch (F0)

The fundamental frequency determines perceived pitch:
- **Male voices**: ~85-180 Hz
- **Female voices**: ~165-255 Hz
- **Children**: ~250-400 Hz

Voice cloning must capture:
- Average pitch (speaker identity)
- Pitch contour (intonation, prosody)
- Pitch range (expressiveness)

## Example Output

When you run the visualizer, you'll see:

```
╔══════════════════════════════════════════════════════════════════╗
║              Voice Cloning Feature Visualizer                     ║
╚══════════════════════════════════════════════════════════════════╝

Generating synthetic speech-like audio...

EXTRACTED FEATURES SUMMARY
==================================================

📊 Basic Info:
   Duration: 4.00 seconds
   Sample Rate: 22050 Hz
   Samples: 88,200

🎵 Spectral Features:
   Mel-spectrogram: (80, 173) (bins × frames)
   Log Mel range: [-80.0, 0.0] dB

🔊 MFCCs:
   Shape: (13, 173) (coefficients × frames)

🎤 Pitch (F0):
   Range: [145.2, 158.3] Hz
   Mean: 151.4 Hz
   Voiced frames: 156/173 (90.2%)
```

## Concepts from PyTorch Tutorials

This project builds on concepts from:
- **Notebook 9**: Audio processing with spectrograms
- **Notebook 10**: Speech feature extraction
- **Notebook 12**: Generative models (relates to voice synthesis)

## Advanced Usage

### Processing Multiple Files

```python
from audio_processor import AudioProcessor
from feature_extractor import VoiceFeatureExtractor
from visualizer import StaticVisualizer

processor = AudioProcessor()
extractor = VoiceFeatureExtractor()
viz = StaticVisualizer()

for audio_file in audio_files:
    audio, sr = processor.load_audio(audio_file)
    features = extractor.extract_all_features(audio)
    viz.plot_all_features(features, save_path=f"{audio_file}_analysis.png")
```

### Custom Feature Extraction

```python
from feature_extractor import VoiceFeatureExtractor

extractor = VoiceFeatureExtractor()

# Extract specific features
mfccs = extractor.extract_mfccs(audio, n_mfcc=20)
pitch, times = extractor.extract_pitch(audio, fmin=75, fmax=300)
energy = extractor.extract_energy(audio)
```

## Hardware Requirements

| Device | Feature Extraction | Visualization |
|--------|-------------------|---------------|
| CPU | ~1 second/file | Real-time |
| M1/M2 Mac | ~0.5 seconds/file | Real-time |

**Minimum Requirements**:
- RAM: 4 GB
- No GPU required
- Audio output device (optional, for playback)

## Troubleshooting

### "librosa import error"

Install audio dependencies:
```bash
# macOS
brew install ffmpeg

# Ubuntu/Debian
sudo apt-get install ffmpeg libsndfile1

# Then reinstall librosa
pip install --upgrade librosa soundfile
```

### "No sound output"

Check audio device:
```bash
# Test pygame audio
python -c "import pygame; pygame.mixer.init(); print('Audio OK')"
```

### "FileNotFoundError for audio file"

Supported formats: WAV, MP3, FLAC, OGG
```bash
# Convert using ffmpeg
ffmpeg -i input.m4a -ar 22050 output.wav
```

### "Pitch extraction returns NaN"

Audio may be too quiet or have no voiced content:
- Check audio levels (should not be silent)
- Try different `fmin`/`fmax` for pitch range
- Ensure audio contains speech (not just noise)

### "Pygame visualization frozen"

Close and restart:
- Press `Q` or `ESC` to exit
- Kill Python process if needed: `pkill -f demo.py`

## References

- [Tacotron 2](https://arxiv.org/abs/1712.05884) - Natural TTS Synthesis
- [VITS](https://arxiv.org/abs/2106.06103) - End-to-End Speech Synthesis
- [librosa](https://librosa.org/) - Audio analysis library
- [Speech Processing Book](https://web.stanford.edu/~jurafsky/slp3/) - Jurafsky & Martin

## License

MIT License - Part of the PyTorch Tutorial series.
