#!/usr/bin/env python3
"""
Voice Cloning Visualizer Demo Application.

Interactive demo showing how voice features are extracted and visualized
in real-time - the same features used in voice cloning systems.

Features visualized:
- Mel-spectrograms (the primary input for modern voice synthesis)
- MFCCs (classic speech recognition features)
- Pitch contour (F0 - fundamental frequency)
- Energy envelope (loudness over time)
- Spectral features (brightness, bandwidth)

Usage:
    python demo.py                    # Use synthetic speech-like audio
    python demo.py --file audio.wav   # Visualize your own audio file
    python demo.py --static           # Generate static plots only
    python demo.py --generate         # Generate sample audio files
"""

import argparse
import sys
import os
import numpy as np
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from audio_processor import (
    AudioProcessor,
    AudioConfig,
    generate_speech_like_audio,
    generate_test_audio
)
from feature_extractor import VoiceFeatureExtractor, VoiceFeatures
from visualizer import VoiceVisualizer, StaticVisualizer, VisualizerConfig


def print_banner():
    """Print welcome banner."""
    banner = """
╔══════════════════════════════════════════════════════════════════╗
║              Voice Cloning Feature Visualizer                     ║
║                                                                   ║
║  Explore the audio features used in modern voice cloning:        ║
║  • Mel-spectrograms  • MFCCs  • Pitch  • Energy  • Spectral      ║
╚══════════════════════════════════════════════════════════════════╝
"""
    print(banner)


def load_or_generate_audio(
    file_path: str = None,
    duration: float = 4.0
) -> tuple:
    """
    Load audio from file or generate synthetic speech-like audio.

    Args:
        file_path: Path to audio file (optional)
        duration: Duration for synthetic audio

    Returns:
        Tuple of (audio array, sample rate, source description)
    """
    processor = AudioProcessor()

    if file_path and os.path.exists(file_path):
        print(f"Loading audio from: {file_path}")
        audio, sr = processor.load_audio(file_path)
        source = f"File: {os.path.basename(file_path)}"
    else:
        if file_path:
            print(f"File not found: {file_path}")
            print("Generating synthetic speech-like audio instead...")
        else:
            print("Generating synthetic speech-like audio...")

        audio = generate_speech_like_audio(
            duration=duration,
            sample_rate=processor.config.sample_rate
        )
        sr = processor.config.sample_rate
        source = "Synthetic speech (formant simulation)"

    return audio, sr, source


def extract_and_print_features(
    audio: np.ndarray,
    extractor: VoiceFeatureExtractor
) -> VoiceFeatures:
    """
    Extract features and print summary.

    Args:
        audio: Audio signal
        extractor: Feature extractor instance

    Returns:
        Extracted VoiceFeatures
    """
    print("\nExtracting voice features...")
    features = extractor.extract_all_features(audio)

    print("\n" + "=" * 50)
    print("EXTRACTED FEATURES SUMMARY")
    print("=" * 50)

    print(f"\n📊 Basic Info:")
    print(f"   Duration: {features.duration:.2f} seconds")
    print(f"   Sample Rate: {features.sample_rate} Hz")
    print(f"   Samples: {len(features.audio):,}")

    print(f"\n🎵 Spectral Features:")
    print(f"   Mel-spectrogram: {features.mel_spectrogram.shape} (bins × frames)")
    print(f"   Log Mel range: [{features.log_mel_spectrogram.min():.1f}, "
          f"{features.log_mel_spectrogram.max():.1f}] dB")

    print(f"\n🔊 MFCCs (Mel-Frequency Cepstral Coefficients):")
    print(f"   Shape: {features.mfccs.shape} (coefficients × frames)")
    print(f"   Delta MFCCs: {features.delta_mfccs.shape}")
    print(f"   Delta-Delta MFCCs: {features.delta2_mfccs.shape}")

    if features.pitch is not None:
        voiced = features.pitch[features.pitch > 0]
        if len(voiced) > 0:
            print(f"\n🎤 Pitch (F0):")
            print(f"   Range: [{voiced.min():.1f}, {voiced.max():.1f}] Hz")
            print(f"   Mean: {np.mean(voiced):.1f} Hz")
            print(f"   Voiced frames: {len(voiced)}/{len(features.pitch)} "
                  f"({100*len(voiced)/len(features.pitch):.1f}%)")

    if features.energy is not None:
        print(f"\n⚡ Energy (RMS):")
        print(f"   Range: [{features.energy.min():.4f}, {features.energy.max():.4f}]")
        print(f"   Mean: {np.mean(features.energy):.4f}")

    if features.spectral_centroid is not None:
        print(f"\n🌈 Spectral Descriptors:")
        print(f"   Centroid: {np.mean(features.spectral_centroid):.1f} Hz (brightness)")
        print(f"   Bandwidth: {np.mean(features.spectral_bandwidth):.1f} Hz (spread)")
        print(f"   Rolloff: {np.mean(features.spectral_rolloff):.1f} Hz (85% energy)")

    if features.zero_crossing_rate is not None:
        print(f"   Zero-crossing rate: {np.mean(features.zero_crossing_rate):.4f}")

    print("\n" + "=" * 50)

    return features


def run_realtime_visualizer(features: VoiceFeatures):
    """
    Run the real-time Pygame visualizer.

    Args:
        features: Extracted voice features
    """
    print("\n🎬 Launching real-time visualizer...")
    print("   Controls:")
    print("   • SPACE - Play/Pause animation")
    print("   • R - Reset to beginning")
    print("   • Q or ESC - Quit")
    print()

    try:
        config = VisualizerConfig(
            width=1200,
            height=800,
            fps=30
        )
        visualizer = VoiceVisualizer(config)
        visualizer.run(features)
    except ImportError as e:
        print(f"Error: {e}")
        print("Install pygame with: pip install pygame")
    except Exception as e:
        print(f"Visualizer error: {e}")


def run_static_visualizer(features: VoiceFeatures, output_path: str = None):
    """
    Generate static matplotlib plots.

    Args:
        features: Extracted voice features
        output_path: Path to save figure (optional)
    """
    print("\n📈 Generating static visualization...")

    static_viz = StaticVisualizer()
    if not static_viz.available:
        print("Error: matplotlib not available")
        print("Install with: pip install matplotlib")
        return

    if output_path is None:
        output_path = "voice_features.png"

    static_viz.plot_all_features(features, save_path=output_path)
    print(f"Saved to: {output_path}")


def generate_sample_files():
    """Generate sample audio files for testing."""
    print("\n🎹 Generating sample audio files...")

    sample_dir = Path(__file__).parent / "samples"
    sample_dir.mkdir(exist_ok=True)

    # Check if we can save audio
    try:
        import soundfile as sf
        can_save = True
    except ImportError:
        can_save = False
        print("Note: soundfile not installed, cannot save WAV files")

    config = AudioConfig()

    samples = [
        ("speech_like", generate_speech_like_audio(3.0, config.sample_rate)),
        ("test_tones", generate_test_audio(3.0, config.sample_rate, [220, 440, 880])),
        ("low_voice", generate_speech_like_audio(3.0, config.sample_rate)),  # Could modify for lower pitch
    ]

    for name, audio in samples:
        if can_save:
            path = sample_dir / f"{name}.wav"
            sf.write(path, audio, config.sample_rate)
            print(f"   Created: {path}")
        else:
            print(f"   Generated: {name} ({len(audio)} samples)")

    print(f"\nSample files saved to: {sample_dir}")


def explain_voice_cloning_features():
    """Print educational explanation of voice features."""
    explanation = """
╔══════════════════════════════════════════════════════════════════╗
║           Understanding Voice Cloning Features                    ║
╚══════════════════════════════════════════════════════════════════╝

🎯 MEL-SPECTROGRAMS
   The most important feature for modern voice synthesis (TTS, voice cloning).
   • Represents frequency content over time
   • Uses mel scale that matches human hearing perception
   • Neural networks (like Tacotron, VITS) predict mel-spectrograms
   • A vocoder then converts mel-spec back to audio

🔊 MFCCs (Mel-Frequency Cepstral Coefficients)
   Classic features for speech recognition and speaker identification.
   • Capture the "spectral envelope" - overall shape of spectrum
   • First coefficient (C0) relates to energy
   • Higher coefficients capture finer spectral details
   • Delta and delta-delta capture temporal dynamics

🎤 PITCH (F0 - Fundamental Frequency)
   Critical for natural-sounding synthesis.
   • Determines the perceived pitch of voice
   • Ranges: ~85-180 Hz (male), ~165-255 Hz (female)
   • Prosody (intonation patterns) conveys emotion and meaning
   • Voice cloning systems must match speaker's pitch patterns

⚡ ENERGY ENVELOPE
   Indicates loudness and speech activity.
   • High energy = voiced speech
   • Low energy = silence, unvoiced sounds, pauses
   • Important for natural rhythm and emphasis

🌈 SPECTRAL FEATURES
   Describe the "color" or "timbre" of the voice.
   • Centroid: "brightness" - higher = brighter voice
   • Bandwidth: spread of frequencies
   • Rolloff: where most energy is concentrated

📚 How These Are Used in Voice Cloning:
   1. Speaker Encoder extracts speaker embedding from reference audio
   2. Text-to-Mel model generates mel-spectrogram conditioned on:
      - Text input
      - Speaker embedding
      - Prosody features (pitch, energy, duration)
   3. Vocoder converts mel-spectrogram to waveform

"""
    print(explanation)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Voice Cloning Feature Visualizer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python demo.py                       # Visualize synthetic audio
  python demo.py --file voice.wav      # Visualize your audio file
  python demo.py --static              # Generate static plots only
  python demo.py --explain             # Learn about voice features
  python demo.py --generate            # Create sample audio files
        """
    )

    parser.add_argument(
        "--file", "-f",
        type=str,
        help="Path to audio file (WAV, MP3, etc.)"
    )
    parser.add_argument(
        "--duration", "-d",
        type=float,
        default=4.0,
        help="Duration for synthetic audio (seconds)"
    )
    parser.add_argument(
        "--static", "-s",
        action="store_true",
        help="Generate static plots instead of real-time visualization"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="voice_features.png",
        help="Output path for static visualization"
    )
    parser.add_argument(
        "--explain", "-e",
        action="store_true",
        help="Print explanation of voice cloning features"
    )
    parser.add_argument(
        "--generate", "-g",
        action="store_true",
        help="Generate sample audio files"
    )

    args = parser.parse_args()

    # Print banner
    print_banner()

    # Handle special modes
    if args.explain:
        explain_voice_cloning_features()
        return

    if args.generate:
        generate_sample_files()
        return

    # Load or generate audio
    audio, sr, source = load_or_generate_audio(
        file_path=args.file,
        duration=args.duration
    )
    print(f"Audio source: {source}")
    print(f"Duration: {len(audio)/sr:.2f}s, Sample rate: {sr} Hz")

    # Create extractor and extract features
    extractor = VoiceFeatureExtractor()
    features = extract_and_print_features(audio, extractor)

    # Run visualizer
    if args.static:
        run_static_visualizer(features, args.output)
    else:
        run_realtime_visualizer(features)

    print("\n✅ Done!")


if __name__ == "__main__":
    main()
