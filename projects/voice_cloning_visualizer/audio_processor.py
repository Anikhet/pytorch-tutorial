"""
Audio processing utilities for voice visualization.

Provides:
- Audio loading and preprocessing
- Mel-spectrogram computation
- STFT and spectrogram generation
- Audio synthesis from features
"""

import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass
import warnings

# Suppress librosa warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

try:
    import librosa
    import librosa.display
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False
    print("Warning: librosa not installed. Using basic audio processing.")

try:
    import soundfile as sf
    SOUNDFILE_AVAILABLE = True
except ImportError:
    SOUNDFILE_AVAILABLE = False


@dataclass
class AudioConfig:
    """Configuration for audio processing."""
    sample_rate: int = 22050
    n_fft: int = 2048
    hop_length: int = 512
    n_mels: int = 80
    fmin: float = 0.0
    fmax: float = 8000.0
    win_length: Optional[int] = None
    power: float = 2.0
    ref_db: float = 20.0
    max_db: float = 100.0


class AudioProcessor:
    """
    Audio processor for voice feature extraction.

    Handles loading, preprocessing, and feature extraction from audio.
    """

    def __init__(self, config: Optional[AudioConfig] = None):
        self.config = config or AudioConfig()
        if self.config.win_length is None:
            self.config.win_length = self.config.n_fft

    def load_audio(
        self,
        path: str,
        sr: Optional[int] = None,
        mono: bool = True
    ) -> Tuple[np.ndarray, int]:
        """
        Load audio file.

        Args:
            path: Path to audio file
            sr: Target sample rate (None = use file's rate)
            mono: Convert to mono

        Returns:
            Tuple of (audio array, sample rate)
        """
        target_sr = sr or self.config.sample_rate

        if LIBROSA_AVAILABLE:
            audio, sr = librosa.load(path, sr=target_sr, mono=mono)
        elif SOUNDFILE_AVAILABLE:
            audio, sr = sf.read(path)
            if mono and audio.ndim > 1:
                audio = np.mean(audio, axis=1)
            # Resample if needed
            if sr != target_sr:
                from scipy import signal
                num_samples = int(len(audio) * target_sr / sr)
                audio = signal.resample(audio, num_samples)
                sr = target_sr
        else:
            raise ImportError("Neither librosa nor soundfile is available")

        return audio.astype(np.float32), sr

    def compute_stft(self, audio: np.ndarray) -> np.ndarray:
        """
        Compute Short-Time Fourier Transform.

        Args:
            audio: Audio signal

        Returns:
            Complex STFT matrix [freq_bins, time_frames]
        """
        if LIBROSA_AVAILABLE:
            stft = librosa.stft(
                audio,
                n_fft=self.config.n_fft,
                hop_length=self.config.hop_length,
                win_length=self.config.win_length
            )
        else:
            from scipy import signal
            _, _, stft = signal.stft(
                audio,
                fs=self.config.sample_rate,
                nperseg=self.config.n_fft,
                noverlap=self.config.n_fft - self.config.hop_length
            )

        return stft

    def compute_spectrogram(self, audio: np.ndarray) -> np.ndarray:
        """
        Compute power spectrogram.

        Args:
            audio: Audio signal

        Returns:
            Power spectrogram [freq_bins, time_frames]
        """
        stft = self.compute_stft(audio)
        spectrogram = np.abs(stft) ** self.config.power
        return spectrogram

    def compute_mel_spectrogram(self, audio: np.ndarray) -> np.ndarray:
        """
        Compute mel-spectrogram.

        The mel scale approximates human perception of pitch,
        making it ideal for voice analysis.

        Args:
            audio: Audio signal

        Returns:
            Mel-spectrogram [n_mels, time_frames]
        """
        if LIBROSA_AVAILABLE:
            mel_spec = librosa.feature.melspectrogram(
                y=audio,
                sr=self.config.sample_rate,
                n_fft=self.config.n_fft,
                hop_length=self.config.hop_length,
                n_mels=self.config.n_mels,
                fmin=self.config.fmin,
                fmax=self.config.fmax,
                power=self.config.power
            )
        else:
            # Basic mel-spectrogram without librosa
            spectrogram = self.compute_spectrogram(audio)
            mel_filter = self._create_mel_filterbank()
            mel_spec = mel_filter @ spectrogram

        return mel_spec

    def compute_log_mel_spectrogram(self, audio: np.ndarray) -> np.ndarray:
        """
        Compute log mel-spectrogram (dB scale).

        Log scale compresses dynamic range for better visualization.

        Args:
            audio: Audio signal

        Returns:
            Log mel-spectrogram in dB [n_mels, time_frames]
        """
        mel_spec = self.compute_mel_spectrogram(audio)

        if LIBROSA_AVAILABLE:
            log_mel = librosa.power_to_db(mel_spec, ref=np.max)
        else:
            # Manual conversion to dB
            log_mel = 10 * np.log10(np.maximum(mel_spec, 1e-10))
            log_mel = log_mel - np.max(log_mel)

        return log_mel

    def _create_mel_filterbank(self) -> np.ndarray:
        """Create mel filterbank matrix."""
        if LIBROSA_AVAILABLE:
            return librosa.filters.mel(
                sr=self.config.sample_rate,
                n_fft=self.config.n_fft,
                n_mels=self.config.n_mels,
                fmin=self.config.fmin,
                fmax=self.config.fmax
            )
        else:
            # Simple linear mel filterbank
            n_bins = self.config.n_fft // 2 + 1
            mel_filter = np.zeros((self.config.n_mels, n_bins))
            mel_points = np.linspace(0, n_bins - 1, self.config.n_mels + 2).astype(int)

            for i in range(self.config.n_mels):
                start, center, end = mel_points[i], mel_points[i+1], mel_points[i+2]
                mel_filter[i, start:center] = np.linspace(0, 1, center - start)
                mel_filter[i, center:end] = np.linspace(1, 0, end - center)

            return mel_filter

    def normalize_audio(self, audio: np.ndarray) -> np.ndarray:
        """Normalize audio to [-1, 1] range."""
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            audio = audio / max_val
        return audio

    def trim_silence(
        self,
        audio: np.ndarray,
        top_db: float = 20.0
    ) -> np.ndarray:
        """
        Trim silence from beginning and end of audio.

        Args:
            audio: Audio signal
            top_db: Threshold below reference to consider as silence

        Returns:
            Trimmed audio
        """
        if LIBROSA_AVAILABLE:
            trimmed, _ = librosa.effects.trim(audio, top_db=top_db)
            return trimmed
        else:
            # Simple energy-based trimming
            energy = audio ** 2
            threshold = np.max(energy) * (10 ** (-top_db / 10))
            mask = energy > threshold

            if not np.any(mask):
                return audio

            start = np.argmax(mask)
            end = len(mask) - np.argmax(mask[::-1])
            return audio[start:end]

    def get_duration(self, audio: np.ndarray) -> float:
        """Get audio duration in seconds."""
        return len(audio) / self.config.sample_rate

    def get_time_axis(self, n_frames: int) -> np.ndarray:
        """Get time axis for spectrogram frames."""
        return np.arange(n_frames) * self.config.hop_length / self.config.sample_rate

    def get_frequency_axis(self) -> np.ndarray:
        """Get frequency axis for spectrogram."""
        return np.linspace(0, self.config.sample_rate / 2, self.config.n_fft // 2 + 1)

    def get_mel_frequency_axis(self) -> np.ndarray:
        """Get mel frequency axis."""
        return np.linspace(self.config.fmin, self.config.fmax, self.config.n_mels)


def generate_test_audio(
    duration: float = 2.0,
    sample_rate: int = 22050,
    frequencies: list = None
) -> np.ndarray:
    """
    Generate synthetic test audio with multiple frequencies.

    Args:
        duration: Duration in seconds
        sample_rate: Sample rate
        frequencies: List of frequencies to include

    Returns:
        Synthetic audio signal
    """
    if frequencies is None:
        frequencies = [220, 440, 880]  # A3, A4, A5

    t = np.linspace(0, duration, int(duration * sample_rate), dtype=np.float32)

    audio = np.zeros_like(t)
    for i, freq in enumerate(frequencies):
        # Add harmonic with decreasing amplitude
        amplitude = 0.3 / (i + 1)
        audio += amplitude * np.sin(2 * np.pi * freq * t)

    # Add envelope
    envelope = np.ones_like(t)
    attack = int(0.1 * sample_rate)
    release = int(0.2 * sample_rate)
    envelope[:attack] = np.linspace(0, 1, attack)
    envelope[-release:] = np.linspace(1, 0, release)

    audio = audio * envelope

    return audio


def generate_speech_like_audio(
    duration: float = 2.0,
    sample_rate: int = 22050
) -> np.ndarray:
    """
    Generate speech-like synthetic audio with formants.

    Simulates vowel sounds using formant frequencies.
    """
    t = np.linspace(0, duration, int(duration * sample_rate), dtype=np.float32)

    # Fundamental frequency (pitch) with vibrato
    f0 = 150 + 5 * np.sin(2 * np.pi * 5 * t)

    # Generate glottal pulse train
    phase = np.cumsum(f0) / sample_rate
    glottal = np.sin(2 * np.pi * phase)

    # Formant frequencies for different vowels
    # Simulating transition between vowels
    vowels = [
        (800, 1200, 2500),   # 'a'
        (300, 2300, 3000),   # 'i'
        (400, 800, 2500),    # 'u'
    ]

    audio = np.zeros_like(t)
    segment_len = len(t) // len(vowels)

    for i, (f1, f2, f3) in enumerate(vowels):
        start = i * segment_len
        end = start + segment_len if i < len(vowels) - 1 else len(t)

        segment = glottal[start:end]
        t_seg = t[start:end]

        # Apply formant filters (simplified)
        for f in [f1, f2, f3]:
            bandwidth = f * 0.1
            resonance = np.exp(-bandwidth * t_seg) * np.sin(2 * np.pi * f * t_seg)
            audio[start:end] += resonance * 0.2

    # Normalize
    audio = audio / (np.max(np.abs(audio)) + 1e-6)

    return audio.astype(np.float32)


# Test when run directly
if __name__ == "__main__":
    print("Testing audio processor...")

    # Create processor
    processor = AudioProcessor()
    print(f"Sample rate: {processor.config.sample_rate}")
    print(f"N_FFT: {processor.config.n_fft}")
    print(f"N_mels: {processor.config.n_mels}")

    # Generate test audio
    audio = generate_test_audio(duration=2.0)
    print(f"\nTest audio shape: {audio.shape}")
    print(f"Duration: {processor.get_duration(audio):.2f}s")

    # Compute features
    stft = processor.compute_stft(audio)
    print(f"STFT shape: {stft.shape}")

    spectrogram = processor.compute_spectrogram(audio)
    print(f"Spectrogram shape: {spectrogram.shape}")

    mel_spec = processor.compute_mel_spectrogram(audio)
    print(f"Mel-spectrogram shape: {mel_spec.shape}")

    log_mel = processor.compute_log_mel_spectrogram(audio)
    print(f"Log mel-spectrogram shape: {log_mel.shape}")
    print(f"Log mel range: [{log_mel.min():.1f}, {log_mel.max():.1f}] dB")

    # Test speech-like audio
    speech = generate_speech_like_audio(duration=1.5)
    print(f"\nSpeech-like audio shape: {speech.shape}")

    print("\nAll tests passed!")
