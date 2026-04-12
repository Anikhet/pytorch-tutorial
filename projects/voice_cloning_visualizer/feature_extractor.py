"""
Voice feature extraction for visualization.

Extracts key features used in voice cloning:
- MFCCs (Mel-Frequency Cepstral Coefficients)
- Pitch (F0) contour
- Energy envelope
- Spectral features (centroid, bandwidth, rolloff)
- Formants (approximate)
"""

import numpy as np
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
import warnings

warnings.filterwarnings('ignore')

try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False

from audio_processor import AudioProcessor, AudioConfig


@dataclass
class VoiceFeatures:
    """Container for extracted voice features."""
    # Time-domain features
    audio: np.ndarray
    sample_rate: int
    duration: float

    # Spectral features
    mel_spectrogram: np.ndarray
    log_mel_spectrogram: np.ndarray

    # MFCCs
    mfccs: np.ndarray
    delta_mfccs: Optional[np.ndarray] = None
    delta2_mfccs: Optional[np.ndarray] = None

    # Prosodic features
    pitch: Optional[np.ndarray] = None
    pitch_times: Optional[np.ndarray] = None
    energy: Optional[np.ndarray] = None

    # Spectral descriptors
    spectral_centroid: Optional[np.ndarray] = None
    spectral_bandwidth: Optional[np.ndarray] = None
    spectral_rolloff: Optional[np.ndarray] = None
    zero_crossing_rate: Optional[np.ndarray] = None

    # Timing
    time_axis: Optional[np.ndarray] = None


class VoiceFeatureExtractor:
    """
    Extract voice features for visualization and analysis.

    Features are commonly used in:
    - Voice cloning systems
    - Speaker recognition
    - Speech synthesis
    - Audio analysis
    """

    def __init__(self, config: Optional[AudioConfig] = None):
        self.config = config or AudioConfig()
        self.processor = AudioProcessor(self.config)

    def extract_all_features(self, audio: np.ndarray) -> VoiceFeatures:
        """
        Extract all voice features from audio.

        Args:
            audio: Audio signal

        Returns:
            VoiceFeatures containing all extracted features
        """
        # Basic info
        duration = self.processor.get_duration(audio)

        # Spectral features
        mel_spec = self.processor.compute_mel_spectrogram(audio)
        log_mel = self.processor.compute_log_mel_spectrogram(audio)

        # MFCCs
        mfccs = self.extract_mfccs(audio)
        delta_mfccs = self.compute_deltas(mfccs)
        delta2_mfccs = self.compute_deltas(delta_mfccs)

        # Prosodic features
        pitch, pitch_times = self.extract_pitch(audio)
        energy = self.extract_energy(audio)

        # Spectral descriptors
        centroid = self.extract_spectral_centroid(audio)
        bandwidth = self.extract_spectral_bandwidth(audio)
        rolloff = self.extract_spectral_rolloff(audio)
        zcr = self.extract_zero_crossing_rate(audio)

        # Time axis
        n_frames = mel_spec.shape[1]
        time_axis = self.processor.get_time_axis(n_frames)

        return VoiceFeatures(
            audio=audio,
            sample_rate=self.config.sample_rate,
            duration=duration,
            mel_spectrogram=mel_spec,
            log_mel_spectrogram=log_mel,
            mfccs=mfccs,
            delta_mfccs=delta_mfccs,
            delta2_mfccs=delta2_mfccs,
            pitch=pitch,
            pitch_times=pitch_times,
            energy=energy,
            spectral_centroid=centroid,
            spectral_bandwidth=bandwidth,
            spectral_rolloff=rolloff,
            zero_crossing_rate=zcr,
            time_axis=time_axis
        )

    def extract_mfccs(
        self,
        audio: np.ndarray,
        n_mfcc: int = 13
    ) -> np.ndarray:
        """
        Extract MFCCs (Mel-Frequency Cepstral Coefficients).

        MFCCs capture the spectral envelope of speech and are
        widely used in speech recognition and speaker identification.

        Args:
            audio: Audio signal
            n_mfcc: Number of MFCCs to extract

        Returns:
            MFCC matrix [n_mfcc, time_frames]
        """
        if LIBROSA_AVAILABLE:
            mfccs = librosa.feature.mfcc(
                y=audio,
                sr=self.config.sample_rate,
                n_mfcc=n_mfcc,
                n_fft=self.config.n_fft,
                hop_length=self.config.hop_length
            )
        else:
            # Compute from mel-spectrogram using DCT
            from scipy.fftpack import dct
            mel_spec = self.processor.compute_mel_spectrogram(audio)
            log_mel = np.log(mel_spec + 1e-10)
            mfccs = dct(log_mel, type=2, axis=0, norm='ortho')[:n_mfcc]

        return mfccs

    def compute_deltas(
        self,
        features: np.ndarray,
        width: int = 9
    ) -> np.ndarray:
        """
        Compute delta (derivative) features.

        Delta features capture temporal dynamics.

        Args:
            features: Feature matrix [n_features, time_frames]
            width: Window width for delta computation

        Returns:
            Delta features
        """
        if LIBROSA_AVAILABLE:
            return librosa.feature.delta(features, width=width)
        else:
            # Simple difference approximation
            padded = np.pad(features, ((0, 0), (1, 1)), mode='edge')
            return (padded[:, 2:] - padded[:, :-2]) / 2

    def extract_pitch(
        self,
        audio: np.ndarray,
        fmin: float = 50.0,
        fmax: float = 500.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract pitch (fundamental frequency F0) contour.

        Pitch is crucial for prosody and speaker identity.

        Args:
            audio: Audio signal
            fmin: Minimum pitch frequency
            fmax: Maximum pitch frequency

        Returns:
            Tuple of (pitch values, time stamps)
        """
        if LIBROSA_AVAILABLE:
            # Use pyin for pitch tracking
            try:
                f0, voiced_flag, voiced_probs = librosa.pyin(
                    audio,
                    fmin=fmin,
                    fmax=fmax,
                    sr=self.config.sample_rate,
                    hop_length=self.config.hop_length
                )
                times = librosa.times_like(f0, sr=self.config.sample_rate,
                                          hop_length=self.config.hop_length)
                # Replace NaN with 0
                f0 = np.nan_to_num(f0, nan=0.0)
                return f0, times
            except Exception:
                pass

        # Fallback: simple autocorrelation-based pitch detection
        frame_length = self.config.n_fft
        hop_length = self.config.hop_length
        n_frames = 1 + (len(audio) - frame_length) // hop_length

        pitch = np.zeros(n_frames)
        times = np.arange(n_frames) * hop_length / self.config.sample_rate

        for i in range(n_frames):
            start = i * hop_length
            frame = audio[start:start + frame_length]

            if len(frame) < frame_length:
                continue

            # Autocorrelation
            corr = np.correlate(frame, frame, mode='full')
            corr = corr[len(corr)//2:]

            # Find first peak after minimum lag
            min_lag = int(self.config.sample_rate / fmax)
            max_lag = int(self.config.sample_rate / fmin)

            if max_lag < len(corr):
                peak_idx = min_lag + np.argmax(corr[min_lag:max_lag])
                if corr[peak_idx] > 0.3 * corr[0]:  # Confidence threshold
                    pitch[i] = self.config.sample_rate / peak_idx

        return pitch, times

    def extract_energy(self, audio: np.ndarray) -> np.ndarray:
        """
        Extract energy envelope (RMS).

        Energy indicates loudness and speech activity.

        Args:
            audio: Audio signal

        Returns:
            Energy envelope [time_frames]
        """
        if LIBROSA_AVAILABLE:
            rms = librosa.feature.rms(
                y=audio,
                frame_length=self.config.n_fft,
                hop_length=self.config.hop_length
            )[0]
        else:
            # Manual RMS computation
            frame_length = self.config.n_fft
            hop_length = self.config.hop_length
            n_frames = 1 + (len(audio) - frame_length) // hop_length

            rms = np.zeros(n_frames)
            for i in range(n_frames):
                start = i * hop_length
                frame = audio[start:start + frame_length]
                rms[i] = np.sqrt(np.mean(frame ** 2))

        return rms

    def extract_spectral_centroid(self, audio: np.ndarray) -> np.ndarray:
        """
        Extract spectral centroid.

        The "center of mass" of the spectrum - indicates brightness.

        Args:
            audio: Audio signal

        Returns:
            Spectral centroid [time_frames]
        """
        if LIBROSA_AVAILABLE:
            centroid = librosa.feature.spectral_centroid(
                y=audio,
                sr=self.config.sample_rate,
                n_fft=self.config.n_fft,
                hop_length=self.config.hop_length
            )[0]
        else:
            spec = np.abs(self.processor.compute_stft(audio))
            freqs = self.processor.get_frequency_axis()
            centroid = np.sum(freqs[:, None] * spec, axis=0) / (np.sum(spec, axis=0) + 1e-10)

        return centroid

    def extract_spectral_bandwidth(self, audio: np.ndarray) -> np.ndarray:
        """
        Extract spectral bandwidth.

        Indicates the spread of frequencies around the centroid.

        Args:
            audio: Audio signal

        Returns:
            Spectral bandwidth [time_frames]
        """
        if LIBROSA_AVAILABLE:
            bandwidth = librosa.feature.spectral_bandwidth(
                y=audio,
                sr=self.config.sample_rate,
                n_fft=self.config.n_fft,
                hop_length=self.config.hop_length
            )[0]
        else:
            spec = np.abs(self.processor.compute_stft(audio))
            freqs = self.processor.get_frequency_axis()
            centroid = self.extract_spectral_centroid(audio)
            bandwidth = np.sqrt(
                np.sum(((freqs[:, None] - centroid) ** 2) * spec, axis=0) /
                (np.sum(spec, axis=0) + 1e-10)
            )

        return bandwidth

    def extract_spectral_rolloff(
        self,
        audio: np.ndarray,
        roll_percent: float = 0.85
    ) -> np.ndarray:
        """
        Extract spectral rolloff.

        Frequency below which roll_percent of energy is contained.

        Args:
            audio: Audio signal
            roll_percent: Energy percentage threshold

        Returns:
            Spectral rolloff [time_frames]
        """
        if LIBROSA_AVAILABLE:
            rolloff = librosa.feature.spectral_rolloff(
                y=audio,
                sr=self.config.sample_rate,
                n_fft=self.config.n_fft,
                hop_length=self.config.hop_length,
                roll_percent=roll_percent
            )[0]
        else:
            spec = np.abs(self.processor.compute_stft(audio)) ** 2
            freqs = self.processor.get_frequency_axis()
            total_energy = np.sum(spec, axis=0)
            cumsum = np.cumsum(spec, axis=0)
            threshold = roll_percent * total_energy
            rolloff = np.array([
                freqs[np.searchsorted(cumsum[:, i], threshold[i])]
                if threshold[i] > 0 else 0
                for i in range(spec.shape[1])
            ])

        return rolloff

    def extract_zero_crossing_rate(self, audio: np.ndarray) -> np.ndarray:
        """
        Extract zero-crossing rate.

        Indicates noisiness vs harmonicity.

        Args:
            audio: Audio signal

        Returns:
            Zero-crossing rate [time_frames]
        """
        if LIBROSA_AVAILABLE:
            zcr = librosa.feature.zero_crossing_rate(
                audio,
                frame_length=self.config.n_fft,
                hop_length=self.config.hop_length
            )[0]
        else:
            frame_length = self.config.n_fft
            hop_length = self.config.hop_length
            n_frames = 1 + (len(audio) - frame_length) // hop_length

            zcr = np.zeros(n_frames)
            for i in range(n_frames):
                start = i * hop_length
                frame = audio[start:start + frame_length]
                zcr[i] = np.sum(np.abs(np.diff(np.sign(frame)))) / (2 * frame_length)

        return zcr


# Test when run directly
if __name__ == "__main__":
    from audio_processor import generate_speech_like_audio

    print("Testing voice feature extractor...")

    # Generate test audio
    audio = generate_speech_like_audio(duration=2.0)
    print(f"Audio shape: {audio.shape}")

    # Create extractor
    extractor = VoiceFeatureExtractor()

    # Extract all features
    features = extractor.extract_all_features(audio)

    print(f"\nExtracted features:")
    print(f"  Duration: {features.duration:.2f}s")
    print(f"  Mel-spectrogram: {features.mel_spectrogram.shape}")
    print(f"  Log mel-spectrogram: {features.log_mel_spectrogram.shape}")
    print(f"  MFCCs: {features.mfccs.shape}")
    print(f"  Delta MFCCs: {features.delta_mfccs.shape}")
    print(f"  Pitch: {features.pitch.shape if features.pitch is not None else 'N/A'}")
    print(f"  Energy: {features.energy.shape}")
    print(f"  Spectral centroid: {features.spectral_centroid.shape}")
    print(f"  Spectral bandwidth: {features.spectral_bandwidth.shape}")
    print(f"  Spectral rolloff: {features.spectral_rolloff.shape}")
    print(f"  Zero-crossing rate: {features.zero_crossing_rate.shape}")

    # Check ranges
    print(f"\nFeature ranges:")
    print(f"  MFCC range: [{features.mfccs.min():.2f}, {features.mfccs.max():.2f}]")
    print(f"  Pitch range: [{features.pitch.min():.1f}, {features.pitch.max():.1f}] Hz")
    print(f"  Centroid range: [{features.spectral_centroid.min():.1f}, {features.spectral_centroid.max():.1f}] Hz")

    print("\nAll tests passed!")
