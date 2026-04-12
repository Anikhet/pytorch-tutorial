"""
Real-time voice feature visualizer using Pygame.

Displays:
- Waveform
- Mel-spectrogram (scrolling)
- MFCCs
- Pitch contour
- Energy envelope
- Spectral features
"""

import numpy as np
from typing import Optional, Tuple, List
from dataclasses import dataclass
import colorsys

try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False
    print("Warning: pygame not installed. Visualization disabled.")

from audio_processor import AudioConfig
from feature_extractor import VoiceFeatures


@dataclass
class VisualizerConfig:
    """Configuration for the visualizer."""
    width: int = 1200
    height: int = 800
    fps: int = 30
    background_color: Tuple[int, int, int] = (15, 15, 25)
    accent_color: Tuple[int, int, int] = (100, 200, 255)
    text_color: Tuple[int, int, int] = (200, 200, 200)
    grid_color: Tuple[int, int, int] = (40, 40, 50)

    # Panel layout
    waveform_height: int = 100
    spectrogram_height: int = 200
    mfcc_height: int = 150
    features_height: int = 150


class ColorMapper:
    """Map values to colors for spectrograms."""

    @staticmethod
    def viridis(value: float) -> Tuple[int, int, int]:
        """Viridis-like colormap (0-1 input)."""
        value = np.clip(value, 0, 1)

        # Simplified viridis approximation
        if value < 0.25:
            r = 68 + int(value * 4 * (33 - 68))
            g = 1 + int(value * 4 * (145 - 1))
            b = 84 + int(value * 4 * (140 - 84))
        elif value < 0.5:
            t = (value - 0.25) * 4
            r = 33 + int(t * (94 - 33))
            g = 145 + int(t * (201 - 145))
            b = 140 + int(t * (98 - 140))
        elif value < 0.75:
            t = (value - 0.5) * 4
            r = 94 + int(t * (190 - 94))
            g = 201 + int(t * (219 - 201))
            b = 98 + int(t * (57 - 98))
        else:
            t = (value - 0.75) * 4
            r = 190 + int(t * (253 - 190))
            g = 219 + int(t * (231 - 219))
            b = 57 + int(t * (37 - 57))

        return (np.clip(r, 0, 255), np.clip(g, 0, 255), np.clip(b, 0, 255))

    @staticmethod
    def magma(value: float) -> Tuple[int, int, int]:
        """Magma-like colormap for spectrograms."""
        value = np.clip(value, 0, 1)

        if value < 0.33:
            t = value * 3
            r = int(t * 183)
            g = int(t * 55)
            b = int(100 + t * 121)
        elif value < 0.66:
            t = (value - 0.33) * 3
            r = 183 + int(t * (252 - 183))
            g = 55 + int(t * (136 - 55))
            b = 221 + int(t * (97 - 221))
        else:
            t = (value - 0.66) * 3
            r = 252
            g = 136 + int(t * (253 - 136))
            b = 97 + int(t * (191 - 97))

        return (np.clip(r, 0, 255), np.clip(g, 0, 255), np.clip(b, 0, 255))


class VoiceVisualizer:
    """
    Real-time voice feature visualizer.

    Displays multiple synchronized views of audio features:
    - Waveform display
    - Scrolling mel-spectrogram
    - MFCC coefficients
    - Pitch and energy contours
    """

    def __init__(self, config: Optional[VisualizerConfig] = None):
        if not PYGAME_AVAILABLE:
            raise ImportError("pygame is required for visualization")

        self.config = config or VisualizerConfig()
        self.running = False
        self.screen = None
        self.clock = None
        self.font = None
        self.small_font = None

        # Feature storage for animation
        self.current_features: Optional[VoiceFeatures] = None
        self.playback_position: int = 0
        self.is_playing: bool = False

        # Scrolling spectrogram buffer
        self.spec_buffer_width = 200
        self.spec_buffer: Optional[np.ndarray] = None

    def initialize(self):
        """Initialize pygame and create window."""
        pygame.init()
        pygame.display.set_caption("Voice Feature Visualizer")

        self.screen = pygame.display.set_mode(
            (self.config.width, self.config.height)
        )
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 24)
        self.small_font = pygame.font.Font(None, 18)
        self.running = True

    def set_features(self, features: VoiceFeatures):
        """Set features to visualize."""
        self.current_features = features
        self.playback_position = 0

        # Initialize spectrogram buffer
        self.spec_buffer = np.zeros(
            (features.log_mel_spectrogram.shape[0], self.spec_buffer_width)
        )

    def draw_panel_background(
        self,
        x: int, y: int,
        width: int, height: int,
        title: str
    ):
        """Draw a panel with background and title."""
        # Background
        pygame.draw.rect(
            self.screen,
            (25, 25, 35),
            (x, y, width, height),
            border_radius=5
        )

        # Border
        pygame.draw.rect(
            self.screen,
            (50, 50, 60),
            (x, y, width, height),
            width=1,
            border_radius=5
        )

        # Title
        title_surface = self.small_font.render(
            title, True, self.config.text_color
        )
        self.screen.blit(title_surface, (x + 10, y + 5))

    def draw_waveform(self, x: int, y: int, width: int, height: int):
        """Draw the audio waveform."""
        self.draw_panel_background(x, y, width, height, "Waveform")

        if self.current_features is None:
            return

        audio = self.current_features.audio

        # Downsample for display
        samples_per_pixel = max(1, len(audio) // width)
        display_samples = len(audio) // samples_per_pixel

        # Calculate waveform points
        center_y = y + height // 2
        points = []

        for i in range(min(display_samples, width - 20)):
            start_idx = i * samples_per_pixel
            end_idx = min(start_idx + samples_per_pixel, len(audio))

            if start_idx < len(audio):
                # Get min/max for this segment
                segment = audio[start_idx:end_idx]
                max_val = np.max(segment) if len(segment) > 0 else 0
                min_val = np.min(segment) if len(segment) > 0 else 0

                # Scale to display
                scale = (height - 40) / 2
                top = center_y - int(max_val * scale)
                bottom = center_y - int(min_val * scale)

                # Draw vertical line for this segment
                pygame.draw.line(
                    self.screen,
                    self.config.accent_color,
                    (x + 10 + i, top),
                    (x + 10 + i, bottom)
                )

        # Draw playback position indicator
        if self.is_playing and len(audio) > 0:
            pos_x = x + 10 + int((self.playback_position / len(audio)) * (width - 20))
            pygame.draw.line(
                self.screen,
                (255, 100, 100),
                (pos_x, y + 20),
                (pos_x, y + height - 10),
                2
            )

    def draw_spectrogram(self, x: int, y: int, width: int, height: int):
        """Draw the mel-spectrogram."""
        self.draw_panel_background(x, y, width, height, "Mel-Spectrogram (Log Scale)")

        if self.current_features is None:
            return

        spec = self.current_features.log_mel_spectrogram

        # Normalize spectrogram for display
        spec_min = spec.min()
        spec_max = spec.max()
        if spec_max - spec_min > 0:
            spec_norm = (spec - spec_min) / (spec_max - spec_min)
        else:
            spec_norm = np.zeros_like(spec)

        # Calculate display parameters
        display_width = width - 20
        display_height = height - 30

        # Resample spectrogram to fit display
        n_mels, n_frames = spec_norm.shape

        # Create surface for spectrogram
        spec_surface = pygame.Surface((display_width, display_height))

        for px in range(display_width):
            frame_idx = int(px * n_frames / display_width)
            frame_idx = min(frame_idx, n_frames - 1)

            for py in range(display_height):
                # Flip y-axis (low frequencies at bottom)
                mel_idx = n_mels - 1 - int(py * n_mels / display_height)
                mel_idx = max(0, min(mel_idx, n_mels - 1))

                value = spec_norm[mel_idx, frame_idx]
                color = ColorMapper.magma(value)
                spec_surface.set_at((px, py), color)

        self.screen.blit(spec_surface, (x + 10, y + 25))

        # Draw frequency labels
        freqs = [0, 1000, 2000, 4000, 8000]
        for freq in freqs:
            if freq <= 8000:
                # Approximate mel position
                mel_pos = int((1 - freq / 8000) * display_height)
                label = self.small_font.render(
                    f"{freq}Hz", True, (150, 150, 150)
                )
                self.screen.blit(
                    label,
                    (x + display_width - 35, y + 25 + mel_pos - 6)
                )

    def draw_mfccs(self, x: int, y: int, width: int, height: int):
        """Draw MFCC coefficients."""
        self.draw_panel_background(x, y, width, height, "MFCCs (Mel-Frequency Cepstral Coefficients)")

        if self.current_features is None:
            return

        mfccs = self.current_features.mfccs
        n_mfcc, n_frames = mfccs.shape

        # Normalize MFCCs
        mfcc_min = mfccs.min()
        mfcc_max = mfccs.max()
        if mfcc_max - mfcc_min > 0:
            mfcc_norm = (mfccs - mfcc_min) / (mfcc_max - mfcc_min)
        else:
            mfcc_norm = np.zeros_like(mfccs)

        display_width = width - 20
        display_height = height - 30

        # Create surface
        mfcc_surface = pygame.Surface((display_width, display_height))

        for px in range(display_width):
            frame_idx = int(px * n_frames / display_width)
            frame_idx = min(frame_idx, n_frames - 1)

            for py in range(display_height):
                mfcc_idx = int(py * n_mfcc / display_height)
                mfcc_idx = min(mfcc_idx, n_mfcc - 1)

                value = mfcc_norm[mfcc_idx, frame_idx]
                color = ColorMapper.viridis(value)
                mfcc_surface.set_at((px, py), color)

        self.screen.blit(mfcc_surface, (x + 10, y + 25))

        # MFCC labels
        for i in range(0, n_mfcc, 3):
            label_y = y + 25 + int(i * display_height / n_mfcc)
            label = self.small_font.render(f"C{i}", True, (150, 150, 150))
            self.screen.blit(label, (x + display_width - 25, label_y))

    def draw_pitch_energy(self, x: int, y: int, width: int, height: int):
        """Draw pitch and energy contours."""
        self.draw_panel_background(x, y, width, height, "Pitch (F0) & Energy")

        if self.current_features is None:
            return

        display_width = width - 40
        display_height = height - 40

        # Draw pitch contour
        if self.current_features.pitch is not None:
            pitch = self.current_features.pitch
            pitch_max = np.max(pitch[pitch > 0]) if np.any(pitch > 0) else 500

            points = []
            for i, p in enumerate(pitch):
                if p > 0:  # Only voiced regions
                    px = x + 20 + int(i * display_width / len(pitch))
                    py = y + height - 20 - int((p / pitch_max) * display_height * 0.8)
                    points.append((px, py))

            if len(points) > 1:
                pygame.draw.lines(
                    self.screen,
                    (255, 150, 100),
                    False,
                    points,
                    2
                )

            # Pitch label
            label = self.small_font.render("Pitch (Hz)", True, (255, 150, 100))
            self.screen.blit(label, (x + 20, y + 20))

        # Draw energy contour
        if self.current_features.energy is not None:
            energy = self.current_features.energy
            energy_norm = energy / (np.max(energy) + 1e-10)

            points = []
            for i, e in enumerate(energy_norm):
                px = x + 20 + int(i * display_width / len(energy_norm))
                py = y + height - 20 - int(e * display_height * 0.4)
                points.append((px, py))

            if len(points) > 1:
                pygame.draw.lines(
                    self.screen,
                    (100, 200, 150),
                    False,
                    points,
                    2
                )

            # Energy label
            label = self.small_font.render("Energy (RMS)", True, (100, 200, 150))
            self.screen.blit(label, (x + 120, y + 20))

    def draw_spectral_features(self, x: int, y: int, width: int, height: int):
        """Draw spectral feature plots."""
        self.draw_panel_background(x, y, width, height, "Spectral Features")

        if self.current_features is None:
            return

        display_width = width - 40
        display_height = height - 40

        features = [
            ("Centroid", self.current_features.spectral_centroid, (100, 150, 255)),
            ("Bandwidth", self.current_features.spectral_bandwidth, (255, 200, 100)),
            ("Rolloff", self.current_features.spectral_rolloff, (200, 100, 255)),
        ]

        legend_x = x + 20
        for i, (name, data, color) in enumerate(features):
            if data is not None:
                # Normalize
                data_norm = data / (np.max(data) + 1e-10)

                points = []
                for j, val in enumerate(data_norm):
                    px = x + 20 + int(j * display_width / len(data_norm))
                    py = y + height - 20 - int(val * display_height * 0.8)
                    points.append((px, py))

                if len(points) > 1:
                    pygame.draw.lines(self.screen, color, False, points, 1)

                # Legend
                label = self.small_font.render(name, True, color)
                self.screen.blit(label, (legend_x, y + 20))
                legend_x += 80

    def draw_stats_panel(self, x: int, y: int, width: int, height: int):
        """Draw statistics panel."""
        self.draw_panel_background(x, y, width, height, "Audio Statistics")

        if self.current_features is None:
            return

        stats = [
            f"Duration: {self.current_features.duration:.2f}s",
            f"Sample Rate: {self.current_features.sample_rate} Hz",
            f"Samples: {len(self.current_features.audio):,}",
            f"Mel Bins: {self.current_features.mel_spectrogram.shape[0]}",
            f"MFCC Coeffs: {self.current_features.mfccs.shape[0]}",
        ]

        # Add pitch stats
        if self.current_features.pitch is not None:
            voiced_pitch = self.current_features.pitch[self.current_features.pitch > 0]
            if len(voiced_pitch) > 0:
                stats.append(f"Avg Pitch: {np.mean(voiced_pitch):.1f} Hz")

        for i, stat in enumerate(stats):
            label = self.small_font.render(stat, True, self.config.text_color)
            self.screen.blit(label, (x + 15, y + 25 + i * 20))

    def draw_instructions(self):
        """Draw keyboard instructions."""
        instructions = [
            "Controls:",
            "  SPACE - Play/Pause",
            "  R - Reset position",
            "  Q/ESC - Quit"
        ]

        for i, text in enumerate(instructions):
            label = self.small_font.render(text, True, (100, 100, 120))
            self.screen.blit(label, (10, self.config.height - 80 + i * 18))

    def render(self):
        """Render all visualizations."""
        if not self.running or self.screen is None:
            return

        # Clear screen
        self.screen.fill(self.config.background_color)

        # Draw title
        title = self.font.render(
            "Voice Feature Visualizer",
            True,
            self.config.accent_color
        )
        self.screen.blit(title, (self.config.width // 2 - 100, 10))

        # Calculate panel positions
        margin = 10
        panel_width = self.config.width - 2 * margin
        half_width = (panel_width - margin) // 2

        y_pos = 40

        # Row 1: Waveform (full width)
        self.draw_waveform(margin, y_pos, panel_width, self.config.waveform_height)
        y_pos += self.config.waveform_height + margin

        # Row 2: Spectrogram (full width)
        self.draw_spectrogram(margin, y_pos, panel_width, self.config.spectrogram_height)
        y_pos += self.config.spectrogram_height + margin

        # Row 3: MFCCs (full width)
        self.draw_mfccs(margin, y_pos, panel_width, self.config.mfcc_height)
        y_pos += self.config.mfcc_height + margin

        # Row 4: Pitch/Energy (left) and Spectral Features (right)
        self.draw_pitch_energy(margin, y_pos, half_width, self.config.features_height)
        self.draw_spectral_features(
            margin + half_width + margin, y_pos,
            half_width, self.config.features_height
        )
        y_pos += self.config.features_height + margin

        # Stats panel
        stats_height = 150
        self.draw_stats_panel(margin, y_pos, 250, stats_height)

        # Instructions
        self.draw_instructions()

        # Update display
        pygame.display.flip()

    def handle_events(self) -> bool:
        """Handle pygame events. Returns False if should quit."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE or event.key == pygame.K_q:
                    return False
                elif event.key == pygame.K_SPACE:
                    self.is_playing = not self.is_playing
                elif event.key == pygame.K_r:
                    self.playback_position = 0

        return True

    def update(self):
        """Update visualization state."""
        if self.is_playing and self.current_features is not None:
            # Update playback position
            samples_per_frame = int(
                self.current_features.sample_rate / self.config.fps
            )
            self.playback_position += samples_per_frame

            if self.playback_position >= len(self.current_features.audio):
                self.playback_position = 0
                self.is_playing = False

    def run(self, features: Optional[VoiceFeatures] = None):
        """
        Main visualization loop.

        Args:
            features: VoiceFeatures to visualize (optional, can set later)
        """
        self.initialize()

        if features is not None:
            self.set_features(features)

        while self.running:
            if not self.handle_events():
                break

            self.update()
            self.render()
            self.clock.tick(self.config.fps)

        pygame.quit()

    def cleanup(self):
        """Clean up pygame resources."""
        if pygame.get_init():
            pygame.quit()


class StaticVisualizer:
    """
    Static visualization using Matplotlib.

    Creates publication-quality figures of voice features.
    """

    def __init__(self):
        try:
            import matplotlib.pyplot as plt
            self.plt = plt
            self.available = True
        except ImportError:
            self.available = False

    def plot_all_features(
        self,
        features: VoiceFeatures,
        figsize: Tuple[int, int] = (14, 12),
        save_path: Optional[str] = None
    ):
        """Create comprehensive feature visualization."""
        if not self.available:
            print("Matplotlib not available for static visualization")
            return

        fig, axes = self.plt.subplots(3, 2, figsize=figsize)
        fig.suptitle("Voice Feature Analysis", fontsize=14, fontweight='bold')

        time_axis = features.time_axis if features.time_axis is not None else \
                    np.linspace(0, features.duration, features.mel_spectrogram.shape[1])

        # 1. Waveform
        ax = axes[0, 0]
        audio_time = np.linspace(0, features.duration, len(features.audio))
        ax.plot(audio_time, features.audio, color='steelblue', linewidth=0.5)
        ax.set_title("Waveform")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude")
        ax.set_xlim(0, features.duration)
        ax.grid(True, alpha=0.3)

        # 2. Mel-Spectrogram
        ax = axes[0, 1]
        img = ax.imshow(
            features.log_mel_spectrogram,
            aspect='auto',
            origin='lower',
            extent=[0, features.duration, 0, 8000],
            cmap='magma'
        )
        ax.set_title("Log Mel-Spectrogram")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Frequency (Hz)")
        fig.colorbar(img, ax=ax, label='dB')

        # 3. MFCCs
        ax = axes[1, 0]
        img = ax.imshow(
            features.mfccs,
            aspect='auto',
            origin='lower',
            extent=[0, features.duration, 0, features.mfccs.shape[0]],
            cmap='viridis'
        )
        ax.set_title("MFCCs")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("MFCC Coefficient")
        fig.colorbar(img, ax=ax, label='Value')

        # 4. Pitch and Energy
        ax = axes[1, 1]
        if features.pitch is not None:
            pitch_times = features.pitch_times if features.pitch_times is not None else time_axis
            voiced_mask = features.pitch > 0
            ax.scatter(
                pitch_times[voiced_mask],
                features.pitch[voiced_mask],
                c='coral', s=1, alpha=0.7, label='Pitch (F0)'
            )
        ax.set_title("Pitch Contour (F0)")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Frequency (Hz)")
        ax.set_xlim(0, features.duration)
        ax.grid(True, alpha=0.3)
        ax.legend()

        # 5. Energy
        ax = axes[2, 0]
        if features.energy is not None:
            energy_time = np.linspace(0, features.duration, len(features.energy))
            ax.fill_between(
                energy_time,
                features.energy,
                alpha=0.7,
                color='seagreen',
                label='RMS Energy'
            )
        ax.set_title("Energy Envelope")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("RMS Energy")
        ax.set_xlim(0, features.duration)
        ax.grid(True, alpha=0.3)

        # 6. Spectral Features
        ax = axes[2, 1]
        if features.spectral_centroid is not None:
            centroid_time = np.linspace(0, features.duration, len(features.spectral_centroid))
            ax.plot(
                centroid_time,
                features.spectral_centroid / 1000,
                label='Centroid (kHz)',
                color='royalblue'
            )
        if features.spectral_rolloff is not None:
            rolloff_time = np.linspace(0, features.duration, len(features.spectral_rolloff))
            ax.plot(
                rolloff_time,
                features.spectral_rolloff / 1000,
                label='Rolloff (kHz)',
                color='purple',
                alpha=0.7
            )
        ax.set_title("Spectral Features")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Frequency (kHz)")
        ax.set_xlim(0, features.duration)
        ax.legend()
        ax.grid(True, alpha=0.3)

        self.plt.tight_layout()

        if save_path:
            self.plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved figure to {save_path}")
        else:
            self.plt.show()

        return fig


# Test when run directly
if __name__ == "__main__":
    from audio_processor import generate_speech_like_audio
    from feature_extractor import VoiceFeatureExtractor

    print("Testing voice visualizer...")

    # Generate test audio
    audio = generate_speech_like_audio(duration=3.0)
    print(f"Generated audio: {len(audio)} samples")

    # Extract features
    extractor = VoiceFeatureExtractor()
    features = extractor.extract_all_features(audio)
    print(f"Extracted features")

    # Test static visualizer first
    print("\nTesting static visualizer (Matplotlib)...")
    static_viz = StaticVisualizer()
    if static_viz.available:
        static_viz.plot_all_features(features, save_path="test_features.png")

    # Test real-time visualizer
    print("\nTesting real-time visualizer (Pygame)...")
    print("Controls: SPACE=Play/Pause, R=Reset, Q/ESC=Quit")

    if PYGAME_AVAILABLE:
        viz = VoiceVisualizer()
        viz.run(features)
    else:
        print("Pygame not available - skipping real-time visualization")

    print("\nAll tests complete!")
