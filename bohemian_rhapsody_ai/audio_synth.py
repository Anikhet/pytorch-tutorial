"""
Guitar-like audio synthesis using pygame.mixer.
Generates guitar tones with harmonics and envelope shaping.
"""

import numpy as np
import pygame
from guitar_solo import get_note_frequency


class GuitarSynth:
    """
    Synthesizes guitar-like sounds with harmonics and ADSR envelope.
    """

    def __init__(self, sample_rate=22050):
        """
        Initialize the synthesizer.

        Args:
            sample_rate: Audio sample rate in Hz
        """
        self.sample_rate = sample_rate
        self.initialized = False
        self.enabled = True
        self.volume = 0.3  # Default volume (0.0 to 1.0)

        # Initialize pygame mixer (stereo for better compatibility)
        try:
            pygame.mixer.init(frequency=sample_rate, size=-16, channels=2, buffer=512)
            self.initialized = True
            print(f"Audio synthesizer initialized at {sample_rate}Hz")
        except Exception as e:
            print(f"Warning: Could not initialize audio: {e}")
            self.initialized = False

        # Pre-generate some sounds to avoid lag
        self.sound_cache = {}

    def generate_guitar_tone(self, frequency, duration=0.3):
        """
        Generate a guitar-like tone with harmonics and envelope.

        Args:
            frequency: Fundamental frequency in Hz
            duration: Note duration in seconds

        Returns:
            pygame.Sound object
        """
        if not self.initialized:
            return None

        # Check cache first
        cache_key = (int(frequency * 10), int(duration * 100))
        if cache_key in self.sound_cache:
            return self.sound_cache[cache_key]

        num_samples = int(self.sample_rate * duration)
        t = np.linspace(0, duration, num_samples, False)

        # ========================================
        # REALISTIC GUITAR STRING PHYSICS
        # ========================================

        wave = np.zeros(num_samples)

        # Inharmonicity factor (real strings aren't perfectly harmonic)
        # Higher for wound strings, lower for plain strings
        inharmonicity = 0.0005 * (frequency / 200.0)  # Increases with pitch

        # Generate harmonics with realistic amplitudes and decay
        # Electric guitar has strong harmonics up to ~10th
        harmonic_amplitudes = [
            1.0,    # Fundamental (loudest)
            0.7,    # 2nd (strong octave)
            0.5,    # 3rd (fifth above octave)
            0.4,    # 4th (two octaves)
            0.3,    # 5th
            0.25,   # 6th
            0.2,    # 7th
            0.15,   # 8th
            0.1,    # 9th
            0.08,   # 10th
        ]

        # Each harmonic decays at different rate (higher harmonics decay faster)
        for n, amp in enumerate(harmonic_amplitudes, start=1):
            # Add slight inharmonicity (makes it sound more like real string)
            harmonic_freq = frequency * n * (1 + inharmonicity * n * n)

            # Decay envelope for this harmonic (higher harmonics die faster)
            decay_rate = 0.5 + (n * 0.3)  # Higher harmonics decay faster
            harmonic_decay = np.exp(-decay_rate * t)

            # Generate harmonic with decay
            wave += amp * harmonic_decay * np.sin(2 * np.pi * harmonic_freq * t)

        # ========================================
        # GUITAR BODY RESONANCE
        # ========================================
        # Electric guitar body has resonant frequencies around 200-400 Hz
        body_resonance = 0.15 * np.sin(2 * np.pi * 300 * t) * np.exp(-3 * t)
        wave += body_resonance

        # ========================================
        # PICK ATTACK NOISE
        # ========================================
        # Initial sharp transient from pick hitting string
        attack_noise_samples = int(0.005 * self.sample_rate)  # 5ms
        if attack_noise_samples > 0:
            # High-frequency noise burst
            attack_noise = np.random.normal(0, 0.3, attack_noise_samples)
            # High-pass filter the noise (keeps only high frequencies)
            attack_noise = np.diff(np.concatenate([[0], attack_noise]))
            wave[:attack_noise_samples] += attack_noise * 0.4

        # ========================================
        # ENVELOPE - Guitar-specific ADSR
        # ========================================
        envelope = np.ones(num_samples)

        # Very fast attack (2ms - guitar pick is sharp)
        attack_samples = int(0.002 * self.sample_rate)
        if attack_samples > 0:
            # Exponential attack for more realistic pluck
            envelope[:attack_samples] = 1 - np.exp(-5 * np.linspace(0, 1, attack_samples))

        # Quick initial decay (30ms) - string energy settles
        decay_samples = int(0.03 * self.sample_rate)
        sustain_level = 0.6  # Guitar sustain is lower than synth
        if decay_samples > 0 and attack_samples + decay_samples < num_samples:
            envelope[attack_samples:attack_samples + decay_samples] = \
                np.linspace(1, sustain_level, decay_samples)

        # Sustain with slow decay (strings naturally lose energy)
        sustain_start = attack_samples + decay_samples
        sustain_samples = num_samples - sustain_start - int(0.05 * self.sample_rate)
        if sustain_samples > 0:
            # Exponential decay during sustain
            sustain_decay = np.exp(-1.5 * np.linspace(0, 1, sustain_samples))
            envelope[sustain_start:sustain_start + sustain_samples] = \
                sustain_level * sustain_decay

        # Release (50ms) - string damping
        release_samples = min(int(0.05 * self.sample_rate),
                            num_samples - sustain_start - sustain_samples)
        if release_samples > 0:
            release_start = sustain_start + sustain_samples
            final_level = envelope[release_start - 1] if release_start > 0 else sustain_level
            envelope[release_start:release_start + release_samples] = \
                final_level * np.exp(-8 * np.linspace(0, 1, release_samples))

        # Apply envelope
        wave = wave * envelope

        # ========================================
        # SATURATION / OVERDRIVE
        # ========================================
        # Subtle overdrive for electric guitar character
        # Soft clipping for warm distortion
        wave = np.tanh(wave * 1.2) * 0.9

        # ========================================
        # FINAL PROCESSING
        # ========================================
        # Normalize
        max_val = np.max(np.abs(wave))
        if max_val > 0:
            wave = wave / max_val

        # Apply volume
        wave = wave * self.volume

        # Add tiny bit of noise for analog warmth
        wave += np.random.normal(0, 0.002, num_samples)

        # Convert to 16-bit integers
        wave = (wave * 32767).astype(np.int16)

        # Convert mono to stereo (duplicate to both channels)
        stereo_wave = np.column_stack((wave, wave))

        # Create pygame Sound
        try:
            sound = pygame.sndarray.make_sound(stereo_wave)
            # Cache the sound
            if len(self.sound_cache) < 200:  # Limit cache size
                self.sound_cache[cache_key] = sound
            return sound
        except Exception as e:
            print(f"Warning: Could not create sound: {e}")
            return None

    def play_note(self, string, fret, duration=0.3):
        """
        Play a guitar note.

        Args:
            string: String number (1-6)
            fret: Fret number (0-22)
            duration: Note duration in seconds
        """
        if not self.initialized or not self.enabled:
            return

        # Get frequency for this note
        frequency = get_note_frequency(string, fret)

        # Generate and play sound
        sound = self.generate_guitar_tone(frequency, duration)
        if sound:
            try:
                sound.play()
            except Exception as e:
                print(f"Warning: Could not play sound: {e}")

    def set_volume(self, volume):
        """
        Set global volume.

        Args:
            volume: Volume level (0.0 to 1.0)
        """
        self.volume = np.clip(volume, 0.0, 1.0)
        # Clear cache when volume changes
        self.sound_cache = {}

    def toggle_audio(self):
        """Toggle audio on/off."""
        self.enabled = not self.enabled
        return self.enabled

    def stop_all(self):
        """Stop all currently playing sounds."""
        if self.initialized:
            pygame.mixer.stop()

    def pre_cache_common_notes(self):
        """
        Pre-generate sounds for common notes to reduce latency.
        Call this during initialization for smoother playback.
        """
        if not self.initialized:
            return

        print("Pre-caching guitar sounds...")
        # Cache notes used in the solo
        from guitar_solo import GUITAR_SOLO

        for string, fret, duration, _ in GUITAR_SOLO:
            freq = get_note_frequency(string, fret)
            # Generate for common durations
            for dur in [0.2, 0.3, 0.5, 1.0]:
                self.generate_guitar_tone(freq, dur)

        print(f"Cached {len(self.sound_cache)} sounds")


# Global synthesizer instance
_synth = None


def get_synth():
    """Get or create the global synthesizer instance."""
    global _synth
    if _synth is None:
        _synth = GuitarSynth()
        _synth.pre_cache_common_notes()
    return _synth


def play_note(string, fret, duration=0.3):
    """
    Convenience function to play a note.

    Args:
        string: String number (1-6)
        fret: Fret number (0-22)
        duration: Note duration in seconds
    """
    synth = get_synth()
    synth.play_note(string, fret, duration)


def set_volume(volume):
    """Set audio volume (0.0 to 1.0)."""
    synth = get_synth()
    synth.set_volume(volume)


def toggle_audio():
    """Toggle audio on/off. Returns new state (True=on, False=off)."""
    synth = get_synth()
    return synth.toggle_audio()


def stop_all():
    """Stop all playing sounds."""
    synth = get_synth()
    synth.stop_all()


if __name__ == "__main__":
    # Test the synthesizer
    print("Testing Guitar Synthesizer...")

    pygame.init()
    synth = GuitarSynth()

    if not synth.initialized:
        print("Audio initialization failed!")
        exit(1)

    print("\nPlaying test notes...")

    # Play a scale
    test_notes = [
        (3, 5),   # C
        (3, 7),   # D
        (3, 9),   # E
        (2, 10),  # A
        (2, 12),  # B
    ]

    for string, fret in test_notes:
        freq = get_note_frequency(string, fret)
        print(f"Playing String {string}, Fret {fret} ({freq:.1f} Hz)")
        synth.play_note(string, fret, duration=0.5)
        pygame.time.wait(600)  # Wait for note to finish

    print("\nTest complete!")
    pygame.quit()
