"""
Guitar player agent that learns to play guitar using neural network control.
"""

import numpy as np
import pygame
from guitar_solo import (
    GUITAR_SOLO, get_note_at_time, get_next_note_info,
    get_note_frequency, BEAT_DURATION, SOLO_DURATION
)
try:
    import audio_synth
    AUDIO_AVAILABLE = True
except ImportError:
    AUDIO_AVAILABLE = False


class GuitarPlayer:
    """
    Represents a guitar player controlled by a neural network.
    The player must learn to play the correct strings and frets at the right times.
    """

    def __init__(self, network, audio_enabled=False):
        """
        Initialize guitar player with a neural network.

        Args:
            network: Neural network that controls the player
            audio_enabled: Whether to play audio when notes are played
        """
        self.network = network
        self.audio_enabled = audio_enabled and AUDIO_AVAILABLE
        self.reset()

    def reset(self):
        """Reset the player state to beginning of solo."""
        self.time_beats = 0.0  # Current time in beats
        self.current_string = 3  # Start on G string
        self.current_fret = 5    # Start at 5th fret
        self.is_playing = False  # Whether currently playing a note

        # Fitness tracking
        self.correct_notes = 0
        self.total_notes_attempted = 0
        self.timing_accuracy = 0.0
        self.notes_missed = 0
        self.note_index = 0  # Track which note we should be playing

        # History for visualization
        self.played_notes = []  # List of (string, fret, time) tuples
        self.alive = True

    def get_sensor_data(self):
        """
        Get current state information for the neural network.

        Returns 8 inputs:
        1. Current time progress (0-1)
        2. Next note string (normalized 0-1)
        3. Next note fret (normalized 0-1)
        4. Time until next note (normalized)
        5. Current string position (normalized)
        6. Current fret position (normalized)
        7. Is currently playing (0 or 1)
        8. Progress through solo (0-1)
        """
        # Get next note information
        next_note_info = get_next_note_info(self.time_beats)

        if next_note_info:
            next_string, next_fret, time_until = next_note_info
            next_string_norm = next_string / 6.0
            next_fret_norm = next_fret / 22.0
            time_until_norm = min(time_until, 2.0) / 2.0  # Normalize to 0-1 (cap at 2 beats)
        else:
            # No more notes
            next_string_norm = 0.5
            next_fret_norm = 0.5
            time_until_norm = 1.0

        sensors = np.array([
            self.time_beats / SOLO_DURATION,  # Progress through time
            next_string_norm,                  # Next note string
            next_fret_norm,                    # Next note fret
            time_until_norm,                   # Time until next note
            self.current_string / 6.0,         # Current string position
            self.current_fret / 22.0,          # Current fret position
            1.0 if self.is_playing else 0.0,   # Is playing
            self.note_index / len(GUITAR_SOLO) # Progress through notes
        ])

        return sensors

    def update(self, dt):
        """
        Update player state and get neural network decision.

        Args:
            dt: Time step in seconds
        """
        if not self.alive:
            return

        # Update time
        time_beats_prev = self.time_beats
        self.time_beats += dt / BEAT_DURATION

        # Check if solo is complete
        if self.time_beats >= SOLO_DURATION:
            self.alive = True  # Made it to the end!
            return

        # Get neural network decision
        sensors = self.get_sensor_data()
        output = self.network.predict(sensors)

        # Decode neural network output
        # Output 0: String selection (0-5 maps to strings 1-6)
        # Output 1: Fret selection (0-22)
        # Output 2: Play trigger (-1 to 1, > 0 means play)
        string_output = output[0]  # -1 to 1
        fret_output = output[1]    # -1 to 1
        play_output = output[2]    # -1 to 1

        # Map outputs to actual string and fret
        self.current_string = int((string_output + 1) * 3) + 1  # Maps to 1-6
        self.current_string = np.clip(self.current_string, 1, 6)

        self.current_fret = int((fret_output + 1) * 11)  # Maps to 0-22
        self.current_fret = np.clip(self.current_fret, 0, 22)

        # Check if trying to play a note
        if play_output > 0.3 and not self.is_playing:
            # Player is attempting to play a note
            self.play_note()
            self.is_playing = True
        elif play_output <= 0:
            self.is_playing = False

    def play_note(self):
        """
        Attempt to play current string/fret and evaluate correctness.
        """
        self.total_notes_attempted += 1

        # Play audio if enabled
        if self.audio_enabled:
            try:
                audio_synth.play_note(self.current_string, self.current_fret, duration=0.3)
            except Exception as e:
                # Silently ignore audio errors to not disrupt training
                pass

        # Record the played note
        self.played_notes.append((
            self.current_string,
            self.current_fret,
            self.time_beats
        ))

        # Check what note(s) should be playing now
        active_notes = get_note_at_time(self.time_beats)

        if not active_notes:
            # No note should be playing - penalty for wrong timing
            self.notes_missed += 1
            return

        # Check if our note matches any active note
        note_correct = False
        best_timing_error = float('inf')

        for target_string, target_fret, time_held in active_notes:
            # Check if string and fret match (allow 1 fret tolerance)
            string_match = (self.current_string == target_string)
            fret_close = abs(self.current_fret - target_fret) <= 1

            if string_match and fret_close:
                note_correct = True
                # Calculate timing accuracy (how close to the start of the note)
                timing_error = time_held  # How far into the note we are
                best_timing_error = min(best_timing_error, timing_error)

        if note_correct:
            self.correct_notes += 1
            # Reward better timing (playing at the start of the note)
            timing_bonus = max(0, 1.0 - best_timing_error)
            self.timing_accuracy += timing_bonus

            # Move to next note
            if self.note_index < len(GUITAR_SOLO) - 1:
                self.note_index += 1
        else:
            self.notes_missed += 1

    def get_fitness(self):
        """
        Calculate fitness score for this player.

        Fitness components:
        - Time survived (baseline reward)
        - Notes attempted (small reward for trying)
        - Correct notes played (main component)
        - Timing accuracy bonus
        - Note progression bonus
        - Small penalty for missed/wrong notes
        - Completion bonus
        """
        # Baseline: reward for time survived (0-20 points)
        time_ratio = min(self.time_beats / SOLO_DURATION, 1.0)
        fitness = time_ratio * 20.0

        # Small reward for attempting notes (1 point per attempt)
        fitness += self.total_notes_attempted * 1.0

        # Main reward: correct notes (worth 15 points each)
        fitness += self.correct_notes * 15.0

        # Timing accuracy bonus (up to 5 points per correct note)
        fitness += self.timing_accuracy * 5.0

        # Progress bonus: reward getting through more of the solo
        progress_ratio = self.note_index / len(GUITAR_SOLO)
        fitness += progress_ratio * 30.0

        # Small penalty for incorrect notes (to prefer quality over quantity)
        fitness -= self.notes_missed * 1.0

        # Big bonus for completing the solo
        if self.time_beats >= SOLO_DURATION:
            fitness += 50.0

        # Ensure fitness is non-negative
        fitness = max(0, fitness)

        return fitness

    def get_statistics(self):
        """Get detailed statistics about performance."""
        return {
            'correct_notes': self.correct_notes,
            'notes_missed': self.notes_missed,
            'total_attempted': self.total_notes_attempted,
            'accuracy': self.correct_notes / max(1, self.total_notes_attempted),
            'timing_accuracy': self.timing_accuracy / max(1, self.correct_notes),
            'progress': self.note_index / len(GUITAR_SOLO),
            'time_beats': self.time_beats,
            'fitness': self.get_fitness()
        }

    def render(self, screen, x_offset=0, y_offset=0, show_target=True):
        """
        Render the guitar player and fretboard.

        Args:
            screen: Pygame screen surface
            x_offset: X position offset
            y_offset: Y position offset
            show_target: Whether to show the target note
        """
        # Fretboard dimensions
        fretboard_width = 400
        fretboard_height = 300
        string_spacing = fretboard_height // 7
        fret_spacing = fretboard_width // 23

        # Draw fretboard background
        fretboard_rect = pygame.Rect(
            x_offset,
            y_offset,
            fretboard_width,
            fretboard_height
        )
        pygame.draw.rect(screen, (139, 90, 43), fretboard_rect)  # Wood color

        # Draw strings (horizontal lines)
        for i in range(1, 7):
            y = y_offset + i * string_spacing
            pygame.draw.line(
                screen,
                (200, 200, 200),
                (x_offset, y),
                (x_offset + fretboard_width, y),
                2
            )

        # Draw frets (vertical lines)
        for i in range(24):
            x = x_offset + i * fret_spacing
            pygame.draw.line(
                screen,
                (150, 150, 150),
                (x, y_offset),
                (x, y_offset + fretboard_height),
                1
            )

        # Draw fret markers at 3, 5, 7, 9, 12, 15, 17, 19, 21
        marker_frets = [3, 5, 7, 9, 12, 15, 17, 19, 21]
        for fret in marker_frets:
            x = x_offset + fret * fret_spacing
            y = y_offset + fretboard_height // 2
            pygame.draw.circle(screen, (100, 100, 100), (x, y), 5)

        # Show target note (if requested)
        if show_target:
            next_note_info = get_next_note_info(self.time_beats)
            if next_note_info:
                target_string, target_fret, _ = next_note_info
                x = x_offset + target_fret * fret_spacing
                y = y_offset + target_string * string_spacing
                pygame.draw.circle(screen, (0, 255, 0), (x, y), 12, 3)  # Green circle

        # Show current player position
        x = x_offset + self.current_fret * fret_spacing
        y = y_offset + self.current_string * string_spacing
        color = (255, 0, 0) if self.is_playing else (255, 100, 100)
        pygame.draw.circle(screen, color, (x, y), 10)  # Red/pink dot

        # Show recently played notes (fading)
        current_time = self.time_beats
        for note_string, note_fret, note_time in self.played_notes[-10:]:
            time_since = current_time - note_time
            if time_since < 1.0:  # Show for 1 beat
                alpha = int(255 * (1.0 - time_since))
                x = x_offset + note_fret * fret_spacing
                y = y_offset + note_string * string_spacing
                # Create surface for fading effect
                s = pygame.Surface((20, 20), pygame.SRCALPHA)
                pygame.draw.circle(s, (255, 255, 0, alpha), (10, 10), 8)
                screen.blit(s, (x - 10, y - 10))
