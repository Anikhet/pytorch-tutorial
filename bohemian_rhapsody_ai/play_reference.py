"""
Reference player - Shows the ideal guitar solo performance.
Plays the correct notes at the correct times with audio.
"""

import pygame
import sys
import time
from guitar_solo import (
    GUITAR_SOLO, SOLO_DURATION, BEAT_DURATION, NUM_NOTES,
    get_note_at_time, get_note_frequency
)
try:
    import audio_synth
    AUDIO_AVAILABLE = True
except ImportError:
    AUDIO_AVAILABLE = False
    print("Warning: Audio not available")

# Constants
SCREEN_WIDTH = 1200
SCREEN_HEIGHT = 800
FPS = 60


def draw_text(screen, text, x, y, size=24, color=(255, 255, 255)):
    """Helper function to draw text."""
    font = pygame.font.Font(None, size)
    text_surface = font.render(text, True, color)
    screen.blit(text_surface, (x, y))


def draw_fretboard(screen, x, y, width, height, current_notes, next_notes):
    """
    Draw a guitar fretboard with current and upcoming notes.

    Args:
        screen: Pygame screen
        x, y: Position
        width, height: Dimensions
        current_notes: List of currently playing (string, fret) tuples
        next_notes: List of upcoming (string, fret, time_until) tuples
    """
    string_spacing = height // 7
    fret_spacing = width // 23

    # Background
    pygame.draw.rect(screen, (139, 90, 43), (x, y, width, height))

    # Draw strings (horizontal)
    for i in range(1, 7):
        string_y = y + i * string_spacing
        pygame.draw.line(screen, (200, 200, 200), (x, string_y), (x + width, string_y), 2)

    # Draw frets (vertical)
    for i in range(24):
        fret_x = x + i * fret_spacing
        pygame.draw.line(screen, (150, 150, 150), (fret_x, y), (fret_x, y + height), 1)

    # Draw fret markers
    for fret in [3, 5, 7, 9, 12, 15, 17, 19, 21]:
        marker_x = x + fret * fret_spacing
        marker_y = y + height // 2
        pygame.draw.circle(screen, (100, 100, 100), (marker_x, marker_y), 5)

    # Draw upcoming notes (yellow outline)
    for string, fret, time_until in next_notes:
        note_x = x + fret * fret_spacing
        note_y = y + string * string_spacing
        alpha = int(255 * (1.0 - min(time_until, 2.0) / 2.0))  # Fade based on time
        pygame.draw.circle(screen, (255, 255, 0), (note_x, note_y), 12, 2)

    # Draw current notes (bright green filled)
    for string, fret, _ in current_notes:
        note_x = x + fret * fret_spacing
        note_y = y + string * string_spacing
        pygame.draw.circle(screen, (0, 255, 0), (note_x, note_y), 15)
        pygame.draw.circle(screen, (255, 255, 255), (note_x, note_y), 15, 2)


def draw_note_list(screen, x, y, current_time):
    """Draw a list of all notes with current position highlighted."""
    draw_text(screen, "Complete Solo Sequence", x, y, 28, (255, 255, 100))

    y_offset = y + 40
    current_note_index = -1

    # Find current note
    for i, (string, fret, duration, timing) in enumerate(GUITAR_SOLO):
        if timing <= current_time < timing + duration:
            current_note_index = i
            break
        elif timing > current_time:
            break

    # Show 10 notes at a time centered around current
    start_idx = max(0, current_note_index - 5)
    end_idx = min(len(GUITAR_SOLO), start_idx + 10)

    for i in range(start_idx, end_idx):
        string, fret, duration, timing = GUITAR_SOLO[i]
        freq = get_note_frequency(string, fret)

        # Highlight current note
        if i == current_note_index:
            color = (0, 255, 0)
            marker = "► "
        elif timing < current_time:
            color = (100, 100, 100)  # Past notes
            marker = "✓ "
        else:
            color = (200, 200, 200)  # Future notes
            marker = "  "

        text = f"{marker}#{i+1:2d}: String {string}, Fret {fret:2d} @ {timing:5.2f}s ({freq:.0f}Hz)"
        draw_text(screen, text, x, y_offset, 18, color)
        y_offset += 25


def draw_timeline(screen, x, y, width, height, current_time):
    """Draw a horizontal timeline of the solo."""
    # Background
    pygame.draw.rect(screen, (30, 30, 40), (x, y, width, height))
    draw_text(screen, "Timeline", x + 10, y + 5, 20, (200, 200, 200))

    timeline_y = y + 40

    # Time markers
    for beat in range(int(SOLO_DURATION) + 1):
        beat_x = x + 20 + int((beat / SOLO_DURATION) * (width - 40))
        pygame.draw.line(screen, (80, 80, 80), (beat_x, timeline_y), (beat_x, timeline_y + 40), 1)
        draw_text(screen, f"{beat}s", beat_x - 10, timeline_y + 45, 14, (100, 100, 100))

    # Current time marker
    current_x = x + 20 + int((current_time / SOLO_DURATION) * (width - 40))
    pygame.draw.line(screen, (255, 0, 0), (current_x, timeline_y), (current_x, timeline_y + 40), 3)

    # Draw all notes on timeline
    for i, (string, fret, duration, timing) in enumerate(GUITAR_SOLO):
        note_x = x + 20 + int((timing / SOLO_DURATION) * (width - 40))
        note_y = timeline_y + 10 + (6 - string) * 5

        if timing <= current_time:
            color = (0, 200, 0)  # Past
        else:
            color = (255, 255, 0)  # Future

        pygame.draw.circle(screen, color, (note_x, note_y), 4)


def main():
    """Main reference player."""
    # Initialize
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Guitar Solo Reference - Ideal Performance")
    clock = pygame.time.Clock()

    # Initialize audio
    audio_enabled = False
    if AUDIO_AVAILABLE:
        audio_synth.get_synth()
        audio_enabled = True
        print("🔊 Audio enabled!")

    # State
    current_time = 0.0
    paused = False
    last_played_notes = set()

    print("\n" + "="*60)
    print("GUITAR SOLO REFERENCE PLAYER")
    print("="*60)
    print("\nControls:")
    print("  SPACE: Pause/Resume")
    print("  R: Reset to beginning")
    print("  M: Toggle audio")
    print("  ESC: Quit")
    print("\nPlaying ideal guitar solo with perfect timing...")
    print()

    # Main loop
    running = True
    while running:
        dt = clock.tick(FPS) / 1000.0

        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_r:
                    current_time = 0.0
                    last_played_notes = set()
                    print("Reset to beginning")
                elif event.key == pygame.K_SPACE:
                    paused = not paused
                    print("Paused" if paused else "Resumed")
                elif event.key == pygame.K_m and AUDIO_AVAILABLE:
                    audio_enabled = not audio_enabled
                    print("🔊 Audio ON" if audio_enabled else "🔇 Audio OFF")

        # Update time
        if not paused:
            current_time += dt

            # Loop when finished
            if current_time >= SOLO_DURATION:
                current_time = 0.0
                last_played_notes = set()
                print("\n🎸 Solo complete! Looping...\n")

        # Get current notes and play audio
        current_notes = get_note_at_time(current_time / BEAT_DURATION)

        if audio_enabled and current_notes:
            # Play notes that just started
            current_note_set = set((s, f) for s, f, _ in current_notes)
            new_notes = current_note_set - last_played_notes

            for string, fret in new_notes:
                audio_synth.play_note(string, fret, duration=0.3)

            last_played_notes = current_note_set
        elif not current_notes:
            last_played_notes = set()

        # Get upcoming notes (within next 2 beats)
        next_notes = []
        current_beats = current_time / BEAT_DURATION
        for string, fret, duration, timing in GUITAR_SOLO:
            if current_beats < timing <= current_beats + 2.0:
                time_until = timing - current_beats
                next_notes.append((string, fret, time_until))

        # Render
        screen.fill((15, 15, 25))

        # Title
        draw_text(screen, "Reference Performance - Ideal Guitar Solo", 50, 20, 36, (255, 255, 100))

        # Stats
        progress = (current_time / SOLO_DURATION) * 100
        draw_text(screen, f"Time: {current_time:.2f}s / {SOLO_DURATION:.1f}s ({progress:.1f}%)",
                 50, 70, 24, (150, 255, 255))
        draw_text(screen, f"Total Notes: {NUM_NOTES}", 600, 70, 24, (150, 255, 255))

        # Fretboard
        draw_fretboard(screen, 50, 120, 600, 300, current_notes, next_notes)

        # Current note info
        if current_notes:
            info_y = 440
            draw_text(screen, "Currently Playing:", 50, info_y, 24, (0, 255, 0))
            for i, (string, fret, time_held) in enumerate(current_notes):
                freq = get_note_frequency(string, fret)
                text = f"  String {string}, Fret {fret} ({freq:.0f} Hz)"
                draw_text(screen, text, 50, info_y + 30 + i*25, 20, (0, 255, 0))

        # Note list
        draw_note_list(screen, 700, 120, current_time / BEAT_DURATION)

        # Timeline
        draw_timeline(screen, 50, SCREEN_HEIGHT - 150, SCREEN_WIDTH - 100, 100, current_time)

        # Audio indicator
        if AUDIO_AVAILABLE:
            audio_icon = "🔊" if audio_enabled else "🔇"
            color = (0, 255, 0) if audio_enabled else (150, 150, 150)
            draw_text(screen, audio_icon, SCREEN_WIDTH - 50, SCREEN_HEIGHT - 40, 24, color)

        # Controls
        draw_text(screen, "SPACE: Pause | R: Reset | M: Audio | ESC: Quit",
                 50, SCREEN_HEIGHT - 30, 18, (150, 150, 150))

        # Pause indicator
        if paused:
            draw_text(screen, "PAUSED", SCREEN_WIDTH // 2 - 60, 50, 48, (255, 100, 100))

        pygame.display.flip()

    pygame.quit()
    print("\nReference player closed.")


if __name__ == "__main__":
    main()
