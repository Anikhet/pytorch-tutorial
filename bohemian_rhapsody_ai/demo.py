"""
Demo script to watch a trained network play the guitar solo.
"""

import pygame
import sys
import os
import torch
from guitar_player import GuitarPlayer
from neural_network import GuitarNetwork
from guitar_solo import SOLO_DURATION, BEAT_DURATION, NUM_NOTES, get_note_at_time
try:
    import audio_synth
    AUDIO_AVAILABLE = True
except ImportError:
    AUDIO_AVAILABLE = False
    print("Warning: Audio module not available")

# Constants
SCREEN_WIDTH = 1200
SCREEN_HEIGHT = 700
FPS = 60


def draw_text(screen, text, x, y, size=24, color=(255, 255, 255)):
    """Helper function to draw text."""
    font = pygame.font.Font(None, size)
    text_surface = font.render(text, True, color)
    screen.blit(text_surface, (x, y))


def draw_progress_bar(screen, x, y, width, height, progress, color=(0, 255, 0)):
    """Draw a progress bar."""
    pygame.draw.rect(screen, (50, 50, 50), (x, y, width, height))
    filled_width = int(width * progress)
    pygame.draw.rect(screen, color, (x, y, filled_width, height))
    pygame.draw.rect(screen, (255, 255, 255), (x, y, width, height), 2)


def draw_note_tracker(screen, current_time, x, y, width, height):
    """
    Draw a visual tracker showing which notes should be played.

    Args:
        screen: Pygame screen
        current_time: Current time in beats
        x, y: Position
        width, height: Dimensions
    """
    # Background
    pygame.draw.rect(screen, (30, 30, 40), (x, y, width, height))
    draw_text(screen, "Note Timeline", x + 10, y + 5, 20, (200, 200, 200))

    # Draw timeline
    time_window = 4.0  # Show 4 beats ahead
    start_time = max(0, current_time - 1.0)
    end_time = current_time + time_window

    # Time markers
    for i in range(int(start_time), int(end_time) + 1):
        if i < 0 or i > SOLO_DURATION:
            continue
        progress = (i - start_time) / time_window
        marker_x = x + int(progress * width)
        pygame.draw.line(screen, (80, 80, 80),
                        (marker_x, y + 30), (marker_x, y + height - 10), 1)
        draw_text(screen, str(i), marker_x - 5, y + height - 25, 14, (100, 100, 100))

    # Draw current time line
    current_progress = (current_time - start_time) / time_window
    if 0 <= current_progress <= 1:
        current_x = x + int(current_progress * width)
        pygame.draw.line(screen, (255, 0, 0),
                        (current_x, y + 30), (current_x, y + height - 10), 3)

    # Draw notes in window
    from guitar_solo import GUITAR_SOLO
    for string, fret, duration, note_time in GUITAR_SOLO:
        if start_time <= note_time <= end_time:
            # Calculate position
            note_progress = (note_time - start_time) / time_window
            note_x = x + int(note_progress * width)
            note_y = y + 35 + int((6 - string) * 15)  # String position

            # Draw note marker
            if abs(note_time - current_time) < 0.2:
                # Note should be played now - bright color
                color = (0, 255, 0)
                radius = 8
            elif note_time < current_time:
                # Past note - dim color
                color = (100, 100, 100)
                radius = 5
            else:
                # Future note - yellow
                color = (255, 255, 0)
                radius = 6

            pygame.draw.circle(screen, color, (note_x, note_y), radius)

            # Draw note duration
            duration_width = int((duration / time_window) * width)
            pygame.draw.rect(screen, (*color, 50),
                           (note_x, note_y - 3, duration_width, 6))


def demo_network(network_path):
    """
    Run demo of trained network.

    Args:
        network_path: Path to saved network file
    """
    # Load network
    print(f"Loading network from {network_path}...")
    network = GuitarNetwork()
    network.load(network_path)
    print("Network loaded successfully!")

    # Initialize Pygame
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Guitar Learning AI - Demo")
    clock = pygame.time.Clock()

    # Create player with audio enabled
    player = GuitarPlayer(network, audio_enabled=True)

    # Initialize audio synth if available
    if AUDIO_AVAILABLE:
        audio_synth.get_synth()  # Initialize and pre-cache sounds
        print("🔊 Audio enabled!")
    else:
        print("🔇 Audio not available")

    # Demo loop
    running = True
    paused = False
    audio_enabled = AUDIO_AVAILABLE

    print("\nDemo Controls:")
    print("  ESC: Quit")
    print("  R: Reset to beginning")
    print("  SPACE: Pause/Resume")
    print("  M: Toggle audio on/off")
    print()

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
                    player.reset()
                    print("Reset to beginning")
                elif event.key == pygame.K_SPACE:
                    paused = not paused
                    print("Paused" if paused else "Resumed")
                elif event.key == pygame.K_m:
                    if AUDIO_AVAILABLE:
                        player.audio_enabled = not player.audio_enabled
                        audio_enabled = player.audio_enabled
                        print("🔊 Audio ON" if audio_enabled else "🔇 Audio OFF")

        # Update player (if not paused)
        if not paused and player.alive:
            player.update(dt)

        # Check if finished
        if player.time_beats >= SOLO_DURATION:
            # Auto-reset after finishing
            pygame.time.wait(2000)  # Wait 2 seconds
            player.reset()

        # Render
        screen.fill((15, 15, 25))

        # Title
        draw_text(screen, "Guitar Learning AI - Demo", 50, 20, 36, (255, 255, 100))

        # Draw fretboard
        player.render(screen, 50, 80, show_target=True)

        # Statistics panel
        stats = player.get_statistics()
        stats_x = 500
        stats_y = 80

        pygame.draw.rect(screen, (30, 30, 40),
                        (stats_x - 10, stats_y - 10, 300, 280), border_radius=10)

        draw_text(screen, "Performance Stats", stats_x, stats_y, 24, (150, 255, 255))
        draw_text(screen, f"Fitness: {stats['fitness']:.1f}", stats_x, stats_y + 35, 22)
        draw_text(screen, f"Correct Notes: {stats['correct_notes']}/{NUM_NOTES}",
                 stats_x, stats_y + 65, 20)
        draw_text(screen, f"Missed: {stats['notes_missed']}", stats_x, stats_y + 95, 20)

        # Accuracy with color
        accuracy = stats['accuracy'] * 100
        acc_color = (0, 255, 0) if accuracy > 70 else (255, 255, 0) if accuracy > 40 else (255, 100, 100)
        draw_text(screen, f"Accuracy: {accuracy:.1f}%", stats_x, stats_y + 125, 20, acc_color)

        # Progress
        draw_text(screen, f"Progress: {stats['progress']*100:.0f}%", stats_x, stats_y + 155, 20)
        draw_text(screen, f"Time: {stats['time_beats']:.1f}/{SOLO_DURATION:.1f} beats",
                 stats_x, stats_y + 185, 18)

        # Playing status
        status = "PLAYING" if player.is_playing else "Silent"
        status_color = (0, 255, 0) if player.is_playing else (150, 150, 150)
        draw_text(screen, f"Status: {status}", stats_x, stats_y + 220, 20, status_color)

        # Current position
        draw_text(screen, f"String: {player.current_string}, Fret: {player.current_fret}",
                 stats_x, stats_y + 245, 18, (200, 200, 255))

        # Note tracker
        draw_note_tracker(screen, player.time_beats, 50, 420, 750, 200)

        # Progress bar
        progress = player.time_beats / SOLO_DURATION
        draw_progress_bar(screen, 50, SCREEN_HEIGHT - 80, SCREEN_WIDTH - 100, 30, progress)
        draw_text(screen, f"Solo Progress: {progress*100:.1f}%",
                 SCREEN_WIDTH // 2 - 100, SCREEN_HEIGHT - 105, 20)

        # Controls reminder
        draw_text(screen, "ESC: Quit | R: Reset | SPACE: Pause | M: Audio",
                 50, SCREEN_HEIGHT - 30, 18, (150, 150, 150))

        # Audio indicator
        if AUDIO_AVAILABLE:
            audio_icon = "🔊" if audio_enabled else "🔇"
            audio_color = (0, 255, 0) if audio_enabled else (150, 150, 150)
            draw_text(screen, audio_icon, SCREEN_WIDTH - 50, SCREEN_HEIGHT - 30, 24, audio_color)

        # Pause indicator
        if paused:
            draw_text(screen, "PAUSED", SCREEN_WIDTH // 2 - 60, 50, 48, (255, 100, 100))

        pygame.display.flip()

    pygame.quit()
    print("\nDemo finished!")

    # Print final stats
    print(f"\nFinal Statistics:")
    print(f"  Correct notes: {stats['correct_notes']}/{NUM_NOTES}")
    print(f"  Accuracy: {stats['accuracy']*100:.1f}%")
    print(f"  Fitness: {stats['fitness']:.1f}")


def main():
    """Main entry point."""
    if len(sys.argv) > 1:
        network_path = sys.argv[1]
    else:
        # Default to the final trained network
        network_path = "best_network_final.pth"

    # Check if file exists
    if not os.path.exists(network_path):
        print(f"Error: Network file not found: {network_path}")
        print("\nAvailable networks:")
        # List all .pth files in current directory
        pth_files = [f for f in os.listdir('.') if f.endswith('.pth')]
        if pth_files:
            for file in pth_files:
                print(f"  {file}")
        else:
            print("  No networks found. Train a network first using main.py")
        sys.exit(1)

    demo_network(network_path)


if __name__ == "__main__":
    main()
