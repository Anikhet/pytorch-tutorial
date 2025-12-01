"""
Enhanced training script with neural network visualization.
Shows the neural network's decision-making process in real-time.
"""

import pygame
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
from guitar_player import GuitarPlayer
from neural_network import create_random_network, GuitarNetwork
from genetic_algorithm import GeneticAlgorithm
from guitar_solo import SOLO_DURATION, BEAT_DURATION, NUM_NOTES

# Constants
SCREEN_WIDTH = 1600
SCREEN_HEIGHT = 900
FPS = 60
SIMULATION_SPEED = 1  # Slower for visualization

# Training parameters
POPULATION_SIZE = 6
MAX_GENERATION_TIME = SOLO_DURATION * BEAT_DURATION
MAX_GENERATIONS = 100


def draw_text(screen, text, x, y, size=24, color=(255, 255, 255)):
    """Helper function to draw text on screen."""
    font = pygame.font.Font(None, size)
    text_surface = font.render(text, True, color)
    screen.blit(text_surface, (x, y))


def draw_neural_network(screen, network, sensor_data, x_offset, y_offset):
    """
    Visualize the neural network architecture and activations.

    Args:
        screen: Pygame screen
        network: Neural network to visualize
        sensor_data: Current input sensor values
        x_offset: X position offset
        y_offset: Y position offset
    """
    # Get network activations
    with torch.no_grad():
        x = torch.FloatTensor(sensor_data)

        # Layer 1
        h1 = torch.relu(network.fc1(x))
        # Layer 2
        h2 = torch.relu(network.fc2(h1))
        # Output
        output = torch.tanh(network.fc3(h2))

    # Convert to numpy
    input_vals = sensor_data
    h1_vals = h1.numpy()
    h2_vals = h2.numpy()
    output_vals = output.numpy()

    # Drawing parameters
    layer_spacing = 150
    neuron_radius = 12
    input_spacing = 35
    hidden_spacing = 35
    output_spacing = 50

    # Draw title
    draw_text(screen, "Neural Network", x_offset, y_offset - 30, 24, (255, 255, 100))

    # Layer positions
    input_x = x_offset
    hidden1_x = x_offset + layer_spacing
    hidden2_x = x_offset + layer_spacing * 2
    output_x = x_offset + layer_spacing * 3

    # Calculate Y positions
    input_start_y = y_offset + 100
    hidden1_start_y = y_offset + 80
    hidden2_start_y = y_offset + 80
    output_start_y = y_offset + 180

    # Draw connections (weights) - only a sample to avoid clutter
    connection_alpha = 50
    for i in range(len(input_vals)):
        for j in range(len(h1_vals)):
            if (i + j) % 2 == 0:  # Draw every other connection
                start_pos = (input_x, input_start_y + i * input_spacing)
                end_pos = (hidden1_x, hidden1_start_y + j * hidden_spacing)
                pygame.draw.line(screen, (100, 100, 100), start_pos, end_pos, 1)

    # Draw INPUT layer
    draw_text(screen, "Input", input_x - 30, y_offset, 18, (150, 150, 255))
    input_labels = ["Time", "Next String", "Next Fret", "Time Until",
                   "Curr String", "Curr Fret", "Playing", "Progress"]

    for i, (val, label) in enumerate(zip(input_vals, input_labels)):
        y = input_start_y + i * input_spacing
        # Draw neuron
        color_intensity = int(255 * abs(val))
        color = (color_intensity, color_intensity, 255)
        pygame.draw.circle(screen, color, (input_x, y), neuron_radius)
        pygame.draw.circle(screen, (255, 255, 255), (input_x, y), neuron_radius, 1)

        # Draw value and label
        draw_text(screen, f"{val:.2f}", input_x + 20, y - 8, 14, (200, 200, 200))
        draw_text(screen, label, input_x - 100, y - 8, 12, (150, 150, 150))

    # Draw HIDDEN layer 1
    draw_text(screen, "Hidden 1", hidden1_x - 40, y_offset, 18, (150, 255, 150))
    for i, val in enumerate(h1_vals):
        y = hidden1_start_y + i * hidden_spacing
        # Draw neuron with activation color
        color_intensity = int(255 * min(abs(val), 1.0))
        color = (color_intensity, 255, color_intensity)
        pygame.draw.circle(screen, color, (hidden1_x, y), neuron_radius)
        pygame.draw.circle(screen, (255, 255, 255), (hidden1_x, y), neuron_radius, 1)

    # Draw HIDDEN layer 2
    draw_text(screen, "Hidden 2", hidden2_x - 40, y_offset, 18, (150, 255, 150))
    for i, val in enumerate(h2_vals):
        y = hidden2_start_y + i * hidden_spacing
        # Draw neuron with activation color
        color_intensity = int(255 * min(abs(val), 1.0))
        color = (color_intensity, 255, color_intensity)
        pygame.draw.circle(screen, color, (hidden2_x, y), neuron_radius)
        pygame.draw.circle(screen, (255, 255, 255), (hidden2_x, y), neuron_radius, 1)

    # Draw OUTPUT layer
    draw_text(screen, "Output", output_x - 30, y_offset, 18, (255, 150, 150))
    output_labels = ["String", "Fret", "Play"]

    for i, (val, label) in enumerate(zip(output_vals, output_labels)):
        y = output_start_y + i * output_spacing
        # Draw neuron
        if val > 0:
            color = (255, int(255 * val), 0)
        else:
            color = (int(255 * abs(val)), 0, 255)
        pygame.draw.circle(screen, color, (output_x, y), neuron_radius + 2)
        pygame.draw.circle(screen, (255, 255, 255), (output_x, y), neuron_radius + 2, 2)

        # Draw value and label
        draw_text(screen, f"{val:.2f}", output_x + 25, y - 8, 16, (255, 255, 100))
        draw_text(screen, label, output_x - 60, y - 8, 14, (200, 200, 200))

        # Draw interpretation
        if label == "String":
            string_val = int((val + 1) * 3) + 1
            draw_text(screen, f"→ {np.clip(string_val, 1, 6)}", output_x + 80, y - 8, 14, (150, 255, 150))
        elif label == "Fret":
            fret_val = int((val + 1) * 11)
            draw_text(screen, f"→ {np.clip(fret_val, 0, 22)}", output_x + 80, y - 8, 14, (150, 255, 150))
        elif label == "Play":
            play_status = "YES" if val > 0.3 else "no"
            color = (0, 255, 0) if val > 0.3 else (100, 100, 100)
            draw_text(screen, f"→ {play_status}", output_x + 80, y - 8, 14, color)


def run_generation(screen, clock, players):
    """
    Run one generation with enhanced visualization.

    Args:
        screen: Pygame screen
        clock: Pygame clock
        players: List of GuitarPlayer instances

    Returns:
        List of fitness scores
    """
    generation_time = 0
    running = True
    focused_player = 0  # Which player to show detailed visualization for

    while running and generation_time < MAX_GENERATION_TIME:
        dt = clock.tick(FPS) / 1000.0 * SIMULATION_SPEED

        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return None
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return None
                elif event.key == pygame.K_SPACE:
                    running = False
                elif pygame.K_1 <= event.key <= pygame.K_9:
                    # Switch focused player (1-9 keys)
                    idx = event.key - pygame.K_1
                    if idx < len(players):
                        focused_player = idx

        # Update all players
        for player in players:
            if player.alive:
                player.update(dt)

        # Auto-focus on best performer
        if generation_time > 1.0:  # After 1 second
            best_idx = max(range(len(players)), key=lambda i: players[i].get_fitness())
            focused_player = best_idx

        # Check if all done
        all_done = all(not player.alive or player.time_beats >= SOLO_DURATION
                       for player in players)
        if all_done:
            # Generation complete - return fitness scores
            return [player.get_fitness() for player in players]

        # Render
        screen.fill((15, 15, 25))

        # Left side: Focused player detail
        focused = players[focused_player]
        draw_text(screen, f"FOCUSED: Player {focused_player + 1}",
                 50, 20, 32, (255, 255, 100))

        # Show fretboard for focused player
        focused.render(screen, 50, 70, show_target=True)

        # Show statistics for focused player
        stats = focused.get_statistics()
        stats_x = 500
        stats_y = 70
        draw_text(screen, "Statistics:", stats_x, stats_y, 24, (150, 255, 255))
        draw_text(screen, f"Fitness: {stats['fitness']:.1f}", stats_x, stats_y + 30, 20)
        draw_text(screen, f"Correct: {stats['correct_notes']}/{NUM_NOTES}", stats_x, stats_y + 55, 20)
        draw_text(screen, f"Missed: {stats['notes_missed']}", stats_x, stats_y + 80, 20)
        draw_text(screen, f"Accuracy: {stats['accuracy']*100:.1f}%", stats_x, stats_y + 105, 20,
                 color=(0, 255, 0) if stats['accuracy'] > 0.5 else (255, 100, 100))
        draw_text(screen, f"Progress: {stats['progress']*100:.0f}%", stats_x, stats_y + 130, 20)
        draw_text(screen, f"Time: {stats['time_beats']:.1f} beats", stats_x, stats_y + 155, 20)

        # Show neural network visualization
        sensor_data = focused.get_sensor_data()
        draw_neural_network(screen, focused.network, sensor_data, 800, 50)

        # Bottom: Small views of all players
        bottom_y = SCREEN_HEIGHT - 250
        pygame.draw.rect(screen, (30, 30, 40), (0, bottom_y - 10, SCREEN_WIDTH, 260))
        draw_text(screen, "All Players:", 20, bottom_y - 35, 20, (200, 200, 200))

        cell_width = SCREEN_WIDTH // len(players)
        for i, player in enumerate(players):
            x = i * cell_width + 10
            y = bottom_y

            # Highlight focused player
            if i == focused_player:
                pygame.draw.rect(screen, (80, 80, 100),
                               (x - 5, y - 5, cell_width - 10, 210), 3)

            # Player info
            color = (100, 255, 100) if player.alive else (100, 100, 100)
            draw_text(screen, f"P{i+1}", x, y, 18, color)
            draw_text(screen, f"F: {player.get_fitness():.0f}", x, y + 20, 14, color)

            # Mini fretboard
            player.render(screen, x, y + 40, show_target=False)

        # Top bar: Generation info
        draw_text(screen, f"Time: {generation_time:.1f}s / {MAX_GENERATION_TIME:.1f}s",
                 20, SCREEN_HEIGHT - 30, 20)
        alive_count = sum(1 for p in players if p.alive)
        draw_text(screen, f"Alive: {alive_count}/{len(players)}",
                 SCREEN_WIDTH - 200, SCREEN_HEIGHT - 30, 20)

        # Instructions
        draw_text(screen, "ESC: Quit | SPACE: Next Gen | 1-9: Focus Player",
                 SCREEN_WIDTH // 2 - 250, SCREEN_HEIGHT - 30, 18, (150, 150, 150))

        pygame.display.flip()
        generation_time += dt

    if running is False or generation_time >= MAX_GENERATION_TIME:
        return [player.get_fitness() for player in players]
    return None


def save_fitness_plot(ga):
    """Save fitness evolution plot."""
    stats = ga.get_statistics()

    # Only save if we have data
    if len(stats['best_fitness_history']) == 0:
        print("No fitness data to plot (no generations completed)")
        return

    plt.figure(figsize=(10, 6))
    plt.plot(stats['best_fitness_history'], label='Best Fitness', linewidth=2)
    plt.plot(stats['avg_fitness_history'], label='Average Fitness', linewidth=2)
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.title('Guitar Learning AI - Fitness Evolution (Visual Mode)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('fitness_history_visual.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Fitness plot saved to fitness_history_visual.png")


def main():
    """Main training loop with visualization."""
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Guitar Learning AI - Training (Visual Mode)")
    clock = pygame.time.Clock()

    print(f"Initializing Guitar Learning AI (Visual Mode)...")
    print(f"  Population size: {POPULATION_SIZE}")
    print(f"  Max generations: {MAX_GENERATIONS}")
    print(f"  Simulation speed: {SIMULATION_SPEED}x")
    print()

    ga = GeneticAlgorithm(
        population_size=POPULATION_SIZE,
        elite_size=1,
        mutation_rate=0.1
    )

    try:
        for generation in range(MAX_GENERATIONS):
            print(f"\n{'='*60}")
            print(f"GENERATION {generation}")
            print('='*60)

            players = [GuitarPlayer(network) for network in ga.get_population()]

            fitness_scores = run_generation(screen, clock, players)

            if fitness_scores is None:
                print("\nTraining interrupted by user")
                break

            # Show results
            print(f"\nGeneration {generation} Results:")
            for i, (player, fitness) in enumerate(zip(players, fitness_scores)):
                stats = player.get_statistics()
                print(f"  Player {i+1}: Fitness={fitness:.1f}, "
                      f"Correct={stats['correct_notes']}, Progress={stats['progress']*100:.1f}%")

            ga.evolve(fitness_scores)

            if (generation + 1) % 10 == 0:
                checkpoint_path = f"best_network_gen_{generation+1}.pth"
                ga.save_best(checkpoint_path)

    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user (Ctrl+C)")

    finally:
        print("\n\nTraining complete!")
        ga.save_best("best_network_final_visual.pth")
        save_fitness_plot(ga)

        stats = ga.get_statistics()
        print(f"\nFinal Statistics:")
        print(f"  Generations: {stats['generation']}")

        if stats['best_fitness'] > -float('inf'):
            print(f"  Best fitness: {stats['best_fitness']:.2f}")
        else:
            print(f"  Best fitness: N/A (no generations completed)")

        if len(stats['avg_fitness_history']) > 0:
            print(f"  Initial avg fitness: {stats['avg_fitness_history'][0]:.2f}")
            print(f"  Final avg fitness: {stats['avg_fitness_history'][-1]:.2f}")

        pygame.quit()


if __name__ == "__main__":
    main()
