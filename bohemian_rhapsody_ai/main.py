"""
Main training script for the Guitar Learning AI.
Trains neural networks to play guitar using genetic algorithms.
"""

import pygame
import sys
import numpy as np
import matplotlib.pyplot as plt
from guitar_player import GuitarPlayer
from neural_network import create_random_network
from genetic_algorithm import GeneticAlgorithm
from guitar_solo import SOLO_DURATION, BEAT_DURATION, NUM_NOTES

# Constants
SCREEN_WIDTH = 1400
SCREEN_HEIGHT = 900
FPS = 60
SIMULATION_SPEED = 2  # Speed multiplier

# Training parameters
POPULATION_SIZE = 8
MAX_GENERATION_TIME = SOLO_DURATION * BEAT_DURATION  # Full solo duration
MAX_GENERATIONS = 100


def draw_text(screen, text, x, y, size=24, color=(255, 255, 255)):
    """Helper function to draw text on screen."""
    font = pygame.font.Font(None, size)
    text_surface = font.render(text, True, color)
    screen.blit(text_surface, (x, y))


def draw_progress_bar(screen, x, y, width, height, progress, color=(0, 255, 0)):
    """Draw a progress bar."""
    # Background
    pygame.draw.rect(screen, (50, 50, 50), (x, y, width, height))
    # Progress
    filled_width = int(width * progress)
    pygame.draw.rect(screen, color, (x, y, filled_width, height))
    # Border
    pygame.draw.rect(screen, (255, 255, 255), (x, y, width, height), 2)


def run_generation(screen, clock, players):
    """
    Run one generation of the simulation.

    Args:
        screen: Pygame screen
        clock: Pygame clock
        players: List of GuitarPlayer instances

    Returns:
        List of fitness scores for each player
    """
    generation_time = 0
    running = True

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
                    # Skip to next generation
                    running = False

        # Update all players
        for player in players:
            if player.alive:
                player.update(dt)

        # Check if all players are done or reached the end
        all_done = all(not player.alive or player.time_beats >= SOLO_DURATION
                       for player in players)
        if all_done:
            # Generation complete - return fitness scores
            return [player.get_fitness() for player in players]

        # Render
        screen.fill((20, 20, 30))  # Dark background

        # Calculate grid layout for players
        cols = 4
        rows = (len(players) + cols - 1) // cols
        cell_width = SCREEN_WIDTH // cols
        cell_height = 400

        # Draw each player's fretboard
        for i, player in enumerate(players):
            row = i // cols
            col = i % cols
            x_offset = col * cell_width + 20
            y_offset = row * cell_height + 50

            # Draw player number and fitness
            fitness = player.get_fitness()
            draw_text(screen, f"Player {i+1}", x_offset, y_offset - 30, 20)
            draw_text(screen, f"Fitness: {fitness:.1f}", x_offset, y_offset - 10, 16,
                     color=(100, 255, 100) if player.alive else (100, 100, 100))

            # Draw fretboard
            if player.alive or player.time_beats >= SOLO_DURATION:
                player.render(screen, x_offset, y_offset, show_target=True)

        # Draw generation info at bottom
        info_y = SCREEN_HEIGHT - 150

        # Progress bar for generation time
        progress = generation_time / MAX_GENERATION_TIME
        draw_progress_bar(screen, 50, info_y, 400, 30, progress)
        draw_text(screen, f"Time: {generation_time:.1f}s / {MAX_GENERATION_TIME:.1f}s",
                 460, info_y + 5, 20)

        # Show alive count
        alive_count = sum(1 for p in players if p.alive)
        draw_text(screen, f"Alive: {alive_count}/{len(players)}",
                 50, info_y + 40, 24)

        # Show average fitness
        avg_fitness = np.mean([p.get_fitness() for p in players])
        draw_text(screen, f"Avg Fitness: {avg_fitness:.1f}",
                 50, info_y + 70, 24)

        # Show best player stats
        best_player = max(players, key=lambda p: p.get_fitness())
        stats = best_player.get_statistics()
        draw_text(screen, f"Best: {stats['correct_notes']}/{NUM_NOTES} notes",
                 50, info_y + 100, 24, color=(0, 255, 0))

        # Instructions
        draw_text(screen, "ESC: Quit | SPACE: Next Generation",
                 SCREEN_WIDTH - 400, info_y + 5, 20, color=(150, 150, 150))

        pygame.display.flip()
        generation_time += dt

    # Return fitness scores
    if running is False or generation_time >= MAX_GENERATION_TIME:
        return [player.get_fitness() for player in players]
    return None


def save_fitness_plot(ga):
    """Save a plot of fitness over generations."""
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
    plt.title('Guitar Learning AI - Fitness Evolution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('fitness_history.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Fitness plot saved to fitness_history.png")


def main():
    """Main training loop."""
    # Initialize Pygame
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Guitar Learning AI - Training")
    clock = pygame.time.Clock()

    # Initialize genetic algorithm
    print(f"Initializing Genetic Algorithm...")
    print(f"  Population size: {POPULATION_SIZE}")
    print(f"  Max generations: {MAX_GENERATIONS}")
    print(f"  Solo duration: {MAX_GENERATION_TIME:.1f} seconds")
    print(f"  Total notes: {NUM_NOTES}")
    print()

    ga = GeneticAlgorithm(
        population_size=POPULATION_SIZE,
        elite_size=2,
        mutation_rate=0.1
    )

    # Training loop
    try:
        for generation in range(MAX_GENERATIONS):
            print(f"\n{'='*60}")
            print(f"GENERATION {generation}")
            print('='*60)

            # Create players for current population
            players = []
            for network in ga.get_population():
                player = GuitarPlayer(network)
                players.append(player)

            # Run generation
            fitness_scores = run_generation(screen, clock, players)

            if fitness_scores is None:
                # User quit
                print("\nTraining interrupted by user")
                break

            # Show detailed stats
            print(f"\nGeneration {generation} Results:")
            for i, (player, fitness) in enumerate(zip(players, fitness_scores)):
                stats = player.get_statistics()
                print(f"  Player {i+1}: Fitness={fitness:.1f}, "
                      f"Correct={stats['correct_notes']}/{NUM_NOTES}, "
                      f"Progress={stats['progress']*100:.1f}%")

            # Evolve population
            best_network = ga.evolve(fitness_scores)

            # Save checkpoint every 10 generations
            if (generation + 1) % 10 == 0:
                checkpoint_path = f"best_network_gen_{generation+1}.pth"
                ga.save_best(checkpoint_path)

            # Check for convergence (optional early stopping)
            stats = ga.get_statistics()
            if len(stats['best_fitness_history']) >= 20:
                recent_best = stats['best_fitness_history'][-20:]
                if max(recent_best) - min(recent_best) < 5:
                    print("\n\nConverged! Fitness has plateaued.")
                    break

    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user (Ctrl+C)")

    finally:
        # Save final results
        print("\n\nTraining complete!")
        ga.save_best("best_network_final.pth")
        save_fitness_plot(ga)

        # Print final statistics
        stats = ga.get_statistics()
        print(f"\nFinal Statistics:")
        print(f"  Generations trained: {stats['generation']}")

        if stats['best_fitness'] > -float('inf'):
            print(f"  Best fitness: {stats['best_fitness']:.2f}")
        else:
            print(f"  Best fitness: N/A (no generations completed)")

        if len(stats['avg_fitness_history']) > 0:
            print(f"  Initial avg fitness: {stats['avg_fitness_history'][0]:.2f}")
            print(f"  Final avg fitness: {stats['avg_fitness_history'][-1]:.2f}")
        else:
            print(f"  No fitness data (training interrupted before first generation)")

        pygame.quit()


if __name__ == "__main__":
    main()
