import pygame
import numpy as np
import matplotlib.pyplot as plt
from track import Track
from car import Car
from genetic_algorithm import GeneticAlgorithm
import sys

# Configuration
SCREEN_WIDTH = 1200
SCREEN_HEIGHT = 800
FPS = 60
SIMULATION_SPEED = 4  # How many physics updates per frame (1=normal, 2=2x speed, 4=4x speed, etc.)
MAX_GENERATION_TIME = 5  # seconds per generation (reduced from 15)
POPULATION_SIZE = 5  # Reduced from 50 for better visibility

# Colors
BACKGROUND_COLOR = (34, 139, 34)
TEXT_COLOR = (255, 255, 255)

def run_generation(track, genetic_algorithm, screen, clock, generation_num):
    """
    Run one generation of evolution.

    Args:
        track: The race track
        genetic_algorithm: GA instance
        screen: Pygame screen
        clock: Pygame clock
        generation_num: Current generation number

    Returns:
        List of fitness scores
    """
    # Get population of neural networks
    population = genetic_algorithm.get_population()

    # Create cars for each network
    cars = []
    for i, network in enumerate(population):
        # Vary colors for visual diversity
        hue = (i / len(population)) * 360
        color = pygame.Color(0)
        color.hsva = (hue, 100, 100, 100)
        car = Car(track.start_pos[0], track.start_pos[1],
                 track.start_angle, track, color=color[:3])
        cars.append(car)

    # Simulation loop
    frame_count = 0
    max_frames = MAX_GENERATION_TIME * FPS

    running = True
    while running and frame_count < max_frames:
        clock.tick(FPS)
        frame_count += 1

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

        # Update all cars (multiple times per frame for faster simulation)
        alive_count = 0
        for _ in range(SIMULATION_SPEED):
            alive_count = 0
            for i, car in enumerate(cars):
                if car.alive:
                    alive_count += 1
                    # Get sensor data
                    sensor_data = car.get_sensor_data()

                    # Get action from neural network
                    action = population[i].predict(sensor_data)

                    # Update car
                    car.update(action)

            # If all cars are dead, end generation early
            if alive_count == 0:
                running = False
                break

        # Render
        track.render(screen)

        # Draw all cars
        for car in cars:
            car.render(screen, show_sensors=False)

        # Draw best car's sensors (first car is elite from previous gen)
        if cars[0].alive:
            cars[0]._render_sensors(screen)

        # Draw UI
        _draw_ui(screen, generation_num, alive_count, len(cars), frame_count, max_frames)

        pygame.display.flip()

    # Calculate fitness scores
    fitness_scores = [car.fitness for car in cars]

    return fitness_scores


def _draw_ui(screen, generation, alive, total, frame, max_frames):
    """Draw UI information on screen."""
    font = pygame.font.Font(None, 36)
    small_font = pygame.font.Font(None, 24)

    # Generation info
    gen_text = font.render(f"Generation: {generation}", True, TEXT_COLOR)
    screen.blit(gen_text, (10, 10))

    # Alive cars
    alive_text = font.render(f"Alive: {alive}/{total}", True, TEXT_COLOR)
    screen.blit(alive_text, (10, 50))

    # Speed indicator
    speed_text = small_font.render(f"Speed: {SIMULATION_SPEED}x", True, (0, 255, 255))
    screen.blit(speed_text, (10, 85))

    # Progress bar
    progress = frame / max_frames
    bar_width = 300
    bar_height = 20
    bar_x, bar_y = 10, 115

    # Background
    pygame.draw.rect(screen, (50, 50, 50), (bar_x, bar_y, bar_width, bar_height))
    # Progress
    pygame.draw.rect(screen, (0, 255, 0), (bar_x, bar_y, int(bar_width * progress), bar_height))
    # Border
    pygame.draw.rect(screen, TEXT_COLOR, (bar_x, bar_y, bar_width, bar_height), 2)

    # Instructions
    instructions = [
        "ESC: Quit",
        "SPACE: Skip to next generation"
    ]
    for i, instruction in enumerate(instructions):
        inst_text = small_font.render(instruction, True, TEXT_COLOR)
        screen.blit(inst_text, (10, SCREEN_HEIGHT - 60 + i * 25))


def plot_fitness_history(genetic_algorithm):
    """Plot fitness history over generations."""
    stats = genetic_algorithm.get_statistics()

    plt.figure(figsize=(10, 6))
    plt.plot(stats['best_fitness'], label='Best Fitness', linewidth=2)
    plt.plot(stats['avg_fitness'], label='Average Fitness', linewidth=2)
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.title('Evolution Progress')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('fitness_history.png')
    print("Fitness history plot saved to fitness_history.png")


def main():
    """Main training loop."""
    # Initialize Pygame
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Racing Car Genetic Algorithm")
    clock = pygame.time.Clock()

    # Create track
    track = Track(SCREEN_WIDTH, SCREEN_HEIGHT)

    # Initialize genetic algorithm
    ga = GeneticAlgorithm(
        population_size=POPULATION_SIZE,
        elite_size=1,  # Keep best car, reduced from 5 for smaller population
        mutation_rate=0.1
    )

    print("=" * 50)
    print("Racing Car Genetic Algorithm Training")
    print("=" * 50)
    print(f"Population Size: {POPULATION_SIZE}")
    print(f"Max Generation Time: {MAX_GENERATION_TIME}s")
    print("\nControls:")
    print("  ESC: Quit")
    print("  SPACE: Skip to next generation")
    print("=" * 50)

    # Training loop
    try:
        generation = 0
        while True:
            # Run one generation
            fitness_scores = run_generation(track, ga, screen, clock, generation)

            # Check if user quit
            if fitness_scores is None:
                break

            # Evolve to next generation
            best_network = ga.evolve(fitness_scores)

            # Save best network every 10 generations
            if generation % 10 == 0:
                ga.save_best(f'best_network_gen_{generation}.pth')

            generation += 1

    except KeyboardInterrupt:
        print("\nTraining interrupted by user")

    # Cleanup
    print("\nTraining finished!")
    print(f"Total generations: {ga.generation}")

    # Save final best network
    ga.save_best('best_network_final.pth')

    # Plot fitness history
    if ga.best_fitness_history:
        plot_fitness_history(ga)

    pygame.quit()


if __name__ == "__main__":
    main()
