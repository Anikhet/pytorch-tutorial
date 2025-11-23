import pygame
import matplotlib.pyplot as plt
from genetic_algorithm import GeneticAlgorithm
from pipe import PipeManager

# Configuration
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
FPS = 60
POPULATION_SIZE = 50
ELITE_SIZE = 5
MUTATION_RATE = 0.1
MAX_GENERATIONS = 100

# Colors
SKY_BLUE = (135, 206, 250)
GROUND_COLOR = (222, 216, 149)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

def run_generation(ga, pipe_manager, screen, clock, show_all=True):
    """
    Run one generation of birds.

    Args:
        ga: GeneticAlgorithm instance
        pipe_manager: PipeManager instance
        screen: Pygame screen
        clock: Pygame clock
        show_all: Whether to show all birds or just best

    Returns:
        fitness_scores: List of fitness values
    """
    # Reset pipes
    pipe_manager.reset()

    # Get population
    birds = ga.get_population()

    # Reset all birds
    for bird in birds:
        bird.y = 300
        bird.velocity = 0
        bird.alive = True
        bird.score = 0
        bird.frames_alive = 0
        bird.fitness = 0

    # Game loop
    running = True
    frame_count = 0

    while running:
        clock.tick(FPS)
        frame_count += 1

        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return None
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    return None
                elif event.key == pygame.K_SPACE:
                    # Skip to next generation
                    running = False

        # Update pipes
        pipe_manager.update()

        # Update all alive birds
        alive_count = 0
        for bird in birds:
            if bird.alive:
                alive_count += 1

                # Bird thinks and decides to jump
                if bird.think(pipe_manager.get_pipes()):
                    bird.jump()

                # Update bird physics
                bird.update()

                # Check collisions
                if pipe_manager.check_collisions(bird):
                    bird.alive = False

                # Check scoring
                if pipe_manager.check_score(bird):
                    bird.score += 1

        # If all birds are dead, end generation
        if alive_count == 0:
            running = False

        # Render
        screen.fill(SKY_BLUE)

        # Draw pipes
        pipe_manager.render(screen)

        # Draw ground
        ground_height = 50
        pygame.draw.rect(screen, GROUND_COLOR,
                        (0, SCREEN_HEIGHT - ground_height, SCREEN_WIDTH, ground_height))

        # Draw birds
        if show_all:
            # Show all alive birds (transparent)
            for bird in birds:
                if bird.alive:
                    bird.render(screen)
        else:
            # Show only best bird from previous generation
            if birds[0].alive:
                birds[0].render(screen)

        # Draw UI
        _draw_ui(screen, ga.generation, alive_count, len(birds), frame_count)

        pygame.display.flip()

    # Calculate fitness for all birds
    fitness_scores = []
    for bird in birds:
        bird.calculate_fitness()
        fitness_scores.append(bird.fitness)

    return fitness_scores

def _draw_ui(screen, generation, alive, total, frames):
    """Draw UI information."""
    font = pygame.font.Font(None, 32)
    small_font = pygame.font.Font(None, 24)

    # Generation
    gen_text = font.render(f"Generation: {generation}", True, BLACK)
    screen.blit(gen_text, (10, 10))

    # Alive count
    alive_text = font.render(f"Alive: {alive}/{total}", True, BLACK)
    screen.blit(alive_text, (10, 45))

    # Frame count
    frame_text = small_font.render(f"Frames: {frames}", True, BLACK)
    screen.blit(frame_text, (10, 80))

    # Instructions
    instructions = [
        "SPACE: Skip to next generation",
        "ESC: Quit"
    ]
    for i, instruction in enumerate(instructions):
        inst_text = small_font.render(instruction, True, BLACK)
        screen.blit(inst_text, (10, SCREEN_HEIGHT - 60 + i * 25))

def plot_progress(ga):
    """Plot fitness and score progress over generations."""
    stats = ga.get_statistics()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    # Plot fitness
    ax1.plot(stats['best_fitness'], label='Best Fitness', linewidth=2, color='green')
    ax1.plot(stats['avg_fitness'], label='Average Fitness', linewidth=2, color='blue')
    ax1.set_xlabel('Generation')
    ax1.set_ylabel('Fitness')
    ax1.set_title('Fitness Evolution')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot score
    ax2.plot(stats['best_score'], label='Best Score', linewidth=2, color='red')
    ax2.set_xlabel('Generation')
    ax2.set_ylabel('Score (Pipes Passed)')
    ax2.set_title('Score Evolution')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('flappy_bird_progress.png')
    print("Progress plot saved to flappy_bird_progress.png")

def main():
    """Main training loop."""
    # Initialize Pygame
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Flappy Bird AI - Genetic Algorithm")
    clock = pygame.time.Clock()

    print("=" * 60)
    print("Flappy Bird AI - Genetic Algorithm Training")
    print("=" * 60)
    print(f"Population Size: {POPULATION_SIZE}")
    print(f"Elite Size: {ELITE_SIZE}")
    print(f"Mutation Rate: {MUTATION_RATE}")
    print(f"Max Generations: {MAX_GENERATIONS}")
    print("=" * 60)
    print("\nWatch the birds learn to play Flappy Bird!")
    print("Yellow birds use neural networks to decide when to flap.")
    print("\n")

    # Initialize genetic algorithm
    ga = GeneticAlgorithm(
        population_size=POPULATION_SIZE,
        elite_size=ELITE_SIZE,
        mutation_rate=MUTATION_RATE
    )

    # Initialize pipe manager
    pipe_manager = PipeManager(SCREEN_WIDTH, SCREEN_HEIGHT)

    # Training loop
    try:
        for generation in range(MAX_GENERATIONS):
            # Run one generation
            fitness_scores = run_generation(ga, pipe_manager, screen, clock, show_all=True)

            # Check if user quit
            if fitness_scores is None:
                break

            # Evolve to next generation
            best_bird = ga.evolve(fitness_scores)

            # Save best bird every 10 generations
            if generation % 10 == 0 and generation > 0:
                ga.save_best(f'best_bird_gen_{generation}.pth')

            # Check if solved (score > 50)
            if best_bird.score >= 50:
                print(f"\n🎉 Bird mastered the game at generation {generation}!")
                print(f"Best score: {best_bird.score}")
                break

    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")

    # Cleanup
    print("\nTraining finished!")
    print(f"Total generations: {ga.generation}")
    if ga.best_score_history:
        print(f"Best score achieved: {max(ga.best_score_history)}")

    # Save final best bird
    ga.save_best('best_bird_final.pth')

    # Plot progress
    if ga.best_fitness_history:
        plot_progress(ga)

    pygame.quit()

if __name__ == "__main__":
    main()
