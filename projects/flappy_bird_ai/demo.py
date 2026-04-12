import pygame
import sys
import os
from bird import Bird
from pipe import PipeManager
from genetic_algorithm import GeneticAlgorithm

# Configuration
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
FPS = 60

# Colors
SKY_BLUE = (135, 206, 250)
GROUND_COLOR = (222, 216, 149)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (0, 200, 0)

def demo_bird(bird_path):
    """
    Demo a trained bird.

    Args:
        bird_path: Path to saved bird model
    """
    # Initialize Pygame
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Flappy Bird AI - Demo")
    clock = pygame.time.Clock()

    # Load trained bird
    if not os.path.exists(bird_path):
        print(f"Error: Model file '{bird_path}' not found!")
        print("Train a bird first using: python main.py")
        return

    ga = GeneticAlgorithm()
    bird = ga.load_best(bird_path)

    # Initialize environment
    pipe_manager = PipeManager(SCREEN_WIDTH, SCREEN_HEIGHT)
    pipe_manager.reset()

    # Reset bird
    bird.y = 300
    bird.velocity = 0
    bird.alive = True
    bird.score = 0
    bird.frames_alive = 0

    print("=" * 60)
    print("Flappy Bird AI - Demo Mode")
    print("=" * 60)
    print(f"Loaded model: {bird_path}")
    print("Watch the trained bird play!")
    print("Press SPACE to restart, ESC to quit")
    print("=" * 60)

    # Game loop
    running = True
    paused = False

    while running:
        clock.tick(FPS)

        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    # Restart
                    bird.y = 300
                    bird.velocity = 0
                    bird.alive = True
                    bird.score = 0
                    bird.frames_alive = 0
                    pipe_manager.reset()
                elif event.key == pygame.K_p:
                    # Pause/unpause
                    paused = not paused

        # Update game (if not paused and bird alive)
        if not paused and bird.alive:
            # Update pipes
            pipe_manager.update()

            # Bird thinks and decides to jump
            if bird.think(pipe_manager.get_pipes()):
                bird.jump()

            # Update bird physics
            bird.update()

            # Check collisions
            if pipe_manager.check_collisions(bird):
                bird.alive = False
                print(f"\nBird died! Score: {bird.score}, Frames alive: {bird.frames_alive}")

            # Check scoring
            if pipe_manager.check_score(bird):
                bird.score += 1

        # Render
        screen.fill(SKY_BLUE)

        # Draw pipes
        pipe_manager.render(screen)

        # Draw ground
        ground_height = 50
        pygame.draw.rect(screen, GROUND_COLOR,
                        (0, SCREEN_HEIGHT - ground_height, SCREEN_WIDTH, ground_height))

        # Draw bird
        bird.render(screen)

        # Draw UI
        font = pygame.font.Font(None, 48)
        small_font = pygame.font.Font(None, 28)
        tiny_font = pygame.font.Font(None, 20)

        # Score (large and prominent)
        score_text = font.render(f"Score: {bird.score}", True, BLACK)
        screen.blit(score_text, (10, 10))

        # Status
        if bird.alive:
            status_text = small_font.render("ALIVE", True, GREEN)
        else:
            status_text = small_font.render("DEAD - Press SPACE to restart", True, (200, 0, 0))
        screen.blit(status_text, (10, 65))

        # Frames alive
        frames_text = tiny_font.render(f"Frames: {bird.frames_alive}", True, BLACK)
        screen.blit(frames_text, (10, 100))

        # Instructions
        instructions = [
            "SPACE: Restart",
            "P: Pause/Unpause",
            "ESC: Quit"
        ]
        for i, instruction in enumerate(instructions):
            inst_text = tiny_font.render(instruction, True, BLACK)
            screen.blit(inst_text, (10, SCREEN_HEIGHT - 80 + i * 22))

        # Paused indicator
        if paused:
            pause_text = font.render("PAUSED", True, (255, 0, 0))
            text_rect = pause_text.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2))
            # Draw shadow
            shadow_rect = text_rect.copy()
            shadow_rect.x += 3
            shadow_rect.y += 3
            pause_shadow = font.render("PAUSED", True, BLACK)
            screen.blit(pause_shadow, shadow_rect)
            screen.blit(pause_text, text_rect)

        pygame.display.flip()

    pygame.quit()

def main():
    """Main entry point."""
    # Check for command line argument
    if len(sys.argv) > 1:
        bird_path = sys.argv[1]
    else:
        # Default to final best bird
        bird_path = "best_bird_final.pth"

    demo_bird(bird_path)

if __name__ == "__main__":
    main()
