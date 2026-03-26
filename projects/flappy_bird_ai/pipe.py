import pygame
import random

class Pipe:
    """Pipe obstacle for Flappy Bird."""

    def __init__(self, x, screen_height=600):
        """
        Initialize pipe.

        Args:
            x: X position of pipe
            screen_height: Height of game screen
        """
        self.x = x
        self.width = 60
        self.screen_height = screen_height

        # Gap parameters
        self.gap_size = 150  # Size of gap bird must fly through
        self.gap_y = random.randint(150, screen_height - 150)  # Center of gap

        # Calculate pipe heights
        self.top_height = self.gap_y - self.gap_size // 2
        self.bottom_y = self.gap_y + self.gap_size // 2

        # Movement
        self.speed = 3

        # Scoring
        self.passed = False

    def update(self):
        """Move pipe to the left."""
        self.x -= self.speed

    def off_screen(self):
        """Check if pipe is off screen."""
        return self.x + self.width < 0

    def collides_with(self, bird):
        """
        Check if pipe collides with bird.

        Args:
            bird: Bird object

        Returns:
            True if collision detected
        """
        # Check if bird is horizontally aligned with pipe
        if bird.x + bird.radius < self.x or bird.x - bird.radius > self.x + self.width:
            return False

        # Check if bird hits top or bottom pipe
        if bird.y - bird.radius < self.top_height or bird.y + bird.radius > self.bottom_y:
            return True

        return False

    def render(self, screen):
        """Render pipe on screen."""
        # Pipe color (green)
        pipe_color = (50, 200, 50)
        border_color = (30, 150, 30)

        # Draw top pipe
        top_rect = pygame.Rect(self.x, 0, self.width, self.top_height)
        pygame.draw.rect(screen, pipe_color, top_rect)
        pygame.draw.rect(screen, border_color, top_rect, 3)

        # Draw pipe cap (top)
        cap_height = 20
        cap_width = self.width + 10
        top_cap_rect = pygame.Rect(self.x - 5, self.top_height - cap_height,
                                   cap_width, cap_height)
        pygame.draw.rect(screen, pipe_color, top_cap_rect)
        pygame.draw.rect(screen, border_color, top_cap_rect, 3)

        # Draw bottom pipe
        bottom_rect = pygame.Rect(self.x, self.bottom_y,
                                  self.width, self.screen_height - self.bottom_y)
        pygame.draw.rect(screen, pipe_color, bottom_rect)
        pygame.draw.rect(screen, border_color, bottom_rect, 3)

        # Draw pipe cap (bottom)
        bottom_cap_rect = pygame.Rect(self.x - 5, self.bottom_y,
                                      cap_width, cap_height)
        pygame.draw.rect(screen, pipe_color, bottom_cap_rect)
        pygame.draw.rect(screen, border_color, bottom_cap_rect, 3)

class PipeManager:
    """Manages multiple pipes."""

    def __init__(self, screen_width=800, screen_height=600):
        """
        Initialize pipe manager.

        Args:
            screen_width: Width of game screen
            screen_height: Height of game screen
        """
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.pipes = []
        self.spawn_distance = 250  # Distance between pipes
        self.next_spawn_x = screen_width

    def update(self):
        """Update all pipes."""
        # Move existing pipes
        for pipe in self.pipes:
            pipe.update()

        # Remove off-screen pipes
        self.pipes = [p for p in self.pipes if not p.off_screen()]

        # Spawn new pipe if needed
        if not self.pipes or self.pipes[-1].x < self.screen_width - self.spawn_distance:
            self.pipes.append(Pipe(self.screen_width, self.screen_height))

    def reset(self):
        """Reset pipes."""
        self.pipes = []
        self.pipes.append(Pipe(self.screen_width, self.screen_height))

    def check_collisions(self, bird):
        """
        Check if bird collides with any pipe.

        Args:
            bird: Bird object

        Returns:
            True if collision detected
        """
        for pipe in self.pipes:
            if pipe.collides_with(bird):
                return True
        return False

    def check_score(self, bird):
        """
        Check if bird passed a pipe (for scoring).

        Args:
            bird: Bird object

        Returns:
            True if bird passed a new pipe
        """
        for pipe in self.pipes:
            if not pipe.passed and pipe.x + pipe.width < bird.x:
                pipe.passed = True
                return True
        return False

    def render(self, screen):
        """Render all pipes."""
        for pipe in self.pipes:
            pipe.render(screen)

    def get_pipes(self):
        """Get list of all pipes."""
        return self.pipes
