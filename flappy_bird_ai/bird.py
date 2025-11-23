import pygame
import numpy as np
from neural_network import BirdNeuralNetwork

class Bird:
    """Flappy Bird with neural network control."""

    def __init__(self, x=100, y=300, brain=None):
        """
        Initialize bird.

        Args:
            x: Starting X position
            y: Starting Y position
            brain: Neural network (creates new if None)
        """
        self.x = x
        self.y = y
        self.velocity = 0
        self.alive = True

        # Physics
        self.gravity = 0.5
        self.jump_strength = -10
        self.max_velocity = 10

        # Size
        self.radius = 15

        # Stats
        self.score = 0
        self.fitness = 0
        self.frames_alive = 0

        # Neural network brain
        self.brain = brain if brain else BirdNeuralNetwork()

    def jump(self):
        """Make bird jump (flap)."""
        if self.alive:
            self.velocity = self.jump_strength

    def update(self):
        """Update bird physics."""
        if not self.alive:
            return

        # Apply gravity
        self.velocity += self.gravity
        self.velocity = min(self.velocity, self.max_velocity)

        # Update position
        self.y += self.velocity

        # Update stats
        self.frames_alive += 1

        # Check if bird went off screen
        if self.y < 0 or self.y > 600:
            self.alive = False

    def think(self, pipes):
        """
        Use neural network to decide whether to jump.

        Args:
            pipes: List of pipe objects

        Returns:
            decision: True if should jump
        """
        if not self.alive:
            return False

        # Find the closest pipe ahead
        closest_pipe = None
        min_distance = float('inf')

        for pipe in pipes:
            # Only consider pipes that are ahead of the bird
            if pipe.x + pipe.width > self.x:
                distance = pipe.x - self.x
                if distance < min_distance:
                    min_distance = distance
                    closest_pipe = pipe

        # If no pipe found, use dummy values
        if closest_pipe is None:
            inputs = np.array([self.y / 600, 0.5, 0.5, self.velocity / 10])
        else:
            # Neural network inputs:
            # 1. Bird's Y position (normalized)
            # 2. Distance to next pipe (normalized)
            # 3. Height of top pipe opening (normalized)
            # 4. Bird's velocity (normalized)
            inputs = np.array([
                self.y / 600,  # Bird Y position (0-1)
                min_distance / 400,  # Horizontal distance to pipe
                closest_pipe.gap_y / 600,  # Y position of gap center
                (self.velocity + 10) / 20  # Velocity normalized to 0-1
            ])

        # Get decision from neural network
        output = self.brain.predict(inputs)

        # If output > 0.5, jump!
        return output[0] > 0.5

    def calculate_fitness(self):
        """Calculate fitness score for genetic algorithm."""
        # Fitness = how long survived + score bonus
        self.fitness = self.frames_alive + self.score * 100

    def render(self, screen):
        """Render bird on screen."""
        if not self.alive:
            return

        # Bird color (yellow)
        color = (255, 200, 0)

        # Draw bird as circle
        pygame.draw.circle(screen, color, (int(self.x), int(self.y)), self.radius)

        # Draw eye
        eye_x = int(self.x + 5)
        eye_y = int(self.y - 3)
        pygame.draw.circle(screen, (0, 0, 0), (eye_x, eye_y), 3)

        # Draw beak (triangle)
        beak_points = [
            (self.x + self.radius, self.y),
            (self.x + self.radius + 10, self.y - 3),
            (self.x + self.radius + 10, self.y + 3)
        ]
        pygame.draw.polygon(screen, (255, 140, 0), beak_points)

    def copy(self):
        """Create a copy of this bird with same brain."""
        new_bird = Bird(self.x, self.y, self.brain.copy())
        return new_bird
