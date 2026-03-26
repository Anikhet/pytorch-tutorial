import pygame
import math
import numpy as np

class Car:
    """Car with physics simulation and distance sensors."""

    def __init__(self, x, y, angle, track, color=None):
        self.x = x
        self.y = y
        self.angle = angle  # In degrees
        self.track = track

        # Physics
        self.velocity = 0
        self.max_velocity = 8  # Increased from 8
        self.acceleration = 0.6  # Increased from 0.3
        self.friction = 0.15  # Slightly increased
        self.turn_speed = 6  # Slightly faster turning

        # Car dimensions
        self.width = 20
        self.height = 35

        # Sensors (5 raycasts in different directions)
        self.num_sensors = 5
        self.sensor_angles = [-60, -30, 0, 30, 60]  # Relative to car angle
        self.sensor_length = 200
        self.sensor_readings = [0] * self.num_sensors

        # State
        self.alive = True
        self.distance_traveled = 0
        self.fitness = 0

        # Color for rendering
        self.color = color if color else (255, 0, 0)

        # For rendering
        self.last_x = x
        self.last_y = y

    def update(self, action):
        """
        Update car physics based on action.
        action: [acceleration, steering] where each is in [-1, 1]
        """
        if not self.alive:
            return

        # Apply actions
        acceleration_input = action[0]  # -1 to 1
        steering_input = action[1]  # -1 to 1

        # Update velocity
        self.velocity += acceleration_input * self.acceleration
        self.velocity -= self.friction  # Apply friction
        self.velocity = max(0, min(self.velocity, self.max_velocity))

        # Update angle based on velocity (can't turn if not moving)
        if self.velocity > 0.5:
            self.angle += steering_input * self.turn_speed

        # Update position
        rad_angle = math.radians(self.angle)
        dx = math.cos(rad_angle) * self.velocity
        dy = math.sin(rad_angle) * self.velocity

        # Store distance traveled for fitness
        distance = math.sqrt(dx**2 + dy**2)
        self.distance_traveled += distance

        self.x += dx
        self.y += dy

        # Update sensors
        self._update_sensors()

        # Check collision
        if not self.track.is_on_track(self.x, self.y):
            self.alive = False

        # Update fitness
        self._update_fitness()

    def _update_sensors(self):
        """Update raycast sensors to detect track boundaries."""
        for i, sensor_angle in enumerate(self.sensor_angles):
            angle = math.radians(self.angle + sensor_angle)
            self.sensor_readings[i] = self._cast_ray(angle)

    def _cast_ray(self, angle):
        """
        Cast a ray in the given direction and return distance to boundary.
        Returns normalized distance (0 to 1, where 1 is max sensor length).
        """
        step_size = 5
        for distance in range(0, self.sensor_length, step_size):
            x = self.x + math.cos(angle) * distance
            y = self.y + math.sin(angle) * distance

            # Check if point is off track
            if not self.track.is_on_track(x, y):
                return distance / self.sensor_length

        return 1.0  # Max distance

    def _update_fitness(self):
        """Calculate fitness score for genetic algorithm."""
        # Fitness is primarily based on distance traveled
        self.fitness = self.distance_traveled

        # Bonus for staying alive longer
        if self.alive:
            self.fitness += 10

    def get_sensor_data(self):
        """Return sensor readings as numpy array."""
        return np.array(self.sensor_readings, dtype=np.float32)

    def render(self, screen, show_sensors=False):
        """Render the car and optionally its sensors."""
        if not self.alive:
            return

        # Draw car as a rotated rectangle
        rad_angle = math.radians(self.angle)

        # Calculate corners of the car
        corners = []
        for dx, dy in [(-self.width/2, -self.height/2),
                       (self.width/2, -self.height/2),
                       (self.width/2, self.height/2),
                       (-self.width/2, self.height/2)]:
            # Rotate around car center
            rotated_x = dx * math.cos(rad_angle) - dy * math.sin(rad_angle)
            rotated_y = dx * math.sin(rad_angle) + dy * math.cos(rad_angle)
            corners.append((int(self.x + rotated_x), int(self.y + rotated_y)))

        pygame.draw.polygon(screen, self.color, corners)

        # Draw direction indicator (white dot at front)
        front_x = self.x + math.cos(rad_angle) * self.height / 2
        front_y = self.y + math.sin(rad_angle) * self.height / 2
        pygame.draw.circle(screen, (255, 255, 255), (int(front_x), int(front_y)), 3)

        # Optionally draw sensors
        if show_sensors:
            self._render_sensors(screen)

    def _render_sensors(self, screen):
        """Render sensor rays."""
        for i, sensor_angle in enumerate(self.sensor_angles):
            angle = math.radians(self.angle + sensor_angle)
            distance = self.sensor_readings[i] * self.sensor_length

            end_x = self.x + math.cos(angle) * distance
            end_y = self.y + math.sin(angle) * distance

            # Color based on distance (green = far, red = close)
            color_value = int(255 * (1 - self.sensor_readings[i]))
            color = (color_value, 255 - color_value, 0)

            pygame.draw.line(screen, color, (int(self.x), int(self.y)),
                           (int(end_x), int(end_y)), 1)
            pygame.draw.circle(screen, color, (int(end_x), int(end_y)), 3)
