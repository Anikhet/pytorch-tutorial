import pygame
import numpy as np
import math

class BalancingEnvironment:
    """Ball balancing environment with Pygame isometric 3D visualization."""

    def __init__(self, gui=True, dt=1./60.):
        """
        Initialize the environment.

        Args:
            gui: Whether to show visualization
            dt: Time step for physics simulation
        """
        self.dt = dt
        self.gui = gui

        # Platform parameters
        self.platform_size = 0.5  # 50cm square
        self.max_tilt_angle = np.radians(15)  # Max 15 degrees tilt

        # Ball parameters
        self.ball_radius = 0.05  # 5cm
        self.ball_mass = 0.1  # 100g
        self.gravity = 9.81

        # State variables
        self.ball_pos = np.zeros(2)  # [x, y] position on platform
        self.ball_vel = np.zeros(2)  # [vx, vy] velocity
        self.platform_tilt = np.zeros(2)  # [tilt_x, tilt_y] in radians

        # Friction coefficient
        self.friction = 0.1

        # Pygame setup
        if self.gui:
            pygame.init()
            self.screen_width = 800
            self.screen_height = 600
            self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
            pygame.display.set_caption("3D Ball Balancer - Genetic Algorithm")
            self.clock = pygame.time.Clock()
            self.font = pygame.font.Font(None, 24)

            # Isometric projection scale
            self.scale = 400  # pixels per meter

    def reset(self):
        """Reset environment to initial state."""
        # Ball starts near center with small random offset
        self.ball_pos = np.random.uniform(-0.02, 0.02, size=2)
        self.ball_vel = np.zeros(2)
        self.platform_tilt = np.zeros(2)

        return self.get_observation()

    def step(self, action):
        """
        Execute one step with the given action.

        Args:
            action: [tilt_x, tilt_y] in range [-1, 1]

        Returns:
            observation, reward, done, info
        """
        # Apply platform tilt
        self.platform_tilt = action * self.max_tilt_angle

        # Calculate acceleration due to gravity and tilt
        accel = np.array([
            self.gravity * np.sin(self.platform_tilt[0]),
            self.gravity * np.sin(self.platform_tilt[1])
        ])

        # Apply friction
        friction_force = -self.friction * self.ball_vel

        # Update velocity and position
        self.ball_vel += (accel + friction_force) * self.dt
        self.ball_pos += self.ball_vel * self.dt

        # Get new observation
        observation = self.get_observation()

        # Calculate reward and check if done
        reward, done, info = self._compute_reward()

        return observation, reward, done, info

    def get_observation(self):
        """Get current state observation."""
        ball_z = self.ball_radius

        obs = np.array([
            self.ball_pos[0],
            self.ball_pos[1],
            ball_z,
            self.ball_vel[0],
            self.ball_vel[1],
            0.0,
            self.platform_tilt[0],
            self.platform_tilt[1],
        ], dtype=np.float32)

        return obs

    def _compute_reward(self):
        """Compute reward and check termination."""
        distance = np.linalg.norm(self.ball_pos)
        velocity = np.linalg.norm(self.ball_vel)

        platform_radius = self.platform_size / 2
        done = False

        if distance > platform_radius:
            done = True
            reward = -10.0
        else:
            reward = 1.0 - (distance / platform_radius)
            reward += (1.0 - min(velocity, 1.0)) * 0.5
            reward += 0.1

        info = {
            'distance_from_center': distance,
            'velocity': velocity,
            'ball_position': self.ball_pos.tolist()
        }

        return reward, done, info

    def render(self):
        """Render the environment using isometric projection."""
        if not self.gui:
            return

        # Handle pygame events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return

        # Clear screen
        self.screen.fill((20, 20, 30))  # Dark background

        # Draw from isometric perspective
        self._draw_isometric()

        # Draw UI
        self._draw_ui()

        pygame.display.flip()
        self.clock.tick(60)

    def _draw_isometric(self):
        """Draw platform and ball in isometric 3D."""
        # Center point on screen
        center_x = self.screen_width // 2
        center_y = self.screen_height // 2 + 50

        # Isometric projection function
        def project(x, y, z):
            """Project 3D point to 2D isometric coordinates."""
            # Isometric projection
            screen_x = (x - y) * self.scale * 0.866  # cos(30°)
            screen_y = (x + y) * self.scale * 0.5 - z * self.scale
            return int(center_x + screen_x), int(center_y + screen_y)

        # Draw platform (tilted)
        platform_half = self.platform_size / 2

        # Calculate platform corners considering tilt
        corners_3d = []
        for dx in [-platform_half, platform_half]:
            for dy in [-platform_half, platform_half]:
                # Height change due to tilt
                z_offset = dx * math.tan(self.platform_tilt[0]) + dy * math.tan(self.platform_tilt[1])
                corners_3d.append((dx, dy, z_offset))

        # Draw platform surface
        corners_2d = [project(x, y, z) for x, y, z in corners_3d]

        # Draw platform as gradient quad
        platform_color = (60, 120, 200)  # Blue
        pygame.draw.polygon(self.screen, platform_color, [
            corners_2d[0], corners_2d[1], corners_2d[3], corners_2d[2]
        ])

        # Draw platform grid
        grid_color = (80, 140, 220)
        for i in range(-2, 3):
            offset = i * 0.1
            p1 = project(-platform_half, offset, 0)
            p2 = project(platform_half, offset, 0)
            pygame.draw.line(self.screen, grid_color, p1, p2, 1)

            p1 = project(offset, -platform_half, 0)
            p2 = project(offset, platform_half, 0)
            pygame.draw.line(self.screen, grid_color, p1, p2, 1)

        # Draw platform outline
        pygame.draw.lines(self.screen, (100, 160, 240), True, corners_2d, 3)

        # Draw ball
        ball_x, ball_y = self.ball_pos
        ball_z = self.ball_radius

        # Account for platform tilt when positioning ball
        platform_z = ball_x * math.tan(self.platform_tilt[0]) + ball_y * math.tan(self.platform_tilt[1])
        ball_screen_x, ball_screen_y = project(ball_x, ball_y, ball_z + platform_z)

        # Draw ball shadow
        shadow_x, shadow_y = project(ball_x, ball_y, platform_z)
        shadow_radius = int(self.ball_radius * self.scale * 0.7)
        pygame.draw.circle(self.screen, (0, 0, 0, 100), (shadow_x, shadow_y), shadow_radius)

        # Draw ball with 3D effect
        ball_radius_px = int(self.ball_radius * self.scale)
        # Outer dark edge
        pygame.draw.circle(self.screen, (150, 30, 30), (ball_screen_x, ball_screen_y), ball_radius_px)
        # Highlight
        highlight_offset = ball_radius_px // 3
        pygame.draw.circle(self.screen, (255, 80, 80),
                          (ball_screen_x - highlight_offset, ball_screen_y - highlight_offset),
                          ball_radius_px // 2)

        # Draw platform support (pillar)
        pillar_bottom = project(0, 0, -0.3)
        pillar_top = project(0, 0, 0)
        pygame.draw.line(self.screen, (100, 100, 100), pillar_bottom, pillar_top, 8)

    def _draw_ui(self):
        """Draw UI information."""
        # Distance from center
        distance = np.linalg.norm(self.ball_pos)
        distance_text = self.font.render(f"Distance: {distance:.3f}m", True, (255, 255, 255))
        self.screen.blit(distance_text, (10, 10))

        # Platform tilt
        tilt_x_deg = np.degrees(self.platform_tilt[0])
        tilt_y_deg = np.degrees(self.platform_tilt[1])
        tilt_text = self.font.render(f"Tilt: X={tilt_x_deg:.1f}° Y={tilt_y_deg:.1f}°",
                                     True, (255, 255, 255))
        self.screen.blit(tilt_text, (10, 35))

        # Velocity
        velocity = np.linalg.norm(self.ball_vel)
        vel_text = self.font.render(f"Velocity: {velocity:.2f}m/s", True, (255, 255, 255))
        self.screen.blit(vel_text, (10, 60))

        # Status
        if distance > self.platform_size / 2:
            status_text = self.font.render("BALL FELL OFF!", True, (255, 50, 50))
            self.screen.blit(status_text, (10, 90))
        else:
            status_text = self.font.render("Balancing...", True, (50, 255, 50))
            self.screen.blit(status_text, (10, 90))

    def close(self):
        """Clean up."""
        if self.gui:
            pygame.quit()

    def get_state_info(self):
        """Get human-readable state information."""
        distance = np.linalg.norm(self.ball_pos)

        return {
            'ball_distance_from_center': distance,
            'ball_position': self.ball_pos.tolist(),
            'platform_tilt': (np.degrees(self.platform_tilt[0]),
                            np.degrees(self.platform_tilt[1]))
        }
