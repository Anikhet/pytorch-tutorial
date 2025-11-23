import numpy as np

class BalancingEnvironment:
    """Simplified ball balancing environment without PyBullet."""

    def __init__(self, dt=1./240.):
        """
        Initialize the environment.

        Args:
            dt: Time step for physics simulation
        """
        self.dt = dt

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
        # When platform tilts, ball rolls downhill
        accel = np.array([
            self.gravity * np.sin(self.platform_tilt[0]),  # X acceleration
            self.gravity * np.sin(self.platform_tilt[1])   # Y acceleration
        ])

        # Apply friction (opposes velocity)
        friction_force = -self.friction * self.ball_vel

        # Update velocity and position (simple Euler integration)
        self.ball_vel += (accel + friction_force) * self.dt
        self.ball_pos += self.ball_vel * self.dt

        # Get new observation
        observation = self.get_observation()

        # Calculate reward and check if done
        reward, done, info = self._compute_reward()

        return observation, reward, done, info

    def get_observation(self):
        """
        Get current state observation.

        Returns:
            observation: [ball_x, ball_y, ball_z, ball_vx, ball_vy, ball_vz,
                         platform_tilt_x, platform_tilt_y]
        """
        # Ball Z position is just radius (always on platform when alive)
        ball_z = self.ball_radius

        obs = np.array([
            self.ball_pos[0],      # Ball X position
            self.ball_pos[1],      # Ball Y position
            ball_z,                # Ball Z position (height)
            self.ball_vel[0],      # Ball X velocity
            self.ball_vel[1],      # Ball Y velocity
            0.0,                   # Ball Z velocity (no jumping)
            self.platform_tilt[0], # Platform tilt X
            self.platform_tilt[1], # Platform tilt Y
        ], dtype=np.float32)

        return obs

    def _compute_reward(self):
        """
        Compute reward and check termination.

        Returns:
            reward, done, info
        """
        # Distance from center
        distance = np.linalg.norm(self.ball_pos)

        # Velocity magnitude
        velocity = np.linalg.norm(self.ball_vel)

        # Check if ball fell off platform
        platform_radius = self.platform_size / 2
        done = False

        if distance > platform_radius:
            done = True
            reward = -10.0  # Big penalty for falling off
        else:
            # Reward for keeping ball near center
            reward = 1.0 - (distance / platform_radius)

            # Bonus for low velocity (stability)
            reward += (1.0 - min(velocity, 1.0)) * 0.5

            # Small time bonus for survival
            reward += 0.1

        info = {
            'distance_from_center': distance,
            'velocity': velocity,
            'ball_position': self.ball_pos.tolist()
        }

        return reward, done, info

    def close(self):
        """Clean up (nothing to do in headless mode)."""
        pass

    def get_state_info(self):
        """Get human-readable state information."""
        distance = np.linalg.norm(self.ball_pos)

        return {
            'ball_distance_from_center': distance,
            'ball_position': self.ball_pos.tolist(),
            'platform_tilt': (np.degrees(self.platform_tilt[0]),
                            np.degrees(self.platform_tilt[1]))
        }
