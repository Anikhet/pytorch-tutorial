import pybullet as p
import pybullet_data
import numpy as np
import time

class BalancingEnvironment:
    """3D PyBullet environment for ball balancing robot."""

    def __init__(self, gui=True, time_step=1./240.):
        """
        Initialize the 3D environment.

        Args:
            gui: Whether to show visualization
            time_step: Physics simulation time step
        """
        # Connect to PyBullet
        if gui:
            self.client = p.connect(p.GUI)
        else:
            self.client = p.connect(p.DIRECT)  # No visualization

        # Set up physics
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(time_step)

        # Load ground plane
        self.plane_id = p.loadURDF("plane.urdf")

        # Platform parameters
        self.platform_size = [0.5, 0.5, 0.05]  # 50cm x 50cm x 5cm
        self.platform_mass = 1.0
        self.platform_height = 0.5  # 50cm above ground

        # Ball parameters
        self.ball_radius = 0.05  # 5cm radius
        self.ball_mass = 0.1  # 100g

        # Create platform and ball
        self.platform_id = None
        self.ball_id = None
        self._create_platform()
        self._create_ball()

        # Camera setup for better view
        p.resetDebugVisualizerCamera(
            cameraDistance=2.0,
            cameraYaw=45,
            cameraPitch=-30,
            cameraTargetPosition=[0, 0, 0.5]
        )

        # Control limits
        self.max_tilt_angle = np.radians(15)  # Max 15 degrees tilt

    def _create_platform(self):
        """Create the tilting platform."""
        # Visual shape (what we see)
        visual_shape = p.createVisualShape(
            shapeType=p.GEOM_BOX,
            halfExtents=[s/2 for s in self.platform_size],
            rgbaColor=[0.2, 0.6, 0.8, 1.0]  # Blue platform
        )

        # Collision shape (for physics)
        collision_shape = p.createCollisionShape(
            shapeType=p.GEOM_BOX,
            halfExtents=[s/2 for s in self.platform_size]
        )

        # Create the platform body
        self.platform_id = p.createMultiBody(
            baseMass=self.platform_mass,
            baseCollisionShapeIndex=collision_shape,
            baseVisualShapeIndex=visual_shape,
            basePosition=[0, 0, self.platform_height],
            baseOrientation=[0, 0, 0, 1]  # Quaternion (no rotation initially)
        )

        # Add friction to platform surface
        p.changeDynamics(
            self.platform_id,
            -1,  # Base link
            lateralFriction=0.8,
            spinningFriction=0.1,
            rollingFriction=0.01
        )

    def _create_ball(self):
        """Create the ball to balance."""
        # Visual shape
        visual_shape = p.createVisualShape(
            shapeType=p.GEOM_SPHERE,
            radius=self.ball_radius,
            rgbaColor=[1.0, 0.2, 0.2, 1.0]  # Red ball
        )

        # Collision shape
        collision_shape = p.createCollisionShape(
            shapeType=p.GEOM_SPHERE,
            radius=self.ball_radius
        )

        # Create ball slightly above platform center
        ball_start_height = self.platform_height + self.platform_size[2]/2 + self.ball_radius + 0.01

        self.ball_id = p.createMultiBody(
            baseMass=self.ball_mass,
            baseCollisionShapeIndex=collision_shape,
            baseVisualShapeIndex=visual_shape,
            basePosition=[0, 0, ball_start_height],
            baseOrientation=[0, 0, 0, 1]
        )

        # Ball physics properties
        p.changeDynamics(
            self.ball_id,
            -1,
            lateralFriction=0.8,
            spinningFriction=0.1,
            rollingFriction=0.01,
            restitution=0.5  # Bounciness
        )

    def reset(self):
        """Reset the environment to initial state."""
        # Reset platform to level
        p.resetBasePositionAndOrientation(
            self.platform_id,
            [0, 0, self.platform_height],
            [0, 0, 0, 1]
        )
        p.resetBaseVelocity(self.platform_id, [0, 0, 0], [0, 0, 0])

        # Reset ball to center with small random offset
        offset_x = np.random.uniform(-0.02, 0.02)  # ±2cm
        offset_y = np.random.uniform(-0.02, 0.02)
        ball_start_height = self.platform_height + self.platform_size[2]/2 + self.ball_radius + 0.01

        p.resetBasePositionAndOrientation(
            self.ball_id,
            [offset_x, offset_y, ball_start_height],
            [0, 0, 0, 1]
        )
        p.resetBaseVelocity(self.ball_id, [0, 0, 0], [0, 0, 0])

        return self.get_observation()

    def step(self, action):
        """
        Execute one step with the given action.

        Args:
            action: [tilt_x, tilt_y] in range [-1, 1]

        Returns:
            observation, reward, done, info
        """
        # Convert action to platform orientation
        tilt_x = action[0] * self.max_tilt_angle
        tilt_y = action[1] * self.max_tilt_angle

        # Convert Euler angles to quaternion
        orientation = p.getQuaternionFromEuler([tilt_x, tilt_y, 0])

        # Apply orientation to platform
        pos, _ = p.getBasePositionAndOrientation(self.platform_id)
        p.resetBasePositionAndOrientation(self.platform_id, pos, orientation)

        # Step physics simulation
        p.stepSimulation()

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
        # Ball position and velocity
        ball_pos, ball_orn = p.getBasePositionAndOrientation(self.ball_id)
        ball_vel, ball_ang_vel = p.getBaseVelocity(self.ball_id)

        # Platform position and orientation
        platform_pos, platform_orn = p.getBasePositionAndOrientation(self.platform_id)
        platform_euler = p.getEulerFromQuaternion(platform_orn)

        # Ball position relative to platform center
        relative_x = ball_pos[0] - platform_pos[0]
        relative_y = ball_pos[1] - platform_pos[1]
        relative_z = ball_pos[2] - (platform_pos[2] + self.platform_size[2]/2)

        # Observation vector (8 dimensions)
        obs = np.array([
            relative_x,           # Ball X position relative to platform
            relative_y,           # Ball Y position
            relative_z,           # Ball Z position
            ball_vel[0],          # Ball X velocity
            ball_vel[1],          # Ball Y velocity
            ball_vel[2],          # Ball Z velocity
            platform_euler[0],    # Platform tilt X
            platform_euler[1],    # Platform tilt Y
        ], dtype=np.float32)

        return obs

    def _compute_reward(self):
        """
        Compute reward and check termination.

        Returns:
            reward, done, info
        """
        obs = self.get_observation()

        ball_x, ball_y, ball_z = obs[0], obs[1], obs[2]
        ball_vx, ball_vy, ball_vz = obs[3], obs[4], obs[5]

        # Distance from center of platform
        distance_from_center = np.sqrt(ball_x**2 + ball_y**2)

        # Ball velocity magnitude
        velocity_magnitude = np.sqrt(ball_vx**2 + ball_vy**2)

        # Check if ball fell off platform
        platform_radius = min(self.platform_size[0], self.platform_size[1]) / 2
        done = False

        if distance_from_center > platform_radius:
            done = True
            reward = -10.0  # Big penalty for falling off
        elif ball_z < -0.1:  # Ball fell below platform significantly
            done = True
            reward = -10.0
        else:
            # Reward for keeping ball near center
            reward = 1.0 - distance_from_center  # Max reward when at center

            # Bonus for low velocity (stability)
            reward += (1.0 - min(velocity_magnitude, 1.0)) * 0.5

            # Small time bonus for survival
            reward += 0.1

        info = {
            'distance_from_center': distance_from_center,
            'velocity': velocity_magnitude,
            'ball_position': [ball_x, ball_y, ball_z]
        }

        return reward, done, info

    def render(self):
        """Render is automatic in GUI mode."""
        pass

    def close(self):
        """Clean up PyBullet."""
        p.disconnect(self.client)

    def get_state_info(self):
        """Get human-readable state information."""
        obs = self.get_observation()
        ball_x, ball_y = obs[0], obs[1]
        distance = np.sqrt(ball_x**2 + ball_y**2)

        return {
            'ball_distance_from_center': distance,
            'ball_position': (ball_x, ball_y),
            'platform_tilt': (np.degrees(obs[6]), np.degrees(obs[7]))
        }
