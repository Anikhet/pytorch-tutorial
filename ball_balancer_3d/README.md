# 3D Ball Balancing Robot with Genetic Algorithms

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)
![License](https://img.shields.io/badge/License-MIT-green)
![GPU](https://img.shields.io/badge/GPU-Optional-yellow)

A **3D physics simulation** where a robot learns to balance a ball on a tilting platform using neural networks and genetic algorithms. Built with **PyBullet** for realistic 3D physics!

## Learning Objectives

By completing this tutorial, you will learn:

- **Genetic Algorithms**: Implement selection, crossover, and mutation for neuroevolution
- **Physics Simulation**: Use PyBullet for realistic 3D physics with gravity and friction
- **Continuous Control**: Train neural networks for continuous action spaces (tilt angles)
- **Fitness Function Design**: Create reward signals that guide evolution toward desired behavior
- **State Representation**: Design input features from sensor readings (position, velocity)
- **Real-time Visualization**: Render learning progress with 3D graphics

## What Is This?

Watch in **real-time 3D** as a robot learns to balance a ball by tilting a platform! The robot uses:
- **8 sensors** (ball position, velocity, platform tilt)
- **Neural network brain** (makes decisions)
- **Genetic evolution** (learns over generations)

## Features

- ✨ **Full 3D visualization** with PyBullet
- 🎮 **Realistic physics** (gravity, friction, collisions)
- 🧠 **Neural network** control (8 inputs → 16 hidden → 16 hidden → 2 outputs)
- 🧬 **Genetic algorithm** evolution
- 📊 **Live training visualization** - watch the robot improve!
- 💾 **Save/load** trained models

## Quick Start

### Installation

```bash
cd ball_balancer_3d

# Using uv (recommended)
uv sync --no-install-project

# Or using pip
pip install pybullet numpy torch matplotlib
```

### Training

```bash
python main.py
```

**A 3D window will open showing:**
- Blue platform that tilts
- Red ball that needs to stay balanced
- Real-time physics simulation

The robot learns over ~100 generations to keep the ball centered!

### Demo Mode

Watch your trained robot:

```bash
python demo.py
```

## How It Works

### The Challenge

Balance a ball on a platform by controlling tilt angles:

```
           🔴 ← Ball (must stay centered!)
        ╱━━━━━━━╲
       ╱         ╲  ← Platform (tilts in X and Y)
      ╱___________╲
         │   │
         │   │  ← Robot controls (tilt_x, tilt_y)
    ═════════════════
```

### Sensors (8 inputs)

1. **Ball X position** relative to platform center
2. **Ball Y position** relative to platform center
3. **Ball Z position** (height above platform)
4. **Ball X velocity**
5. **Ball Y velocity**
6. **Ball Z velocity**
7. **Platform tilt X** (current angle)
8. **Platform tilt Y** (current angle)

### Actions (2 outputs)

1. **Tilt X** (-1 to +1, max ±15°)
2. **Tilt Y** (-1 to +1, max ±15°)

### Neural Network

```
Sensors (8) → Hidden (16) → Hidden (16) → Actions (2)
             Tanh           Tanh           Tanh
```

**Total parameters:** ~400 weights evolved through genetic algorithm!

### Reward Function

```python
if ball_fell_off:
    reward = -10  # Big penalty

else:
    # Reward for keeping ball near center
    reward = 1.0 - distance_from_center

    # Bonus for stability (low velocity)
    reward += (1.0 - velocity) * 0.5

    # Time bonus
    reward += 0.1
```

### Learning Process

**Generation 0:** Random tilting → Ball falls off immediately

**Generation 10-20:** Starting to keep ball on platform briefly

**Generation 30-50:** Can balance for several seconds!

**Generation 100+:** Expert balancing, keeps ball centered perfectly!

## Project Structure

```
ball_balancer_3d/
├── environment.py          # PyBullet 3D physics environment
├── neural_network.py       # Neural network (brain)
├── genetic_algorithm.py    # Evolution logic
├── main.py                 # Training loop
├── demo.py                 # Visualize trained robot
├── pyproject.toml          # Dependencies
└── README.md              # This file
```

## Configuration

Edit parameters in `main.py`:

```python
POPULATION_SIZE = 10       # Robots per generation
ELITE_SIZE = 2             # Best robots kept unchanged
MUTATION_RATE = 0.1        # Probability of weight mutation
MAX_EPISODE_STEPS = 500    # Steps per evaluation (~2 seconds)
GENERATIONS = 100          # Training generations
```

## Understanding the 3D View

### Camera Controls (PyBullet)
- **Left click + drag:** Rotate view
- **Right click + drag:** Pan camera
- **Scroll wheel:** Zoom in/out
- **Ctrl + Left click:** Rotate around object

### What You'll See

**Blue Platform:**
- Tilts to balance the ball
- Can tilt ±15° in X and Y directions
- 50cm × 50cm square

**Red Ball:**
- 5cm radius
- Affected by gravity and friction
- Must stay on platform

**Ground Plane:**
- Gray checkerboard
- Shows when ball has fallen

## Comparing to 2D Racing Car Project

| Feature | Racing Car (2D) | Ball Balancer (3D) |
|---------|----------------|-------------------|
| Visualization | Pygame (2D top-down) | PyBullet (3D realistic) |
| Physics | Simple custom | Full 3D rigid body |
| Task | Navigate track | Balance ball |
| Sensors | 5 raycasts | 8 state variables |
| Actions | Accel + Steering | Tilt X + Tilt Y |
| Difficulty | Medium | Hard (requires precision) |

## Advanced Topics

### Why PyBullet?

**PyBullet** is perfect for this because:
- Industry-standard physics engine
- Used in robotics research
- Realistic friction, collisions, gravity
- Easy to create custom robots
- Free and open-source

### Real-World Applications

This same approach is used for:
- **Robot control** (Boston Dynamics, humanoids)
- **Drone stabilization** (quadcopters)
- **Inverted pendulum** (classic control theory)
- **Balancing robots** (Segway-style)

### Extending the Project

**1. Add obstacles:**
```python
# environment.py
obstacle = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.1, 0.1, 0.1])
p.createMultiBody(baseMass=0, basePosition=[0.2, 0.2, 0.6], ...)
```

**2. Multiple balls:**
```python
self.balls = [self._create_ball() for _ in range(3)]
```

**3. Moving platform:**
```python
# Make platform move in a circle
angle = time * 0.5
x = 0.2 * cos(angle)
y = 0.2 * sin(angle)
p.resetBasePosition(self.platform_id, [x, y, self.platform_height])
```

**4. Different shapes:**
```python
# Try balancing a cube instead
shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.05, 0.05, 0.05])
```

## Troubleshooting

**PyBullet window doesn't open:**
- Make sure you have a display/GUI available
- Try updating PyBullet: `pip install --upgrade pybullet`
- On Linux, may need: `sudo apt-get install python3-opengl`

**Ball falls through platform:**
- This is normal early in training!
- Random weights → random tilting → ball falls
- Keep training, robot will learn

**Training too slow:**
- Reduce `POPULATION_SIZE`
- Reduce `MAX_EPISODE_STEPS`
- Set `gui=False` in `BalancingEnvironment()` (no visualization, much faster!)

**Robot learns then forgets:**
- Increase `ELITE_SIZE` to 3-4
- Reduce `MUTATION_RATE` to 0.05
- Increase `POPULATION_SIZE` for stability

## Performance Tips

### Fast Training (No GUI)

```python
# main.py, modify:
env = BalancingEnvironment(gui=False)  # 10x faster!

# Only visualize best network at end
if generation == GENERATIONS - 1:
    env_gui = BalancingEnvironment(gui=True)
    evaluate_network(env_gui, best_network, render=True)
```

### Parallel Training

```python
from multiprocessing import Pool

def evaluate_wrapper(network):
    env = BalancingEnvironment(gui=False)
    return evaluate_network(env, network)

# Evaluate population in parallel
with Pool(4) as pool:
    fitness_scores = pool.map(evaluate_wrapper, population)
```

## Learning Resources

**PyBullet:**
- Official Quickstart: https://docs.google.com/document/d/10sXEhzFRSnvFcl3XxNGhnD4N2SedqwdAvK3dsihxVUA/
- Examples: https://github.com/bulletphysics/bullet3/tree/master/examples/pybullet

**Control Theory:**
- Inverted Pendulum problem
- PID controllers (classical approach)
- LQR (Linear Quadratic Regulator)

**Related RL Algorithms:**
- PPO (Proximal Policy Optimization)
- SAC (Soft Actor-Critic)
- TD3 (Twin Delayed DDPG)

## Comparison to Other Approaches

### vs. PID Controller (Classical)

**PID:**
```python
error = ball_position - target_position
tilt = Kp * error + Ki * integral + Kd * derivative
```
- Must hand-tune Kp, Ki, Kd
- Works well for this simple task
- Doesn't generalize to complex scenarios

**Neural Network (Our approach):**
- Learns automatically
- Can handle non-linearities
- Generalizes to variations

### vs. Deep RL (PPO/SAC)

**Deep RL:**
- More sample efficient
- Better for complex tasks
- Requires more code/complexity

**Genetic Algorithm:**
- Simpler to implement
- More intuitive
- Works great for this problem size
- No gradients needed

## Expected Results

### Early Training (Gen 0-20)

```
Generation 0
Best Fitness: -5.23  (Ball falls immediately)
Average Fitness: -8.45

Generation 10
Best Fitness: 2.15  (Ball stays on for ~1 second)
Average Fitness: -2.33
```

### Mid Training (Gen 20-60)

```
Generation 30
Best Fitness: 45.67  (Can balance for ~5 seconds!)
Average Fitness: 15.23

Generation 50
Best Fitness: 120.45  (Excellent balance ~10+ seconds)
Average Fitness: 67.89
```

### Expert Level (Gen 80+)

```
Generation 100
Best Fitness: 450.23  (Perfect balance, full episode!)
Average Fitness: 380.12
```

## What Makes This Challenging?

1. **Continuous state space** (infinite positions/velocities)
2. **Continuous action space** (infinite tilt angles)
3. **Delayed rewards** (need to plan ahead)
4. **Unstable equilibrium** (ball naturally falls off)
5. **Precision required** (small errors compound)

This is much harder than the 2D racing car!

## Key Takeaways

1. **3D physics simulation** adds realism and complexity
2. **Genetic algorithms work for continuous control** tasks
3. **PyBullet is powerful** for robotics/RL research
4. **Same GA principles** from 2D project apply to 3D
5. **Visualization helps understanding** - see learning happen!

## Next Steps

Want to learn more?

1. Train the robot and watch it improve
2. Try different reward functions
3. Modify the platform shape/size
4. Add multiple balls or obstacles
5. Try other RL algorithms (PPO, SAC)

---

**Happy balancing!** 🤖⚽

*Built with PyTorch + PyBullet + Evolution* 🧬
