# Genetic Algorithms - Evolution in Action

## What Is a Genetic Algorithm?

A **genetic algorithm** is an optimization technique inspired by natural evolution. It uses concepts like:
- **Survival of the fittest**
- **Reproduction**
- **Mutation**
- **Natural selection**

## The Core Idea

> In nature, organisms with better traits survive and reproduce. Over many generations, the population becomes better adapted to their environment.

We apply the same principle to our cars!

## Natural Evolution vs Our Cars

| Nature | Our Project |
|--------|-------------|
| Organisms | Cars with neural networks |
| DNA | Network weights (138 numbers) |
| Environment | Race track |
| Fitness | How far they travel |
| Reproduction | Combining weights from parents |
| Mutation | Random weight changes |
| Generation | One round of 5 cars |

## The Genetic Algorithm Cycle

```
┌─────────────────────────────────────┐
│  GENERATION 1                       │
│  5 cars with random brains          │
│  Most crash immediately             │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  EVALUATION                         │
│  Car 1: Distance = 45 (worst)       │
│  Car 2: Distance = 120              │
│  Car 3: Distance = 89               │
│  Car 4: Distance = 310 (BEST!)      │
│  Car 5: Distance = 76               │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  SELECTION                          │
│  Car 4 is the best - keep it!       │
│  Others will be replaced            │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  CROSSOVER & MUTATION               │
│  Create 4 new cars from Car 4       │
│  Add random variations              │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  GENERATION 2                       │
│  1 old best car + 4 new children    │
│  Hopefully better than Gen 1!       │
└──────────────┬──────────────────────┘
               │
               ▼
         (Repeat...)
```

## Step-by-Step Breakdown

### Step 1: Initialize Population

**Generation 0:**
```python
population = []
for i in range(5):
    car = Car()
    car.brain = NeuralNetwork()  # Random weights!
    population.append(car)
```

**What happens:**
- Create 5 cars
- Each gets a neural network with **random weights**
- Weights are small random numbers (like -0.5 to +0.5)

**Result:**
- Cars drive completely randomly
- Most crash within seconds
- Pure chaos!

### Step 2: Evaluation (Fitness Function)

Let each car drive until it crashes or time runs out:

```python
for car in population:
    while car.alive and time < 5_seconds:
        sensors = car.get_sensor_readings()
        action = car.brain.predict(sensors)
        car.update(action)
        car.distance_traveled += car.velocity * dt

    car.fitness = car.distance_traveled
```

**Fitness function:**
```python
fitness = distance_traveled + (10 if still_alive else 0)
```

**Example results:**
```
Car 1: 45.2   fitness (crashed quickly)
Car 2: 120.8  fitness
Car 3: 89.3   fitness
Car 4: 310.5  fitness (WINNER!)
Car 5: 76.1   fitness
```

**Why this fitness function?**
- Simple and intuitive
- Rewards exploration (going far)
- Small bonus for survival
- No complex calculations needed

### Step 3: Selection

**Elitism Strategy:**
Keep the best car unchanged!

```python
# Sort by fitness (highest first)
sorted_cars = sort(population, key=lambda x: x.fitness, reverse=True)

next_generation = []
next_generation.append(sorted_cars[0])  # Best car survives!
```

**Why keep the best?**
- Guarantees we never get worse
- Best solution is always available
- Provides stability to evolution

### Step 4: Crossover (Reproduction)

Create new cars by combining weights from successful parents.

**Tournament Selection:**
```python
def select_parent(population, fitness_scores):
    # Pick 3 random cars
    contestants = random.sample(population, 3)

    # Return the best of the 3
    return max(contestants, key=lambda c: c.fitness)
```

**Why tournament instead of always picking the best?**
- Adds diversity
- Gives decent cars a chance to reproduce
- Prevents premature convergence

**Single-Point Crossover:**
```python
def crossover(parent1, parent2):
    child = NeuralNetwork()

    # Get weights from both parents
    weights1 = parent1.get_weights()  # 138 numbers
    weights2 = parent2.get_weights()  # 138 numbers

    # Pick a random split point
    split = random.randint(0, 138)

    # First half from parent1, second half from parent2
    child_weights = weights1[:split] + weights2[split:]

    child.set_weights(child_weights)
    return child
```

**Visual example:**
```
Parent 1: [0.5, 0.2, -0.3, 0.8, 0.1, -0.2, 0.4, ...]
Parent 2: [0.1, -0.4, 0.6, -0.1, 0.9, 0.3, -0.5, ...]
                            ↑ split point = 4

Child:    [0.5, 0.2, -0.3, 0.8, 0.9, 0.3, -0.5, ...]
          └─── from parent 1 ───┘ └─── from parent 2 ───┘
```

### Step 5: Mutation

Add random changes to keep things interesting!

```python
def mutate(network, mutation_rate=0.1, mutation_scale=0.3):
    weights = network.get_weights()

    for i in range(len(weights)):
        # 10% chance to mutate each weight
        if random.random() < mutation_rate:
            # Add small random value
            weights[i] += random.gauss(0, mutation_scale)

    network.set_weights(weights)
```

**Example:**
```
Before mutation: [0.5, 0.2, -0.3, 0.8, 0.1]
Mutations:        [---  ---  +0.2  ---  -0.1]  (10% of 138 weights)
After mutation:  [0.5, 0.2, -0.1, 0.8, 0.0]
```

**Why mutate?**
- **Exploration** - Tries new solutions
- **Escapes local optima** - Can get unstuck
- **Adds diversity** - Prevents all cars becoming identical
- **Fine-tuning** - Small adjustments improve performance

**Mutation rate (10%):**
- Too low: Evolution is too slow
- Too high: Good solutions get destroyed
- 10% is a good balance

**Mutation scale (0.3):**
- Controls how big the random changes are
- Small values = gentle exploration
- Large values = wild changes

### Step 6: Create Next Generation

```python
def create_next_generation():
    next_gen = []

    # 1. Elitism - Keep best car
    next_gen.append(best_car.copy())

    # 2. Create 4 children through crossover + mutation
    for _ in range(4):
        parent1 = tournament_selection(population)
        parent2 = tournament_selection(population)

        child = crossover(parent1, parent2)
        mutate(child)

        next_gen.append(child)

    return next_gen  # 5 cars ready for next generation!
```

## Evolution Over Time

### Generation 1-5: Random Chaos
- **Fitness:** 50-200
- **Behavior:** Drive in circles, crash immediately
- **Learning:** Finding basic patterns (wall = bad)

### Generation 10-20: Basic Driving
- **Fitness:** 200-500
- **Behavior:** Can avoid walls for a bit, make simple turns
- **Learning:** Steering correlates with sensors

### Generation 30-50: Competent Drivers
- **Fitness:** 500-1000
- **Behavior:** Navigate several turns, smoother driving
- **Learning:** Speed control, turn anticipation

### Generation 100+: Expert Racers
- **Fitness:** 1000-2000+
- **Behavior:** Complete full laps, optimal racing lines
- **Learning:** Fine-tuned control, efficient paths

## Why Does This Work?

### The Mathematics

Each generation, the average fitness increases:

```
E[fitness(gen_n+1)] > E[fitness(gen_n)]
```

**Why?**
1. Best solution always survives (elitism)
2. Children inherit good traits from parents
3. Mutation occasionally finds improvements
4. Bad solutions are eliminated

### The Search Space

With 138 weights, there are **infinite** possible cars!

**Genetic Algorithm explores this space efficiently:**
- Starts random (broad exploration)
- Converges toward good solutions (exploitation)
- Mutation prevents getting stuck
- Crossover combines good traits

## Comparison to Other Approaches

### Gradient Descent (Backpropagation)
```
Pros: Very efficient, proven for many tasks
Cons: Needs differentiable reward, can get stuck
```

### Genetic Algorithm (Our approach)
```
Pros: No gradients needed, very visual, explores widely
Cons: Needs many evaluations, slower for complex tasks
```

### Reinforcement Learning (DQN, PPO)
```
Pros: State-of-the-art for many problems
Cons: More complex, harder to implement/debug
```

**For this project, GA is perfect because:**
- ✅ Easy to understand and implement
- ✅ Very visual (watch evolution happen!)
- ✅ Works great for this problem size
- ✅ Educational value is high

## Key Parameters and Tuning

### Population Size (Currently: 5)
```
Too small (1-2): Not enough diversity, slow evolution
Just right (5-10): Good for this project, fast, visual
Too large (50-100): More robust but slower, harder to track
```

### Elite Size (Currently: 1)
```
Too small (0): Might lose best solution!
Just right (1): Keeps best, allows diversity
Too large (3+): Too conservative, not enough new ideas
```

### Mutation Rate (Currently: 0.1 = 10%)
```
Too low (1%): Evolution too slow
Just right (10%): Good exploration/exploitation balance
Too high (50%): Destroys good solutions
```

### Mutation Scale (Currently: 0.3)
```
Too small (0.01): Changes too subtle
Just right (0.3): Noticeable but not destructive
Too large (2.0): Completely randomizes weights
```

## Common Pitfalls

### Premature Convergence
**Problem:** All cars become identical too quickly

**Solution:**
- Increase mutation rate
- Increase population size
- Add diversity bonus to fitness

### No Improvement
**Problem:** Fitness plateaus, stuck at local optimum

**Solution:**
- Increase mutation rate temporarily
- Reset population partially
- Change fitness function

### Too Much Randomness
**Problem:** Fitness jumps around, no consistent progress

**Solution:**
- Increase elite size
- Decrease mutation rate
- Run for more generations

## Visualizing Evolution

Watch for these patterns in the UI:

**Early generations:**
```
Best Fitness: 150
Avg Fitness:  80
Worst Fitness: 30
```
Large variance = lots of experimentation

**Later generations:**
```
Best Fitness: 1200
Avg Fitness:  950
Worst Fitness: 800
```
Small variance = population converging on solution

## The Code

Our implementation in `genetic_algorithm.py`:

```python
class GeneticAlgorithm:
    def __init__(self, population_size=5, elite_size=1, mutation_rate=0.1):
        # Create 5 random networks
        self.population = [CarNeuralNetwork() for _ in range(5)]

    def evolve(self, fitness_scores):
        # 1. Sort by fitness
        sorted_pop = sort_by_fitness(self.population, fitness_scores)

        # 2. Keep best (elitism)
        new_pop = [sorted_pop[0].copy()]

        # 3. Create children
        while len(new_pop) < 5:
            parent1 = tournament_selection(sorted_pop)
            parent2 = tournament_selection(sorted_pop)

            child = crossover(parent1, parent2)
            mutate(child)

            new_pop.append(child)

        self.population = new_pop
```

## Key Takeaways

1. **Genetic algorithms mimic natural evolution** to solve optimization problems

2. **Fitness function is crucial** - it defines what "good" means

3. **Balance exploration vs exploitation** through mutation and elitism

4. **Evolution requires patience** - many generations for good results

5. **Highly visual and intuitive** - see learning happen in real-time!

6. **No gradients or backprop needed** - works on any fitness function

## Next Steps

Now you understand how the brains work (neural networks) and how they improve (genetic algorithms).

Next, let's see how cars "see" their environment!

Continue to **03_SENSOR_SYSTEM.md** → 👀
