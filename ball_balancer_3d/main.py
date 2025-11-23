import numpy as np
import matplotlib.pyplot as plt
from environment_pygame import BalancingEnvironment  # Using Pygame 3D version
from genetic_algorithm import GeneticAlgorithm

# Configuration
POPULATION_SIZE = 10
ELITE_SIZE = 2
MUTATION_RATE = 0.1
MAX_EPISODE_STEPS = 500  # About 2 seconds at 240Hz
GENERATIONS = 100

def evaluate_network(env, network, max_steps=MAX_EPISODE_STEPS, render=True):
    """
    Evaluate a single network in the environment.

    Args:
        env: BalancingEnvironment instance
        network: Neural network to evaluate
        max_steps: Maximum steps per episode
        render: Whether to show visualization

    Returns:
        total_reward: Cumulative reward
        steps_survived: Number of steps before failure
    """
    observation = env.reset()
    total_reward = 0
    steps_survived = 0

    for step in range(max_steps):
        # Get action from network
        action = network.predict(observation)

        # Step environment
        observation, reward, done, info = env.step(action)
        total_reward += reward
        steps_survived += 1

        # Render if requested
        if render:
            env.render()

        if done:
            break

    return total_reward, steps_survived

def run_generation(env, ga, generation_num):
    """
    Evaluate all networks in the population for one generation.

    Args:
        env: BalancingEnvironment
        ga: GeneticAlgorithm instance
        generation_num: Current generation number

    Returns:
        List of fitness scores
    """
    population = ga.get_population()
    fitness_scores = []

    for i, network in enumerate(population):
        # Only visualize the first robot (elite from previous generation)
        show_viz = (i == 0)

        # Evaluate this network
        total_reward, steps = evaluate_network(env, network, render=show_viz)

        # Fitness is the total reward earned
        fitness = total_reward

        fitness_scores.append(fitness)

        # Print progress
        if i == 0:  # Best from previous generation
            state_info = env.get_state_info()
            print(f"  Best robot: Reward={total_reward:.2f}, "
                  f"Steps={steps}, Distance={state_info['ball_distance_from_center']:.3f}m")

    return fitness_scores

def plot_fitness_history(ga):
    """Plot fitness history over generations."""
    stats = ga.get_statistics()

    plt.figure(figsize=(10, 6))
    plt.plot(stats['best_fitness'], label='Best Fitness', linewidth=2)
    plt.plot(stats['avg_fitness'], label='Average Fitness', linewidth=2)
    plt.xlabel('Generation')
    plt.ylabel('Fitness (Total Reward)')
    plt.title('Evolution Progress - Ball Balancing Robot')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('fitness_history.png')
    print("Fitness history plot saved to fitness_history.png")

def main():
    """Main training loop."""
    print("=" * 60)
    print("3D Ball Balancing Robot - Pygame Visualization")
    print("=" * 60)
    print(f"Population Size: {POPULATION_SIZE}")
    print(f"Elite Size: {ELITE_SIZE}")
    print(f"Mutation Rate: {MUTATION_RATE}")
    print(f"Max Steps per Episode: {MAX_EPISODE_STEPS}")
    print(f"Target Generations: {GENERATIONS}")
    print("=" * 60)
    print("\nWatch the 3D isometric view!")
    print("Blue platform tilts to balance the red ball.")
    print("Visualizing best robot from each generation.\n")

    # Create environment with GUI
    env = BalancingEnvironment(gui=True)

    # Initialize genetic algorithm
    ga = GeneticAlgorithm(
        population_size=POPULATION_SIZE,
        elite_size=ELITE_SIZE,
        mutation_rate=MUTATION_RATE
    )

    try:
        # Training loop
        for generation in range(GENERATIONS):
            # Evaluate all networks in population
            fitness_scores = run_generation(env, ga, generation)

            # Evolve to next generation
            best_network = ga.evolve(fitness_scores)

            # Save best network periodically
            if generation % 10 == 0 and generation > 0:
                ga.save_best(f'best_network_gen_{generation}.pth')

            # Early stopping if solved
            if ga.best_fitness_history[-1] > 400:  # Arbitrary threshold
                print(f"\n🎉 Task solved at generation {generation}!")
                print(f"Best fitness: {ga.best_fitness_history[-1]:.2f}")
                break

    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")

    # Cleanup
    print("\nTraining finished!")
    print(f"Total generations: {ga.generation}")
    print(f"Best fitness achieved: {max(ga.best_fitness_history):.2f}")

    # Save final best network
    ga.save_best('best_network_final.pth')

    # Plot fitness history
    if ga.best_fitness_history:
        plot_fitness_history(ga)

    # Close environment
    env.close()

if __name__ == "__main__":
    main()
