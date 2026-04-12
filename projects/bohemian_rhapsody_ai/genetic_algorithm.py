"""
Genetic algorithm for evolving guitar-playing neural networks.
"""

import numpy as np
from neural_network import GuitarNetwork, create_random_network


class GeneticAlgorithm:
    """
    Implements genetic algorithm to evolve a population of neural networks.

    The algorithm works by:
    1. Evaluating fitness of all networks in population
    2. Selecting the best performers (elitism)
    3. Creating new networks through tournament selection, crossover, and mutation
    4. Repeating until performance converges
    """

    def __init__(self, population_size=10, elite_size=2, mutation_rate=0.1):
        """
        Initialize the genetic algorithm.

        Args:
            population_size: Number of networks in population
            elite_size: Number of top performers to keep unchanged
            mutation_rate: Probability of mutating each weight
        """
        self.population_size = population_size
        self.elite_size = elite_size
        self.mutation_rate = mutation_rate

        # Create initial random population
        self.population = [create_random_network() for _ in range(population_size)]

        # Statistics tracking
        self.generation = 0
        self.best_fitness_history = []
        self.avg_fitness_history = []
        self.best_network = None
        self.best_fitness = -float('inf')

    def evolve(self, fitness_scores):
        """
        Evolve the population based on fitness scores.

        Args:
            fitness_scores: List of fitness values for each network

        Returns:
            Best network from this generation
        """
        # Sort population by fitness (descending)
        sorted_indices = np.argsort(fitness_scores)[::-1]
        sorted_population = [self.population[i] for i in sorted_indices]
        sorted_fitness = [fitness_scores[i] for i in sorted_indices]

        # Track statistics
        best_fitness_gen = sorted_fitness[0]
        avg_fitness_gen = np.mean(sorted_fitness)
        self.best_fitness_history.append(best_fitness_gen)
        self.avg_fitness_history.append(avg_fitness_gen)

        # Update best network if improved
        if best_fitness_gen > self.best_fitness:
            self.best_fitness = best_fitness_gen
            self.best_network = sorted_population[0].copy()

        print(f"Generation {self.generation}: "
              f"Best={best_fitness_gen:.2f}, "
              f"Avg={avg_fitness_gen:.2f}, "
              f"Worst={sorted_fitness[-1]:.2f}")

        # Create new population
        new_population = []

        # Elitism: Keep top performers unchanged
        for i in range(self.elite_size):
            new_population.append(sorted_population[i].copy())

        # Fill rest of population with offspring
        while len(new_population) < self.population_size:
            # Tournament selection for parents
            parent1 = self._tournament_selection(sorted_population, sorted_fitness)
            parent2 = self._tournament_selection(sorted_population, sorted_fitness)

            # Crossover
            child = GuitarNetwork.crossover(parent1, parent2)

            # Mutation
            child.mutate(
                mutation_rate=self.mutation_rate,
                mutation_scale=0.3
            )

            new_population.append(child)

        self.population = new_population
        self.generation += 1

        return self.best_network

    def _tournament_selection(self, population, fitness_scores, tournament_size=5):
        """
        Select a network using tournament selection.

        Args:
            population: List of networks
            fitness_scores: List of fitness scores
            tournament_size: Number of networks in tournament

        Returns:
            Selected network
        """
        # Randomly select tournament_size networks
        tournament_size = min(tournament_size, len(population))
        tournament_indices = np.random.choice(
            len(population),
            size=tournament_size,
            replace=False
        )

        # Find best in tournament
        best_idx = tournament_indices[0]
        best_fitness = fitness_scores[best_idx]

        for idx in tournament_indices[1:]:
            if fitness_scores[idx] > best_fitness:
                best_idx = idx
                best_fitness = fitness_scores[idx]

        return population[best_idx]

    def get_population(self):
        """Get current population of networks."""
        return self.population

    def get_statistics(self):
        """Get evolution statistics."""
        return {
            'generation': self.generation,
            'best_fitness': self.best_fitness,
            'best_fitness_history': self.best_fitness_history,
            'avg_fitness_history': self.avg_fitness_history
        }

    def save_best(self, filepath):
        """Save the best network to file."""
        if self.best_network:
            self.best_network.save(filepath)
            print(f"Best network saved to {filepath}")

    def load_best(self, filepath):
        """Load a network from file as the best network."""
        network = GuitarNetwork()
        network.load(filepath)
        self.best_network = network
        print(f"Network loaded from {filepath}")
        return network


if __name__ == "__main__":
    # Test the genetic algorithm
    print("Testing Genetic Algorithm...")

    ga = GeneticAlgorithm(population_size=5, elite_size=1, mutation_rate=0.1)
    print(f"Population size: {len(ga.get_population())}")

    # Simulate a few generations with random fitness
    for gen in range(3):
        # Random fitness scores for testing
        fitness_scores = np.random.rand(5) * 100

        print(f"\nGeneration {gen} fitness scores: {fitness_scores}")
        best_net = ga.evolve(fitness_scores)

    # Show statistics
    stats = ga.get_statistics()
    print(f"\n\nFinal statistics:")
    print(f"  Generation: {stats['generation']}")
    print(f"  Best fitness: {stats['best_fitness']:.2f}")
    print(f"  Best fitness history: {stats['best_fitness_history']}")
    print(f"  Avg fitness history: {stats['avg_fitness_history']}")

    print("\nGenetic Algorithm tests passed!")
