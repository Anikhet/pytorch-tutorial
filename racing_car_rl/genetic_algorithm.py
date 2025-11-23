import numpy as np
from neural_network import CarNeuralNetwork

class GeneticAlgorithm:
    """Genetic Algorithm for evolving car neural networks."""

    def __init__(self, population_size=50, elite_size=5, mutation_rate=0.1):
        """
        Args:
            population_size: Number of networks in population
            elite_size: Number of best networks to keep unchanged
            mutation_rate: Probability of mutating each weight
        """
        self.population_size = population_size
        self.elite_size = elite_size
        self.mutation_rate = mutation_rate

        # Create initial population
        self.population = [CarNeuralNetwork() for _ in range(population_size)]
        self.generation = 0
        self.best_fitness_history = []
        self.avg_fitness_history = []

    def evolve(self, fitness_scores):
        """
        Create next generation based on fitness scores.

        Args:
            fitness_scores: List of fitness values for each network
        """
        # Sort population by fitness (descending)
        sorted_indices = np.argsort(fitness_scores)[::-1]
        sorted_population = [self.population[i] for i in sorted_indices]
        sorted_fitness = [fitness_scores[i] for i in sorted_indices]

        # Track statistics
        self.best_fitness_history.append(sorted_fitness[0])
        self.avg_fitness_history.append(np.mean(fitness_scores))

        print(f"\nGeneration {self.generation}")
        print(f"Best Fitness: {sorted_fitness[0]:.2f}")
        print(f"Average Fitness: {np.mean(fitness_scores):.2f}")
        print(f"Worst Fitness: {sorted_fitness[-1]:.2f}")

        # Create new population
        new_population = []

        # 1. Keep elite (best performers)
        for i in range(self.elite_size):
            new_population.append(sorted_population[i].copy())

        # 2. Create children through crossover and mutation
        while len(new_population) < self.population_size:
            # Select parents using tournament selection
            parent1 = self._tournament_selection(sorted_population, sorted_fitness)
            parent2 = self._tournament_selection(sorted_population, sorted_fitness)

            # Crossover
            child = CarNeuralNetwork.crossover(parent1, parent2)

            # Mutation
            child.mutate(self.mutation_rate)

            new_population.append(child)

        self.population = new_population
        self.generation += 1

        return sorted_population[0]  # Return best network

    def _tournament_selection(self, population, fitness_scores, tournament_size=5):
        """
        Select a network using tournament selection.

        Args:
            population: List of networks
            fitness_scores: List of fitness scores
            tournament_size: Number of networks to compete

        Returns:
            Selected network
        """
        # Randomly select tournament_size networks
        tournament_indices = np.random.choice(
            len(population),
            size=min(tournament_size, len(population)),
            replace=False
        )

        # Find the best one
        best_idx = tournament_indices[0]
        best_fitness = fitness_scores[tournament_indices[0]]

        for idx in tournament_indices[1:]:
            if fitness_scores[idx] > best_fitness:
                best_idx = idx
                best_fitness = fitness_scores[idx]

        return population[best_idx]

    def get_population(self):
        """Return current population."""
        return self.population

    def get_statistics(self):
        """Return training statistics."""
        return {
            'generation': self.generation,
            'best_fitness': self.best_fitness_history,
            'avg_fitness': self.avg_fitness_history
        }

    def save_best(self, filename):
        """Save the best network to file."""
        if self.best_fitness_history:
            # The first network in population is the best after evolve()
            best_network = self.population[0]
            import torch
            torch.save(best_network.state_dict(), filename)
            print(f"Best network saved to {filename}")

    def load_best(self, filename):
        """Load a saved network."""
        import torch
        network = CarNeuralNetwork()
        network.load_state_dict(torch.load(filename))
        return network
