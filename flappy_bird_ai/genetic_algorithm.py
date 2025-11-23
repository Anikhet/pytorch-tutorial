import numpy as np
from bird import Bird

class GeneticAlgorithm:
    """Genetic Algorithm for evolving Flappy Bird neural networks."""

    def __init__(self, population_size=50, elite_size=5, mutation_rate=0.1):
        """
        Args:
            population_size: Number of birds in population
            elite_size: Number of best birds to keep unchanged
            mutation_rate: Probability of mutating each weight
        """
        self.population_size = population_size
        self.elite_size = elite_size
        self.mutation_rate = mutation_rate

        # Create initial population
        self.population = [Bird() for _ in range(population_size)]
        self.generation = 0
        self.best_fitness_history = []
        self.avg_fitness_history = []
        self.best_score_history = []

    def evolve(self, fitness_scores):
        """
        Create next generation based on fitness scores.

        Args:
            fitness_scores: List of fitness values for each bird
        """
        # Sort population by fitness (descending)
        sorted_indices = np.argsort(fitness_scores)[::-1]
        sorted_population = [self.population[i] for i in sorted_indices]
        sorted_fitness = [fitness_scores[i] for i in sorted_indices]

        # Track statistics
        best_bird = sorted_population[0]
        self.best_fitness_history.append(sorted_fitness[0])
        self.avg_fitness_history.append(np.mean(fitness_scores))
        self.best_score_history.append(best_bird.score)

        print(f"\nGeneration {self.generation}")
        print(f"Best Score: {best_bird.score}")
        print(f"Best Fitness: {sorted_fitness[0]:.2f}")
        print(f"Average Fitness: {np.mean(fitness_scores):.2f}")
        print(f"Alive at end: {sum(1 for b in self.population if b.alive)}")

        # Create new population
        new_population = []

        # 1. Keep elite (best performers)
        for i in range(self.elite_size):
            new_population.append(sorted_population[i].copy())

        # 2. Create children through crossover and mutation
        while len(new_population) < self.population_size:
            # Select parents using tournament selection
            parent1_brain = self._tournament_selection(sorted_population, sorted_fitness).brain
            parent2_brain = self._tournament_selection(sorted_population, sorted_fitness).brain

            # Crossover
            child_brain = parent1_brain.crossover(parent1_brain, parent2_brain)

            # Mutation
            child_brain.mutate(self.mutation_rate)

            # Create new bird with evolved brain
            child_bird = Bird(brain=child_brain)
            new_population.append(child_bird)

        self.population = new_population
        self.generation += 1

        return sorted_population[0]  # Return best bird

    def _tournament_selection(self, population, fitness_scores, tournament_size=5):
        """
        Select a bird using tournament selection.

        Args:
            population: List of birds
            fitness_scores: List of fitness scores
            tournament_size: Number of birds to compete

        Returns:
            Selected bird
        """
        # Randomly select tournament_size birds
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
            'avg_fitness': self.avg_fitness_history,
            'best_score': self.best_score_history
        }

    def save_best(self, filename):
        """Save the best bird's brain to file."""
        if self.best_fitness_history:
            best_bird = self.population[0]
            import torch
            torch.save(best_bird.brain.state_dict(), filename)
            print(f"Best bird saved to {filename}")

    def load_best(self, filename):
        """Load a saved bird brain."""
        import torch
        from neural_network import BirdNeuralNetwork
        brain = BirdNeuralNetwork()
        brain.load_state_dict(torch.load(filename))
        return Bird(brain=brain)
