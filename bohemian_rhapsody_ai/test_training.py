"""
Quick test to verify training loop works without GUI.
"""

from guitar_player import GuitarPlayer
from neural_network import create_random_network
from genetic_algorithm import GeneticAlgorithm

print("Testing training without GUI...")

# Create GA
ga = GeneticAlgorithm(population_size=3, elite_size=1)

# Run 3 generations
for gen in range(3):
    print(f"\nGeneration {gen}")

    # Create players
    players = [GuitarPlayer(network) for network in ga.get_population()]

    # Simulate for a bit
    for _ in range(100):  # 100 timesteps
        for player in players:
            if player.alive:
                player.update(0.016)  # 60 FPS

    # Get fitness
    fitness_scores = [p.get_fitness() for p in players]
    print(f"Fitness scores: {[f'{f:.1f}' for f in fitness_scores]}")

    # Evolve
    ga.evolve(fitness_scores)

print("\n✅ Training loop works!")
print("The issue might be with pygame window/display.")
