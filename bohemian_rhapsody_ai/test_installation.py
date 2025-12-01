"""
Quick installation test script.
Verifies all dependencies and modules are working correctly.
"""

import sys

def test_imports():
    """Test that all required packages can be imported."""
    print("Testing imports...")

    try:
        import torch
        print(f"  ✓ PyTorch {torch.__version__}")
    except ImportError as e:
        print(f"  ✗ PyTorch not found: {e}")
        return False

    try:
        import numpy as np
        print(f"  ✓ NumPy {np.__version__}")
    except ImportError as e:
        print(f"  ✗ NumPy not found: {e}")
        return False

    try:
        import pygame
        print(f"  ✓ Pygame {pygame.version.ver}")
    except ImportError as e:
        print(f"  ✗ Pygame not found: {e}")
        return False

    try:
        import matplotlib
        print(f"  ✓ Matplotlib {matplotlib.__version__}")
    except ImportError as e:
        print(f"  ✗ Matplotlib not found: {e}")
        return False

    return True


def test_modules():
    """Test that all project modules can be imported."""
    print("\nTesting project modules...")

    try:
        from guitar_solo import GUITAR_SOLO, NUM_NOTES
        print(f"  ✓ guitar_solo.py ({NUM_NOTES} notes)")
    except Exception as e:
        print(f"  ✗ guitar_solo.py error: {e}")
        return False

    try:
        from neural_network import GuitarNetwork, create_random_network
        net = create_random_network()
        params = net.count_parameters()
        print(f"  ✓ neural_network.py ({params} parameters)")
    except Exception as e:
        print(f"  ✗ neural_network.py error: {e}")
        return False

    try:
        from genetic_algorithm import GeneticAlgorithm
        ga = GeneticAlgorithm(population_size=2)
        print(f"  ✓ genetic_algorithm.py")
    except Exception as e:
        print(f"  ✗ genetic_algorithm.py error: {e}")
        return False

    try:
        from guitar_player import GuitarPlayer
        player = GuitarPlayer(net)
        print(f"  ✓ guitar_player.py")
    except Exception as e:
        print(f"  ✗ guitar_player.py error: {e}")
        return False

    return True


def test_functionality():
    """Test basic functionality."""
    print("\nTesting functionality...")

    try:
        from neural_network import create_random_network
        from guitar_player import GuitarPlayer
        import numpy as np

        # Create a player
        net = create_random_network()
        player = GuitarPlayer(net)

        # Run a quick simulation
        for _ in range(10):
            player.update(0.016)  # 60 FPS

        fitness = player.get_fitness()
        print(f"  ✓ Quick simulation ran (fitness: {fitness:.1f})")

    except Exception as e:
        print(f"  ✗ Simulation error: {e}")
        return False

    try:
        # Test genetic algorithm
        from genetic_algorithm import GeneticAlgorithm
        import numpy as np

        ga = GeneticAlgorithm(population_size=3)
        fitness_scores = [10.0, 20.0, 15.0]
        best = ga.evolve(fitness_scores)

        print(f"  ✓ Genetic algorithm evolution works")

    except Exception as e:
        print(f"  ✗ Genetic algorithm error: {e}")
        return False

    return True


def main():
    """Run all tests."""
    print("="*60)
    print("Guitar Learning AI - Installation Test")
    print("="*60)

    all_passed = True

    # Test imports
    if not test_imports():
        all_passed = False
        print("\n❌ Import test failed!")
        print("\nTo fix, run:")
        print("  pip3 install -r requirements.txt")

    # Test modules
    if not test_modules():
        all_passed = False
        print("\n❌ Module test failed!")

    # Test functionality
    if not test_functionality():
        all_passed = False
        print("\n❌ Functionality test failed!")

    # Summary
    print("\n" + "="*60)
    if all_passed:
        print("✅ All tests passed! Installation is working correctly.")
        print("\nYou're ready to start training!")
        print("\nNext steps:")
        print("  python3 main.py          # Start training")
        print("  python3 main_visual.py   # Training with visualization")
        print("\nSee QUICKSTART.md for more information.")
    else:
        print("❌ Some tests failed. Please fix the errors above.")
        print("\nCommon fixes:")
        print("  pip3 install -r requirements.txt")
        print("  pip3 install torch numpy pygame matplotlib")
    print("="*60)

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
