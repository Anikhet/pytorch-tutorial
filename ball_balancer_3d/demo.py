import sys
import time
from environment import BalancingEnvironment
from neural_network import BalancerNeuralNetwork
import torch

def run_demo(network_file='best_network_final.pth', num_episodes=5):
    """
    Run a demo of the trained network.

    Args:
        network_file: Path to saved network weights
        num_episodes: Number of episodes to run
    """
    print(f"Loading network from {network_file}...")

    try:
        network = BalancerNeuralNetwork()
        network.load_state_dict(torch.load(network_file))
        print("Network loaded successfully!\n")
    except FileNotFoundError:
        print(f"Error: {network_file} not found. Please train the network first.")
        return

    # Create environment with GUI
    env = BalancingEnvironment(gui=True)

    print("=" * 60)
    print("3D Ball Balancing Robot - Demo Mode")
    print("=" * 60)
    print(f"Running {num_episodes} episodes...")
    print("Watch the robot balance the ball on the tilting platform!")
    print("Close the PyBullet window to exit.\n")

    try:
        for episode in range(num_episodes):
            print(f"\n--- Episode {episode + 1}/{num_episodes} ---")

            observation = env.reset()
            total_reward = 0
            steps = 0
            max_steps = 1000  # Longer episodes for demo

            for step in range(max_steps):
                # Get action from network
                action = network.predict(observation)

                # Step environment
                observation, reward, done, info = env.step(action)
                total_reward += reward
                steps += 1

                # Real-time visualization
                time.sleep(1./240.)

                if done:
                    break

            # Episode summary
            print(f"  Total Reward: {total_reward:.2f}")
            print(f"  Steps Survived: {steps}")
            print(f"  Final Distance from Center: {info['distance_from_center']:.3f}m")

            if steps >= max_steps:
                print("  ✓ Episode completed successfully!")
            else:
                print("  ✗ Ball fell off platform")

            # Pause between episodes
            if episode < num_episodes - 1:
                print("\nStarting next episode in 2 seconds...")
                time.sleep(2)

    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user")

    # Cleanup
    env.close()
    print("\nDemo finished!")

if __name__ == "__main__":
    # Allow user to specify network file
    network_file = 'best_network_final.pth'
    if len(sys.argv) > 1:
        network_file = sys.argv[1]

    run_demo(network_file)
