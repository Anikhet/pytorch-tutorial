"""
Visualization utilities for federated learning experiments.

Generates two types of plots:
  1. Convergence curves: accuracy vs. round for different configurations
  2. Privacy tradeoff:   final accuracy vs. noise multiplier

All plots are saved as PNG files for easy sharing and inclusion in reports.
"""

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for saving to file
import matplotlib.pyplot as plt

from server import run_simulation
from privacy import compute_privacy_spent


def plot_convergence(histories_dict, title="FL Convergence", save_path="convergence.png"):
    """
    Plot accuracy over rounds for multiple FL configurations.

    Each entry in histories_dict is plotted as a separate line,
    making it easy to compare IID vs non-IID, DP vs non-DP, etc.

    Args:
        histories_dict: Maps label string -> {round: accuracy} dict.
                        Example: {"IID": {1: 85.0, 2: 90.0, ...}}
        title:          Plot title.
        save_path:      Where to save the PNG file.
    """
    plt.figure(figsize=(10, 6))

    for label, history in histories_dict.items():
        rounds = sorted(history.keys())
        accuracies = [history[r] for r in rounds]
        plt.plot(rounds, accuracies, marker="o", label=label, linewidth=2)

    plt.xlabel("Communication Round", fontsize=12)
    plt.ylabel("Test Accuracy (%)", fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

    print(f"Saved convergence plot to {save_path}")


def plot_privacy_tradeoff(results, save_path="privacy_tradeoff.png"):
    """
    Plot the privacy-accuracy tradeoff.

    Shows how increasing the noise multiplier (better privacy)
    affects the final model accuracy. Also annotates each point
    with its estimated epsilon value.

    Args:
        results:   List of (noise_multiplier, final_accuracy, epsilon) tuples.
        save_path: Where to save the PNG file.
    """
    noise_values = [r[0] for r in results]
    accuracies = [r[1] for r in results]
    epsilons = [r[2] for r in results]

    plt.figure(figsize=(10, 6))
    plt.plot(noise_values, accuracies, marker="s", color="crimson", linewidth=2)

    # Annotate each point with its epsilon value
    for noise, acc, eps in results:
        label = f"eps={eps:.1f}"
        plt.annotate(
            label, (noise, acc),
            textcoords="offset points", xytext=(0, 12),
            ha="center", fontsize=9,
        )

    plt.xlabel("Noise Multiplier", fontsize=12)
    plt.ylabel("Final Test Accuracy (%)", fontsize=12)
    plt.title("Privacy-Accuracy Tradeoff", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

    print(f"Saved privacy tradeoff plot to {save_path}")


def run_convergence_comparison(num_clients=5, num_rounds=10):
    """
    Run IID and non-IID simulations and plot their convergence.

    Args:
        num_clients: Number of federated clients.
        num_rounds:  Number of communication rounds.

    Returns:
        Dict with "IID" and "Non-IID" histories.
    """
    print("\n--- Running IID simulation ---")
    iid_history = run_simulation(
        num_clients=num_clients, num_rounds=num_rounds, iid=True
    )

    print("\n--- Running Non-IID simulation ---")
    non_iid_history = run_simulation(
        num_clients=num_clients, num_rounds=num_rounds, iid=False
    )

    histories = {"IID": iid_history, "Non-IID": non_iid_history}
    plot_convergence(histories, title="IID vs Non-IID Convergence")

    return histories


def run_privacy_comparison(num_clients=5, num_rounds=5):
    """
    Run simulations with varying noise levels and plot the tradeoff.

    Tests multiple noise multipliers to show how privacy strength
    affects model utility.

    Args:
        num_clients: Number of federated clients.
        num_rounds:  Number of communication rounds.

    Returns:
        List of (noise_multiplier, final_accuracy, epsilon) tuples.
    """
    noise_levels = [0.1, 0.5, 1.0, 2.0, 5.0]
    results = []

    for noise in noise_levels:
        print(f"\n--- DP simulation (noise={noise}) ---")
        history = run_simulation(
            num_clients=num_clients,
            num_rounds=num_rounds,
            use_dp=True,
            noise_multiplier=noise,
        )
        final_accuracy = history[num_rounds]
        epsilon = compute_privacy_spent(
            noise_multiplier=noise,
            num_rounds=num_rounds,
            num_samples=60000,
        )
        results.append((noise, final_accuracy, epsilon))
        print(f"  -> Accuracy: {final_accuracy:.2f}%, Epsilon: {epsilon:.2f}")

    plot_privacy_tradeoff(results)
    return results


if __name__ == "__main__":
    print("=== Federated Learning Visualization ===\n")

    print("1) Convergence comparison (IID vs Non-IID)")
    run_convergence_comparison(num_clients=5, num_rounds=10)

    print("\n2) Privacy-accuracy tradeoff")
    run_privacy_comparison(num_clients=5, num_rounds=5)

    print("\nAll plots saved!")
