"""
Generate a clean neural network + evolution loop diagram for Twitter.

Produces a single PNG image showing:
1. The bird's 4-input → 8-hidden → 1-output neural network
2. The genetic algorithm cycle (evaluate → select → crossover → mutate)

Designed to be readable on mobile (large fonts, high contrast, minimal clutter).
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


def draw_neural_network(ax):
    """Draw the 4→8→1 neural network with labeled inputs and output."""
    input_labels = ["Bird Y", "Distance", "Gap Y", "Velocity"]
    hidden_count = 8
    output_label = "Flap?"

    input_x = 0.15
    hidden_x = 0.50
    output_x = 0.85

    input_positions = np.linspace(0.2, 0.8, len(input_labels))
    hidden_positions = np.linspace(0.1, 0.9, hidden_count)
    output_y = 0.5

    node_radius = 0.025
    input_color = "#4ECDC4"
    hidden_color = "#FFD93D"
    output_color = "#FF6B6B"

    # Draw connections (input → hidden)
    for iy in input_positions:
        for hy in hidden_positions:
            ax.plot(
                [input_x, hidden_x], [iy, hy],
                color="#CCCCCC", linewidth=0.6, zorder=1
            )

    # Draw connections (hidden → output)
    for hy in hidden_positions:
        ax.plot(
            [hidden_x, output_x], [hy, output_y],
            color="#CCCCCC", linewidth=0.6, zorder=1
        )

    # Draw input nodes + labels
    for i, (y_pos, label) in enumerate(zip(input_positions, input_labels)):
        circle = plt.Circle(
            (input_x, y_pos), node_radius,
            color=input_color, ec="white", linewidth=2, zorder=3
        )
        ax.add_patch(circle)
        ax.text(
            input_x - 0.06, y_pos, label,
            ha="right", va="center", fontsize=11,
            fontweight="bold", color="#333333"
        )

    # Draw hidden nodes
    for y_pos in hidden_positions:
        circle = plt.Circle(
            (hidden_x, y_pos), node_radius,
            color=hidden_color, ec="white", linewidth=2, zorder=3
        )
        ax.add_patch(circle)

    # Draw output node + label
    circle = plt.Circle(
        (output_x, output_y), node_radius * 1.3,
        color=output_color, ec="white", linewidth=2, zorder=3
    )
    ax.add_patch(circle)
    ax.text(
        output_x + 0.06, output_y, output_label,
        ha="left", va="center", fontsize=13,
        fontweight="bold", color="#333333"
    )

    # Layer labels
    ax.text(
        input_x, 0.95, "Sensors",
        ha="center", va="center", fontsize=10, color="#888888"
    )
    ax.text(
        hidden_x, 0.95, "8 Neurons",
        ha="center", va="center", fontsize=10, color="#888888"
    )
    ax.text(
        output_x, 0.95, "Action",
        ha="center", va="center", fontsize=10, color="#888888"
    )

    # Title
    ax.text(
        0.50, 1.05, "The Bird's Brain (49 weights)",
        ha="center", va="center", fontsize=16,
        fontweight="bold", color="#222222"
    )

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.1)
    ax.set_aspect("equal")
    ax.axis("off")


def draw_evolution_loop(ax):
    """Draw the genetic algorithm cycle as a clean circular flow."""
    steps = [
        ("Play", "50 birds play\nthe game"),
        ("Die", "All birds\neventually die"),
        ("Rank", "Rank by\nfitness score"),
        ("Breed", "Best brains\ncrossover"),
        ("Mutate", "Random tweaks\nto weights"),
    ]

    n = len(steps)
    center_x, center_y = 0.5, 0.45
    radius = 0.30
    angles = np.linspace(90, 90 - 360, n, endpoint=False)

    colors = ["#4ECDC4", "#FF6B6B", "#FFD93D", "#A78BFA", "#F97316"]

    node_positions = []
    for i, (angle_deg, (title, desc)) in enumerate(zip(angles, steps)):
        angle_rad = np.radians(angle_deg)
        x = center_x + radius * np.cos(angle_rad)
        y = center_y + radius * np.sin(angle_rad)
        node_positions.append((x, y))

        circle = plt.Circle(
            (x, y), 0.08,
            color=colors[i], ec="white", linewidth=3, zorder=3, alpha=0.9
        )
        ax.add_patch(circle)

        ax.text(
            x, y + 0.015, title,
            ha="center", va="center", fontsize=11,
            fontweight="bold", color="white", zorder=4
        )

        desc_offset_x = np.cos(angle_rad) * 0.14
        desc_offset_y = np.sin(angle_rad) * 0.14
        ax.text(
            x + desc_offset_x, y + desc_offset_y, desc,
            ha="center", va="center", fontsize=8,
            color="#555555", zorder=4
        )

    # Draw arrows between nodes
    for i in range(n):
        x1, y1 = node_positions[i]
        x2, y2 = node_positions[(i + 1) % n]

        dx = x2 - x1
        dy = y2 - y1
        dist = np.sqrt(dx**2 + dy**2)

        shrink = 0.09
        start_x = x1 + (dx / dist) * shrink
        start_y = y1 + (dy / dist) * shrink
        end_x = x2 - (dx / dist) * shrink
        end_y = y2 - (dy / dist) * shrink

        ax.annotate(
            "",
            xy=(end_x, end_y),
            xytext=(start_x, start_y),
            arrowprops={
                "arrowstyle": "->,head_width=0.3,head_length=0.15",
                "color": "#AAAAAA",
                "lw": 2,
                "connectionstyle": "arc3,rad=0.15",
            },
            zorder=2,
        )

    # Center text
    ax.text(
        center_x, center_y, "Repeat\n100 gens",
        ha="center", va="center", fontsize=10,
        color="#999999", fontstyle="italic"
    )

    # Title
    ax.text(
        0.50, 0.95, "The Evolution Loop",
        ha="center", va="center", fontsize=16,
        fontweight="bold", color="#222222"
    )

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.axis("off")


def main():
    """Generate the combined diagram."""
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(14, 6),
        facecolor="white"
    )

    fig.subplots_adjust(wspace=0.05, left=0.02, right=0.98, top=0.92, bottom=0.02)

    draw_neural_network(ax1)
    draw_evolution_loop(ax2)

    output_path = "nn_evolution_diagram.png"
    fig.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Diagram saved to {output_path}")


if __name__ == "__main__":
    main()
