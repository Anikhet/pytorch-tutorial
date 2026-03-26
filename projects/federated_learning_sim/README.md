# Federated Learning Simulator

A visual, beginner-friendly federated learning simulator that demonstrates
training across multiple clients without sharing data.

## What This Project Does

- Simulates federated learning with N clients
- Supports IID and non-IID data splits
- Implements FedAvg algorithm
- Includes differential privacy (DP-SGD)
- Visualizes convergence and privacy tradeoffs

## Quick Start

```bash
pip install -r requirements.txt
python server.py         # Run FL simulation (5 clients, 10 rounds)
python visualize.py      # Generate convergence plots
```

## Project Structure

| File           | Description                              |
|----------------|------------------------------------------|
| `server.py`    | FL server with weight averaging          |
| `client.py`    | FL client with local training            |
| `model.py`     | Shared CNN model                         |
| `data.py`      | IID and non-IID data partitioning        |
| `privacy.py`   | Differential privacy (DP-SGD)            |
| `visualize.py` | Convergence and privacy plots            |

## How Federated Learning Works

1. Server sends global model to all clients
2. Each client trains locally on their private data
3. Clients send updated weights (NOT data) to server
4. Server averages the weights (FedAvg)
5. Repeat until convergence

```
Round 1:
  Server ──(global weights)──> Client 1, Client 2, ..., Client N
  Client 1 ──(updated weights)──┐
  Client 2 ──(updated weights)──┤
  ...                            ├──> Server averages
  Client N ──(updated weights)──┘
  Server updates global model

Round 2:
  Repeat...
```

## Differential Privacy

This simulator includes DP-SGD (Differentially Private Stochastic Gradient
Descent) to protect individual training examples:

- **Gradient clipping**: Bounds each sample's influence on the model
- **Noise addition**: Adds calibrated Gaussian noise to gradients
- **Privacy accounting**: Tracks cumulative privacy loss (epsilon)

Higher noise = more privacy but slower/worse convergence. The `visualize.py`
script generates plots showing this tradeoff.

## Configuration

Key parameters in `server.py`:

| Parameter          | Default | Description                        |
|--------------------|---------|------------------------------------|
| `num_clients`      | 5       | Number of federated clients        |
| `num_rounds`       | 10      | Communication rounds               |
| `local_epochs`     | 3       | Local training epochs per round    |
| `iid`              | True    | IID vs non-IID data split          |
| `use_dp`           | False   | Enable differential privacy        |
| `noise_multiplier` | 1.0     | DP noise scale (higher = more private) |

## Non-IID Data

In real federated settings, data is rarely identically distributed. The
non-IID split gives each client only 2 digit classes (e.g., Client 1 gets
only 0s and 1s), making convergence harder and more realistic.

## Example Output

```
=== Federated Learning Simulation ===
Clients: 5 | Rounds: 10 | IID: True

Round  1/10 - Accuracy: 85.23%
Round  2/10 - Accuracy: 90.47%
Round  3/10 - Accuracy: 93.12%
...
Round 10/10 - Accuracy: 97.81%
```

## Requirements

- Python 3.8+
- PyTorch 2.0+
- torchvision 0.15+
- matplotlib 3.5+

## References

- McMahan et al., "Communication-Efficient Learning of Deep Networks
  from Decentralized Data" (2017)
- Abadi et al., "Deep Learning with Differential Privacy" (2016)
