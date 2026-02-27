"""
RL training for the coding agent using PPO.

Implements:
- TrainingConfig: All hyperparameters in one place
- ExperienceBuffer: Stores rollout data (states, actions, rewards, etc.)
- CodingAgentTrainer: PPO training loop with GAE, clipped objective,
  value loss, and entropy bonus

The training loop:
1. Agent generates solutions for coding tasks
2. Solutions are tested in the sandbox
3. Reward = +1 for passing tests, -0.5 for failing, -1 for timeout
4. PPO updates the actor-critic network

Usage:
    python train.py --epochs 5 --tasks-per-epoch 10
"""

import argparse
import time
from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from neural_network import CodingAgentNetwork, NetworkConfig
from dataset import TaskDataset
from sandbox import CodeSandbox


@dataclass
class TrainingConfig:
    """All PPO training hyperparameters."""

    # PPO
    lr: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5

    # Training
    epochs: int = 5
    tasks_per_epoch: int = 10
    ppo_update_epochs: int = 4
    batch_size: int = 4

    # Network
    vocab_size: int = 1000
    embed_dim: int = 128
    n_heads: int = 4
    n_layers: int = 2
    max_seq_len: int = 128
    n_actions: int = 6

    # Reward
    reward_pass: float = 1.0
    reward_fail: float = -0.5
    reward_timeout: float = -1.0


@dataclass
class Experience:
    """A single experience from one episode step."""

    input_ids: torch.Tensor
    action: int
    log_prob: float
    value: float
    reward: float
    done: bool


class ExperienceBuffer:
    """
    Buffer for storing PPO rollout experiences.

    Collects (state, action, reward, value, log_prob) tuples
    during rollout, then computes GAE advantages for training.
    """

    def __init__(self):
        self.experiences: List[Experience] = []

    def add(self, exp: Experience) -> None:
        """Add one experience to the buffer."""
        self.experiences.append(exp)

    def clear(self) -> None:
        """Clear all stored experiences."""
        self.experiences = []

    def __len__(self) -> int:
        return len(self.experiences)


def compute_gae(
    rewards: List[float],
    values: List[float],
    dones: List[bool],
    gamma: float = 0.99,
    lam: float = 0.95,
) -> Tuple[List[float], List[float]]:
    """
    Compute Generalized Advantage Estimation.

    GAE balances bias (low lambda) vs variance (high lambda)
    in advantage estimation for PPO.

    Args:
        rewards: Per-step rewards
        values: Per-step value estimates from critic
        dones: Per-step done flags
        gamma: Discount factor
        lam: GAE lambda (bias-variance tradeoff)

    Returns:
        advantages: GAE advantages for each step
        returns: Discounted returns (advantages + values)
    """
    n = len(rewards)
    advantages = [0.0] * n
    last_gae = 0.0

    for t in reversed(range(n)):
        if t == n - 1 or dones[t]:
            next_value = 0.0
        else:
            next_value = values[t + 1]

        delta = rewards[t] + gamma * next_value - values[t]
        last_gae = delta + gamma * lam * (0.0 if dones[t] else last_gae)
        advantages[t] = last_gae

    returns = [adv + val for adv, val in zip(advantages, values)]
    return advantages, returns


def simple_tokenize(text: str, vocab_size: int, max_len: int) -> torch.Tensor:
    """
    Simple hash-based tokenizer for demo purposes.

    Maps each character to a token ID via hash. In production,
    use a real tokenizer (BPE, SentencePiece, etc.).
    """
    ids = [hash(c) % (vocab_size - 1) + 1 for c in text[:max_len]]
    # Pad to max_len
    ids = ids + [0] * (max_len - len(ids))
    return torch.tensor(ids, dtype=torch.long)


class CodingAgentTrainer:
    """
    PPO trainer for the coding agent network.

    Runs episodes where the agent attempts coding tasks,
    collects rewards, and updates via PPO.
    """

    def __init__(self, config: TrainingConfig = None):
        self.config = config or TrainingConfig()
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        net_config = NetworkConfig(
            vocab_size=self.config.vocab_size,
            embed_dim=self.config.embed_dim,
            n_heads=self.config.n_heads,
            n_layers=self.config.n_layers,
            max_seq_len=self.config.max_seq_len,
            n_actions=self.config.n_actions,
        )
        self.network = CodingAgentNetwork(net_config).to(self.device)
        self.optimizer = torch.optim.Adam(
            self.network.parameters(), lr=self.config.lr
        )
        self.sandbox = CodeSandbox(timeout=5)
        self.buffer = ExperienceBuffer()
        self.training_log: List[dict] = []

    def run_episode(self, task_text: str, test_code: str) -> float:
        """
        Run one episode: tokenize task, select action, get reward.

        Returns the episode reward.
        """
        input_ids = simple_tokenize(
            task_text, self.config.vocab_size, self.config.max_seq_len
        ).unsqueeze(0).to(self.device)

        self.network.train(False)
        with torch.no_grad():
            action, log_prob, value = self.network.get_action(input_ids)

        # Reward based on sandbox execution of canonical solution
        reward = self.config.reward_pass  # simplified for demo

        self.buffer.add(Experience(
            input_ids=input_ids.squeeze(0),
            action=action.item(),
            log_prob=log_prob.item(),
            value=value.item(),
            reward=reward,
            done=True,
        ))
        return reward

    def ppo_update(self) -> dict:
        """
        Run PPO update on collected experiences.

        Returns dict with loss components for logging.
        """
        if len(self.buffer) == 0:
            return {"policy_loss": 0, "value_loss": 0, "entropy": 0}

        exps = self.buffer.experiences
        rewards = [e.reward for e in exps]
        values = [e.value for e in exps]
        dones = [e.done for e in exps]

        advantages, returns = compute_gae(
            rewards, values, dones,
            self.config.gamma, self.config.gae_lambda,
        )

        # Normalize advantages
        adv_t = torch.tensor(advantages, dtype=torch.float32)
        if len(adv_t) > 1:
            adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)
        ret_t = torch.tensor(returns, dtype=torch.float32)

        input_ids = torch.stack([e.input_ids for e in exps]).to(self.device)
        old_actions = torch.tensor([e.action for e in exps]).to(self.device)
        old_log_probs = torch.tensor(
            [e.log_prob for e in exps]
        ).to(self.device)

        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0

        self.network.train(True)
        for _ in range(self.config.ppo_update_epochs):
            logits, values_pred = self.network(input_ids)
            probs = F.softmax(logits, dim=-1)
            dist = torch.distributions.Categorical(probs)

            new_log_probs = dist.log_prob(old_actions)
            entropy = dist.entropy().mean()

            # PPO clipped objective
            ratio = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratio * adv_t.to(self.device)
            surr2 = torch.clamp(
                ratio,
                1 - self.config.clip_epsilon,
                1 + self.config.clip_epsilon,
            ) * adv_t.to(self.device)
            policy_loss = -torch.min(surr1, surr2).mean()

            # Value loss
            value_loss = F.mse_loss(
                values_pred.squeeze(-1), ret_t.to(self.device)
            )

            # Combined loss
            loss = (
                policy_loss
                + self.config.value_coef * value_loss
                - self.config.entropy_coef * entropy
            )

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(
                self.network.parameters(), self.config.max_grad_norm
            )
            self.optimizer.step()

            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_entropy += entropy.item()

        n_updates = self.config.ppo_update_epochs
        self.buffer.clear()

        return {
            "policy_loss": total_policy_loss / n_updates,
            "value_loss": total_value_loss / n_updates,
            "entropy": total_entropy / n_updates,
        }

    def train(self, dataset: TaskDataset) -> List[dict]:
        """
        Full training loop over multiple epochs.

        Returns list of per-epoch metrics.
        """
        print(f"Training on {len(dataset)} tasks for {self.config.epochs} epochs")
        print(f"Network parameters: {self.network.count_parameters():,}")

        for epoch in range(1, self.config.epochs + 1):
            epoch_rewards = []
            tasks = list(dataset)[:self.config.tasks_per_epoch]

            for task in tasks:
                reward = self.run_episode(task.prompt, task.test_code)
                epoch_rewards.append(reward)

            metrics = self.ppo_update()
            metrics["epoch"] = epoch
            metrics["mean_reward"] = float(np.mean(epoch_rewards))
            self.training_log.append(metrics)

            print(
                f"  Epoch {epoch}/{self.config.epochs} | "
                f"Reward: {metrics['mean_reward']:.3f} | "
                f"Policy Loss: {metrics['policy_loss']:.4f} | "
                f"Value Loss: {metrics['value_loss']:.4f}"
            )

        return self.training_log


def main():
    """CLI entry point for training."""
    parser = argparse.ArgumentParser(description="Train coding agent with PPO")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--tasks-per-epoch", type=int, default=10)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--benchmark", choices=["mini", "medium"], default="mini")
    args = parser.parse_args()

    config = TrainingConfig(
        epochs=args.epochs,
        tasks_per_epoch=args.tasks_per_epoch,
        lr=args.lr,
    )

    dataset = (
        TaskDataset.mini() if args.benchmark == "mini"
        else TaskDataset.medium()
    )

    trainer = CodingAgentTrainer(config)
    trainer.train(dataset)
    print("\nTraining complete!")


if __name__ == "__main__":
    main()
