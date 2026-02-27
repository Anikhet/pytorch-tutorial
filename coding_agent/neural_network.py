"""
Neural network components for the coding agent.

Architecture:
- CodeEncoder: Small transformer that encodes code/text into embeddings.
- PolicyHead: Maps embeddings to action (tool selection) logits.
- ValueHead: Maps embeddings to a scalar state value for PPO.
- CodingAgentNetwork: Actor-critic network combining all three.

The network processes tokenized code/text, produces tool-selection
probabilities (policy) and a value estimate (critic) for RL training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass
from typing import Tuple


@dataclass
class NetworkConfig:
    """Configuration for the coding agent network."""

    vocab_size: int = 1000
    embed_dim: int = 128
    n_heads: int = 4
    n_layers: int = 2
    max_seq_len: int = 512
    n_actions: int = 6  # number of tools
    dropout: float = 0.1


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for transformer inputs."""

    def __init__(self, embed_dim: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2).float()
            * (-math.log(10000.0) / embed_dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input embeddings."""
        return x + self.pe[:, : x.size(1)]


class CodeEncoder(nn.Module):
    """
    Small transformer encoder for code/text sequences.

    Takes token IDs, embeds them, adds positional encoding,
    and processes through transformer encoder layers.
    Returns the [CLS]-style pooled representation (first token).
    """

    def __init__(self, config: NetworkConfig):
        super().__init__()
        self.embedding = nn.Embedding(config.vocab_size, config.embed_dim)
        self.pos_encoding = PositionalEncoding(
            config.embed_dim, config.max_seq_len
        )
        self.dropout = nn.Dropout(config.dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.embed_dim,
            nhead=config.n_heads,
            dim_feedforward=config.embed_dim * 4,
            dropout=config.dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=config.n_layers
        )
        self.layer_norm = nn.LayerNorm(config.embed_dim)

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Encode token sequence into a fixed-size representation.

        Args:
            input_ids: (batch, seq_len) token IDs
            attention_mask: (batch, seq_len) 1=attend, 0=pad

        Returns:
            (batch, embed_dim) pooled representation
        """
        x = self.embedding(input_ids)
        x = self.pos_encoding(x)
        x = self.dropout(x)

        # Convert attention_mask to transformer format (True=ignore)
        src_key_padding_mask = None
        if attention_mask is not None:
            src_key_padding_mask = attention_mask == 0

        x = self.transformer(x, src_key_padding_mask=src_key_padding_mask)
        x = self.layer_norm(x)

        # Pool: use first token as sequence representation
        return x[:, 0, :]


class PolicyHead(nn.Module):
    """Maps encoded representation to action logits for tool selection."""

    def __init__(self, embed_dim: int, n_actions: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, n_actions),
        )

    def forward(self, encoding: torch.Tensor) -> torch.Tensor:
        """Return action logits (batch, n_actions)."""
        return self.net(encoding)


class ValueHead(nn.Module):
    """Maps encoded representation to a scalar state value."""

    def __init__(self, embed_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 1),
        )

    def forward(self, encoding: torch.Tensor) -> torch.Tensor:
        """Return state value (batch, 1)."""
        return self.net(encoding)


class CodingAgentNetwork(nn.Module):
    """
    Actor-critic network for the coding agent.

    Combines CodeEncoder + PolicyHead (actor) + ValueHead (critic).
    Used by PPOTrainer for reinforcement learning.
    """

    def __init__(self, config: NetworkConfig = None):
        super().__init__()
        self.config = config or NetworkConfig()
        self.encoder = CodeEncoder(self.config)
        self.policy = PolicyHead(self.config.embed_dim, self.config.n_actions)
        self.value = ValueHead(self.config.embed_dim)

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass returning both policy logits and value estimate.

        Returns:
            action_logits: (batch, n_actions)
            value: (batch, 1)
        """
        encoding = self.encoder(input_ids, attention_mask)
        action_logits = self.policy(encoding)
        value = self.value(encoding)
        return action_logits, value

    def get_action(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample an action from the policy and return log-prob + value.

        Returns:
            action: (batch,) sampled action indices
            log_prob: (batch,) log probability of sampled actions
            value: (batch,) state value estimates
        """
        logits, value = self.forward(input_ids, attention_mask)
        probs = F.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob, value.squeeze(-1)

    def count_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
