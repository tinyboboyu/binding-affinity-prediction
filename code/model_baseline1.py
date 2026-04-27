"""Baseline 1 model: crystal-only experimental binding free energy prediction."""

from __future__ import annotations

import torch
from torch import nn

from model_baseline3 import SharedGINEncoder


class Baseline1ExpModel(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 64, num_layers: int = 2, dropout: float = 0.0) -> None:
        super().__init__()
        self.encoder = SharedGINEncoder(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
        )
        self.head_exp = nn.Linear(hidden_dim, 1)

    def forward(self, crystal_batch) -> dict[str, torch.Tensor]:
        h_crystal = self.encoder(crystal_batch)
        pred_exp = self.head_exp(h_crystal)
        return {
            "h_crystal": h_crystal,
            "pred_exp": pred_exp,
        }
