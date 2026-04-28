"""Baseline 3 model: crystal prediction with MD-frame auxiliary supervision."""

from __future__ import annotations

import torch
from torch import nn
from torch_geometric.nn import GINConv, global_mean_pool


def build_gin_mlp(hidden_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(hidden_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim),
    )


class SharedGINEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 64, num_layers: int = 2, dropout: float = 0.0) -> None:
        super().__init__()
        self.input_proj = nn.Linear(in_dim, hidden_dim)
        self.convs = nn.ModuleList([GINConv(build_gin_mlp(hidden_dim)) for _ in range(num_layers)])
        self.activation = nn.ReLU()
        self.dropout_layer = nn.Dropout(dropout)

    def forward(self, data) -> torch.Tensor:
        x = data.x.float()
        edge_index = data.edge_index
        batch = data.batch

        h = self.activation(self.input_proj(x))
        for conv in self.convs:
            h = self.activation(conv(h, edge_index))
            h = self.dropout_layer(h)
        return global_mean_pool(h, batch)


class Baseline3PBModel(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 64, num_layers: int = 2, dropout: float = 0.0) -> None:
        super().__init__()
        self.encoder = SharedGINEncoder(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
        )
        self.head_exp = nn.Linear(hidden_dim, 1)
        self.head_avg_pb = nn.Linear(hidden_dim, 6)
        self.head_frame_pb = nn.Linear(hidden_dim, 6)

    def forward(self, crystal_batch, frame_batch=None) -> dict[str, torch.Tensor | None]:
        h_crystal = self.encoder(  )
        pred_exp = self.head_exp(h_crystal)
        pred_avg_pb = self.head_avg_pb(h_crystal)

        pred_frame_pb = None
        h_frame = None
        if frame_batch is not None:
            h_frame = self.encoder(frame_batch)
            pred_frame_pb = self.head_frame_pb(h_frame)

        return {
            "h_crystal": h_crystal,
            "h_frame": h_frame,
            "pred_exp": pred_exp,
            "pred_avg_pb": pred_avg_pb,
            "pred_frame_pb": pred_frame_pb,
        }

