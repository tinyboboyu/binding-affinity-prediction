"""Minimal multi-task GNN baseline for protein-ligand complex graph regression."""

from __future__ import annotations

import torch
from torch import nn
from torch_geometric.nn import GINConv, global_mean_pool


def build_gin_mlp(hidden_dim: int) -> nn.Sequential:
    """Construct the MLP used inside each GIN layer."""
    return nn.Sequential(
        nn.Linear(hidden_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim),
    )


class MultiTaskComplexGNN(nn.Module):
    """Shared GIN encoder with separate heads for experimental and auxiliary targets."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {num_layers}")

        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout

        self.input_proj = nn.Linear(in_dim, hidden_dim)
        self.convs = nn.ModuleList([GINConv(build_gin_mlp(hidden_dim)) for _ in range(num_layers)])
        self.activation = nn.ReLU()
        self.dropout_layer = nn.Dropout(dropout)

        self.head_exp = nn.Linear(hidden_dim, 1)
        self.head_aux = nn.Linear(hidden_dim, 4)

    def forward(self, data) -> dict[str, torch.Tensor]:
        """Encode the graph with node features and connectivity only."""
        x = data.x.float()
        edge_index = data.edge_index
        batch = data.batch

        h = self.input_proj(x)
        h = self.activation(h)

        for conv in self.convs:
            h = conv(h, edge_index)
            h = self.activation(h)
            h = self.dropout_layer(h)

        h_graph = global_mean_pool(h, batch)
        pred_exp = self.head_exp(h_graph)
        pred_aux = self.head_aux(h_graph)

        return {
            "graph_embedding": h_graph,
            "pred_exp": pred_exp,
            "pred_aux": pred_aux,
        }
