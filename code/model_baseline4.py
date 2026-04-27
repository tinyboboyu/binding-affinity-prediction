"""Baseline 4 model: Baseline 3 plus MD-ensemble representation distillation."""

from __future__ import annotations

import torch
from torch import nn

from model_baseline3 import SharedGINEncoder


class Baseline4PBModel(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.encoder = SharedGINEncoder(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
        )
        self.head_exp = nn.Linear(hidden_dim, 1)
        self.head_avg_pb = nn.Linear(hidden_dim, 6)
        self.head_frame_pb = nn.Linear(hidden_dim, 6)
        self.projector = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.register_buffer("exp_mean", torch.zeros(1))
        self.register_buffer("exp_std", torch.ones(1))
        self.register_buffer("avg_pb_mean", torch.zeros(6))
        self.register_buffer("avg_pb_std", torch.ones(6))
        self.register_buffer("frame_pb_mean", torch.zeros(6))
        self.register_buffer("frame_pb_std", torch.ones(6))
        self.register_buffer("normalization_enabled", torch.tensor(True, dtype=torch.bool))

    def set_normalization_stats(
        self,
        exp_mean: torch.Tensor,
        exp_std: torch.Tensor,
        avg_pb_mean: torch.Tensor,
        avg_pb_std: torch.Tensor,
        frame_pb_mean: torch.Tensor,
        frame_pb_std: torch.Tensor,
        enabled: bool,
    ) -> None:
        self.exp_mean.copy_(exp_mean.detach().view_as(self.exp_mean))
        self.exp_std.copy_(exp_std.detach().view_as(self.exp_std))
        self.avg_pb_mean.copy_(avg_pb_mean.detach().view_as(self.avg_pb_mean))
        self.avg_pb_std.copy_(avg_pb_std.detach().view_as(self.avg_pb_std))
        self.frame_pb_mean.copy_(frame_pb_mean.detach().view_as(self.frame_pb_mean))
        self.frame_pb_std.copy_(frame_pb_std.detach().view_as(self.frame_pb_std))
        self.normalization_enabled.fill_(enabled)

    def _normalize_prediction(self, tensor: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
        if not bool(self.normalization_enabled.item()):
            return tensor
        return (tensor - mean.to(tensor.device)) / std.to(tensor.device)

    def forward(
        self,
        crystal_batch,
        frame_batch=None,
        num_frames_per_sample: int = 5,
    ) -> dict[str, torch.Tensor | None]:
        h_crystal = self.encoder(crystal_batch)
        pred_exp = self.head_exp(h_crystal)
        pred_avg_pb = self.head_avg_pb(h_crystal)
        pred_exp_norm = self._normalize_prediction(pred_exp, self.exp_mean, self.exp_std)
        pred_avg_pb_norm = self._normalize_prediction(pred_avg_pb, self.avg_pb_mean, self.avg_pb_std)

        outputs: dict[str, torch.Tensor | None] = {
            "h_crystal": h_crystal,
            "h_frames": None,
            "h_teacher": None,
            "z_crystal": None,
            "pred_exp": pred_exp,
            "pred_exp_norm": pred_exp_norm,
            "pred_avg_pb": pred_avg_pb,
            "pred_avg_pb_norm": pred_avg_pb_norm,
            "pred_frame_pb": None,
            "pred_frame_pb_norm": None,
        }

        if frame_batch is None:
            return outputs

        frame_count = int(frame_batch.num_graphs)
        batch_size = int(h_crystal.size(0))
        expected_frames = batch_size * num_frames_per_sample
        if frame_count != expected_frames:
            raise ValueError(
                f"Expected {expected_frames} frame graphs for batch_size={batch_size} "
                f"and num_frames_per_sample={num_frames_per_sample}, got {frame_count}"
            )

        h_frame_flat = self.encoder(frame_batch)
        pred_frame_pb_flat = self.head_frame_pb(h_frame_flat)
        pred_frame_pb_norm_flat = self._normalize_prediction(
            pred_frame_pb_flat,
            self.frame_pb_mean,
            self.frame_pb_std,
        )

        h_frames = h_frame_flat.view(batch_size, num_frames_per_sample, self.hidden_dim)
        h_teacher = h_frames.mean(dim=1)
        z_crystal = self.projector(h_crystal)

        outputs.update(
            {
                "h_frames": h_frames,
                "h_teacher": h_teacher,
                "z_crystal": z_crystal,
                "pred_frame_pb": pred_frame_pb_flat.view(batch_size, num_frames_per_sample, 6),
                "pred_frame_pb_norm": pred_frame_pb_norm_flat.view(batch_size, num_frames_per_sample, 6),
            }
        )
        return outputs
