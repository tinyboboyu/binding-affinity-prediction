"""Label normalization utilities for Baseline 3."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch


def _safe_std(tensor: torch.Tensor) -> torch.Tensor:
    std = tensor.std(dim=0, unbiased=False)
    std = torch.where(std == 0, torch.ones_like(std), std)
    return std


@dataclass
class ExpLabelNormalizer:
    enabled: bool
    exp_mean: torch.Tensor
    exp_std: torch.Tensor

    @classmethod
    def from_training_graphs(cls, training_graphs: list[object], enabled: bool = True) -> "ExpLabelNormalizer":
        exp = torch.tensor(
            [[float(graph.y_exp.view(-1)[0].item())] for graph in training_graphs],
            dtype=torch.float32,
        )
        return cls(
            enabled=enabled,
            exp_mean=exp.mean(dim=0),
            exp_std=_safe_std(exp),
        )

    def normalize_exp(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor if not self.enabled else (tensor - self.exp_mean.to(tensor.device)) / self.exp_std.to(tensor.device)

    def denormalize_exp(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor if not self.enabled else tensor * self.exp_std.to(tensor.device) + self.exp_mean.to(tensor.device)

    def save(self, path: str | Path) -> None:
        payload = {
            "enabled": self.enabled,
            "exp_mean": self.exp_mean.tolist(),
            "exp_std": self.exp_std.tolist(),
        }
        Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")


@dataclass
class ExpAvgPBLabelNormalizer:
    enabled: bool
    exp_mean: torch.Tensor
    exp_std: torch.Tensor
    avg_pb_mean: torch.Tensor
    avg_pb_std: torch.Tensor

    @classmethod
    def from_training_records(cls, training_records: list[object], enabled: bool = True) -> "ExpAvgPBLabelNormalizer":
        exp = torch.tensor([[float(getattr(record, "y_exp"))] for record in training_records], dtype=torch.float32)
        avg_pb = torch.stack([getattr(record, "y_avg_pb") for record in training_records]).float()
        return cls(
            enabled=enabled,
            exp_mean=exp.mean(dim=0),
            exp_std=_safe_std(exp),
            avg_pb_mean=avg_pb.mean(dim=0),
            avg_pb_std=_safe_std(avg_pb),
        )

    def normalize_exp(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor if not self.enabled else (tensor - self.exp_mean.to(tensor.device)) / self.exp_std.to(tensor.device)

    def normalize_avg_pb(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor if not self.enabled else (tensor - self.avg_pb_mean.to(tensor.device)) / self.avg_pb_std.to(tensor.device)

    def denormalize_exp(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor if not self.enabled else tensor * self.exp_std.to(tensor.device) + self.exp_mean.to(tensor.device)

    def denormalize_avg_pb(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor if not self.enabled else tensor * self.avg_pb_std.to(tensor.device) + self.avg_pb_mean.to(tensor.device)

    def save(self, path: str | Path) -> None:
        payload = {
            "enabled": self.enabled,
            "exp_mean": self.exp_mean.tolist(),
            "exp_std": self.exp_std.tolist(),
            "avg_pb_mean": self.avg_pb_mean.tolist(),
            "avg_pb_std": self.avg_pb_std.tolist(),
        }
        Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")


@dataclass
class LabelNormalizer:
    enabled: bool
    exp_mean: torch.Tensor
    exp_std: torch.Tensor
    avg_pb_mean: torch.Tensor
    avg_pb_std: torch.Tensor
    frame_pb_mean: torch.Tensor
    frame_pb_std: torch.Tensor

    @classmethod
    def from_training_records(cls, training_records: list[dict[str, object]], enabled: bool = True) -> "LabelNormalizer":
        exp = torch.tensor([[float(getattr(record, "y_exp"))] for record in training_records], dtype=torch.float32)
        avg_pb = torch.stack([getattr(record, "y_avg_pb") for record in training_records]).float()
        frame_pb = torch.cat([getattr(record, "y_frame_pb") for record in training_records], dim=0).float()
        return cls(
            enabled=enabled,
            exp_mean=exp.mean(dim=0),
            exp_std=_safe_std(exp),
            avg_pb_mean=avg_pb.mean(dim=0),
            avg_pb_std=_safe_std(avg_pb),
            frame_pb_mean=frame_pb.mean(dim=0),
            frame_pb_std=_safe_std(frame_pb),
        )

    def normalize_exp(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor if not self.enabled else (tensor - self.exp_mean.to(tensor.device)) / self.exp_std.to(tensor.device)

    def normalize_avg_pb(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor if not self.enabled else (tensor - self.avg_pb_mean.to(tensor.device)) / self.avg_pb_std.to(tensor.device)

    def normalize_frame_pb(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor if not self.enabled else (tensor - self.frame_pb_mean.to(tensor.device)) / self.frame_pb_std.to(tensor.device)

    def denormalize_exp(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor if not self.enabled else tensor * self.exp_std.to(tensor.device) + self.exp_mean.to(tensor.device)

    def denormalize_avg_pb(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor if not self.enabled else tensor * self.avg_pb_std.to(tensor.device) + self.avg_pb_mean.to(tensor.device)

    def save(self, path: str | Path) -> None:
        payload = {
            "enabled": self.enabled,
            "exp_mean": self.exp_mean.tolist(),
            "exp_std": self.exp_std.tolist(),
            "avg_pb_mean": self.avg_pb_mean.tolist(),
            "avg_pb_std": self.avg_pb_std.tolist(),
            "frame_pb_mean": self.frame_pb_mean.tolist(),
            "frame_pb_std": self.frame_pb_std.tolist(),
        }
        Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")
