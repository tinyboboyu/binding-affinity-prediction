"""Dataset utilities for loading processed protein-ligand graph samples."""

from __future__ import annotations

from pathlib import Path

import torch

from binding_graph_preprocessing.constants import DEFAULT_VALID_SAMPLE_IDS

VALID_SAMPLE_IDS = list(DEFAULT_VALID_SAMPLE_IDS)


def load_graph(graph_path: str | Path):
    """Load a saved PyG graph file with PyTorch 2.6-compatible defaults."""
    path = Path(graph_path)
    if not path.exists():
        raise FileNotFoundError(f"Graph file not found: {path}")
    return torch.load(path, weights_only=False)


class MMPBSAGraphDataset:
    """A lightweight dataset wrapper around saved PyTorch Geometric Data files."""

    def __init__(self, graph_dir: str | Path, sample_ids: list[str] | None = None) -> None:
        self.graph_dir = Path(graph_dir)
        if not self.graph_dir.exists():
            raise FileNotFoundError(f"Graph directory not found: {self.graph_dir}")

        if sample_ids is None:
            resolved_ids = [
                sample_id
                for sample_id in VALID_SAMPLE_IDS
                if (self.graph_dir / f"{sample_id}.pt").exists()
            ]
        else:
            resolved_ids = self._validate_requested_sample_ids(sample_ids)

        if not resolved_ids:
            raise ValueError(f"No valid graph files found in {self.graph_dir}")

        self.sample_ids = resolved_ids
        self.graph_paths = [self.graph_dir / f"{sample_id}.pt" for sample_id in self.sample_ids]

        for sample_id, graph_path in zip(self.sample_ids, self.graph_paths):
            if not graph_path.exists():
                raise FileNotFoundError(f"Requested sample {sample_id} is missing graph file: {graph_path}")

    def __len__(self) -> int:
        return len(self.graph_paths)

    def __getitem__(self, index: int):
        return load_graph(self.graph_paths[index])

    @staticmethod
    def _validate_requested_sample_ids(sample_ids: list[str]) -> list[str]:
        invalid_ids = [sample_id for sample_id in sample_ids if sample_id not in VALID_SAMPLE_IDS]
        if invalid_ids:
            raise ValueError(
                f"Invalid sample IDs requested: {invalid_ids}. "
                f"Valid sample IDs are: {VALID_SAMPLE_IDS}"
            )
        return list(sample_ids)
