"""Batch inspection utility for processed PyG graph files."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch

from binding_graph_preprocessing.constants import DEFAULT_VALID_SAMPLE_IDS

VALID_SAMPLE_IDS = list(DEFAULT_VALID_SAMPLE_IDS)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inspect all processed graph files and save a CSV summary.")
    parser.add_argument(
        "--graph_dir",
        default="../data/MMPBSA/processed/graphs",
        help="Directory containing .pt graph files.",
    )
    parser.add_argument("--save_csv", default="graph_summary.csv", help="Path to save the summary CSV.")
    parser.add_argument("--sample_ids", nargs="*", default=VALID_SAMPLE_IDS, help="Sample IDs to inspect.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    rows = []

    for sample_id in args.sample_ids:
        if sample_id not in VALID_SAMPLE_IDS:
            raise ValueError(f"Invalid sample ID: {sample_id}. Valid sample IDs are {VALID_SAMPLE_IDS}")

        graph_path = Path(args.graph_dir) / f"{sample_id}.pt"
        if not graph_path.exists():
            raise FileNotFoundError(f"Missing graph file for sample {sample_id}: {graph_path}")

        data = torch.load(graph_path, weights_only=False)
        edge_types = data.edge_type.tolist() if hasattr(data, "edge_type") else []

        row = {
            "sample_id": sample_id,
            "num_nodes": int(data.x.shape[0]),
            "num_edges": int(data.edge_index.shape[1]),
            "num_ligand_nodes": int(data.ligand_mask.sum()),
            "num_protein_nodes": int(data.protein_mask.sum()),
            "num_metal_nodes": int(data.metal_mask.sum()),
            "num_ligand_covalent_edges": sum(1 for value in edge_types if value == 0),
            "num_protein_spatial_edges": sum(1 for value in edge_types if value == 1),
            "num_ligand_protein_edges": sum(1 for value in edge_types if value == 2),
            "y_exp": float(data.y_exp[0]),
            "y_vdw": float(data.y_vdw[0]),
            "y_elec": float(data.y_elec[0]),
            "y_polar": float(data.y_polar[0]),
            "y_nonpolar": float(data.y_nonpolar[0]),
            "delta_total_kcal": float(data.delta_total_kcal[0]),
            "has_nan_x": bool(torch.isnan(data.x).any().item()),
            "has_nan_pos": bool(torch.isnan(data.pos).any().item()),
            "has_nan_edge_attr": bool(torch.isnan(data.edge_attr).any().item()),
        }
        rows.append(row)
        print(
            f"{sample_id}: nodes={row['num_nodes']}, edges={row['num_edges']}, "
            f"ligand={row['num_ligand_nodes']}, protein={row['num_protein_nodes']}, metal={row['num_metal_nodes']}"
        )

    save_summary(rows, args.save_csv)
    print(f"Saved graph summary to {args.save_csv}")


def save_summary(rows: list[dict[str, object]], csv_path: str | Path) -> None:
    fieldnames = [
        "sample_id",
        "num_nodes",
        "num_edges",
        "num_ligand_nodes",
        "num_protein_nodes",
        "num_metal_nodes",
        "num_ligand_covalent_edges",
        "num_protein_spatial_edges",
        "num_ligand_protein_edges",
        "y_exp",
        "y_vdw",
        "y_elec",
        "y_polar",
        "y_nonpolar",
        "delta_total_kcal",
        "has_nan_x",
        "has_nan_pos",
        "has_nan_edge_attr",
    ]
    with Path(csv_path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


if __name__ == "__main__":
    main()
