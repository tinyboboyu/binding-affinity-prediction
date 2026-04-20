"""Inspect a saved PyTorch Geometric graph sample."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inspect a saved protein-ligand graph .pt file.")
    parser.add_argument("graph_path", help="Path to a saved .pt graph file.")
    return parser


def tensor_shape(value) -> str:
    if hasattr(value, "shape"):
        return str(tuple(value.shape))
    return "-"


def main() -> None:
    args = build_parser().parse_args()
    graph_path = Path(args.graph_path)
    if not graph_path.exists():
        raise FileNotFoundError(f"Graph file not found: {graph_path}")

    data = torch.load(graph_path, weights_only=False)

    print(f"Graph file: {graph_path}")
    print(f"Type: {type(data).__name__}")
    print()

    print("Core tensors")
    print(f"  x: {tensor_shape(data.x)}")
    print(f"  pos: {tensor_shape(data.pos)}")
    print(f"  edge_index: {tensor_shape(data.edge_index)}")
    print(f"  edge_attr: {tensor_shape(data.edge_attr)}")
    print()

    print("Node counts")
    print(f"  total nodes: {data.x.shape[0]}")
    print(f"  ligand nodes: {int(data.ligand_mask.sum())}")
    print(f"  protein nodes: {int(data.protein_mask.sum())}")
    print(f"  metal nodes: {int(data.metal_mask.sum())}")
    print()

    print("Edge counts")
    print(f"  total directed edges: {data.edge_index.shape[1]}")
    if hasattr(data, "edge_type"):
        edge_types = data.edge_type.tolist()
        print(f"  ligand_covalent edges: {sum(1 for value in edge_types if value == 0)}")
        print(f"  protein_spatial edges: {sum(1 for value in edge_types if value == 1)}")
        print(f"  ligand_protein_spatial edges: {sum(1 for value in edge_types if value == 2)}")
    print()

    print("Labels")
    print(f"  y_exp: {float(data.y_exp[0]):.4f} {data.energy_unit}")
    print(f"  y_vdw: {float(data.y_vdw[0]):.4f}")
    print(f"  y_elec: {float(data.y_elec[0]):.4f}")
    print(f"  y_polar: {float(data.y_polar[0]):.4f}")
    print(f"  y_nonpolar: {float(data.y_nonpolar[0]):.4f}")
    print(f"  delta_total_kcal: {float(data.delta_total_kcal[0]):.4f}")
    print()

    print("Metadata")
    print(f"  sample_id: {data.sample_id}")
    print(f"  pdb_id: {data.pdb_id}")
    print(f"  pdb_path: {data.pdb_path}")
    print(f"  mmpbsa_path: {data.mmpbsa_path}")
    print(f"  ligand_resname: {data.ligand_resname}")
    print(f"  ligand_resid: {data.ligand_resid}")
    print(f"  ligand_chain: {data.ligand_chain!r}")
    chemistry_source = data.metadata.get("ligand_chemistry_source", "unknown")
    print(f"  ligand_chemistry_source: {chemistry_source}")
    print()

    print("Feature names")
    print(f"  node features: {data.metadata.get('node_feature_names', [])}")
    print(f"  edge features: {data.metadata.get('edge_feature_names', [])}")


if __name__ == "__main__":
    main()
