"""Command-line entrypoint for the v1 dataset preprocessing workflow."""

from __future__ import annotations

import argparse
from pathlib import Path

from .pipeline import ComplexPreprocessorConfig, preprocess_dataset


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Preprocess protein-ligand complexes into merged PyTorch Geometric graphs."
    )
    parser.add_argument(
        "--root-dir",
        default="../data/MMPBSA",
        help="Root directory containing sample subdirectories and the bd file.",
    )
    parser.add_argument(
        "--output-dir",
        default="../data/MMPBSA/processed",
        help="Output directory for graphs, metadata.csv, and failures.json.",
    )
    parser.add_argument("--ligand-resname", default="J5W", help="Default ligand residue name.")
    parser.add_argument("--ligand-resid", type=int, default=139, help="Default ligand residue id.")
    parser.add_argument("--ligand-chain", default=None, help="Optional ligand chain identifier.")
    parser.add_argument("--pocket-cutoff", type=float, default=5.0, help="Pocket residue cutoff in Angstrom.")
    parser.add_argument("--protein-edge-cutoff", type=float, default=4.5, help="Protein spatial edge cutoff.")
    parser.add_argument(
        "--ligand-protein-edge-cutoff",
        type=float,
        default=5.0,
        help="Ligand-protein spatial edge cutoff.",
    )
    parser.add_argument(
        "--sample-ids",
        nargs="*",
        default=None,
        help="Optional explicit sample ids. Defaults to 6QLN 6QLO 6QLP 6QLR 6QLT.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = ComplexPreprocessorConfig(
        ligand_resname=args.ligand_resname,
        ligand_resid=args.ligand_resid,
        ligand_chain=args.ligand_chain,
        pocket_cutoff=args.pocket_cutoff,
        protein_edge_cutoff=args.protein_edge_cutoff,
        ligand_protein_edge_cutoff=args.ligand_protein_edge_cutoff,
        keep_hydrogens=True,
        remove_water=True,
        keep_metals=True,
    )
    summary = preprocess_dataset(
        root_dir=Path(args.root_dir),
        output_dir=Path(args.output_dir),
        config=config,
        sample_ids=args.sample_ids,
    )
    print(
        f"Processed {summary['num_processed']} samples. "
        f"Failures: {summary['num_failed']}. "
        f"Outputs written under {args.output_dir}"
    )


if __name__ == "__main__":
    main()
