"""Validate that ligand nodes in saved .pt graphs match the source complex.pdb files."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from binding_graph_preprocessing.constants import DEFAULT_VALID_SAMPLE_IDS
from binding_graph_preprocessing.pipeline import ComplexPreprocessorConfig
from binding_graph_preprocessing.structure import parse_pdb_file, select_ligand_atoms


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate ligand content in processed PyG graphs.")
    parser.add_argument(
        "--root-dir",
        default="../data/MMPBSA",
        help="Dataset root containing sample directories.",
    )
    parser.add_argument(
        "--processed-dir",
        default="../data/MMPBSA/processed/graphs",
        help="Directory containing saved .pt graph files.",
    )
    parser.add_argument("--ligand-resname", default="J5W", help="Expected ligand residue name.")
    parser.add_argument("--ligand-resid", type=int, default=139, help="Expected ligand residue id.")
    parser.add_argument("--ligand-chain", default=None, help="Optional ligand chain identifier.")
    parser.add_argument(
        "--sample-ids",
        nargs="*",
        default=DEFAULT_VALID_SAMPLE_IDS,
        help="Sample ids to validate.",
    )
    parser.add_argument(
        "--coord-tol",
        type=float,
        default=1e-3,
        help="Absolute coordinate tolerance in Angstrom for ligand node matching.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    root_dir = Path(args.root_dir)
    processed_dir = Path(args.processed_dir)
    config = ComplexPreprocessorConfig(
        ligand_resname=args.ligand_resname,
        ligand_resid=args.ligand_resid,
        ligand_chain=args.ligand_chain,
    )

    all_ok = True
    for sample_id in args.sample_ids:
        pdb_path = root_dir / sample_id / "complex.pdb"
        graph_path = processed_dir / f"{sample_id}.pt"
        result = validate_single_sample(
            sample_id=sample_id,
            pdb_path=pdb_path,
            graph_path=graph_path,
            config=config,
            coord_tol=args.coord_tol,
        )
        print(result)
        if not result.startswith("[OK]"):
            all_ok = False

    if not all_ok:
        raise SystemExit(1)


def validate_single_sample(
    sample_id: str,
    pdb_path: Path,
    graph_path: Path,
    config: ComplexPreprocessorConfig,
    coord_tol: float,
) -> str:
    if not pdb_path.exists():
        return f"[FAIL] {sample_id}: missing PDB file {pdb_path}"
    if not graph_path.exists():
        return f"[FAIL] {sample_id}: missing graph file {graph_path}"

    parsed = parse_pdb_file(pdb_path)
    ligand_atoms = select_ligand_atoms(
        atoms=parsed.atoms,
        ligand_resname=config.ligand_resname,
        ligand_resid=config.ligand_resid,
        ligand_chain=config.ligand_chain,
    )
    data = torch.load(graph_path, weights_only=False)

    ligand_indices = data.ligand_mask.nonzero(as_tuple=False).view(-1).tolist()
    ligand_pos = data.pos[ligand_indices]
    ligand_node_type = data.node_type[ligand_indices].tolist()

    if len(ligand_atoms) != len(ligand_indices):
        return (
            f"[FAIL] {sample_id}: ligand atom count mismatch, "
            f"pdb={len(ligand_atoms)} pt={len(ligand_indices)}"
        )
    if any(node_type != 0 for node_type in ligand_node_type):
        return f"[FAIL] {sample_id}: some ligand-masked nodes do not have node_type=0"

    pdb_elements = [atom.element for atom in ligand_atoms]
    pt_atomic_numbers = data.x[ligand_indices, 0].tolist()
    pt_elements_ok = all(int(atomic_number) > 0 for atomic_number in pt_atomic_numbers)
    if not pt_elements_ok:
        return f"[FAIL] {sample_id}: invalid atomic number found in ligand nodes"

    for atom, pos_tensor in zip(ligand_atoms, ligand_pos):
        pos = pos_tensor.tolist()
        if not coords_close(atom.coord, pos, coord_tol):
            return (
                f"[FAIL] {sample_id}: ligand coordinate mismatch for atom {atom.atom_name} "
                f"pdb={atom.coord} pt={tuple(round(v, 4) for v in pos)}"
            )

    pdb_resname = ligand_atoms[0].residue_id.residue_name if ligand_atoms else config.ligand_resname
    pdb_resid = ligand_atoms[0].residue_id.residue_number if ligand_atoms else config.ligand_resid
    if str(data.ligand_resname) != str(pdb_resname):
        return f"[FAIL] {sample_id}: ligand_resname mismatch, pdb={pdb_resname} pt={data.ligand_resname}"
    if int(data.ligand_resid) != int(pdb_resid):
        return f"[FAIL] {sample_id}: ligand_resid mismatch, pdb={pdb_resid} pt={data.ligand_resid}"

    return (
        f"[OK] {sample_id}: ligand nodes={len(ligand_indices)}, "
        f"resname={data.ligand_resname}, resid={data.ligand_resid}, "
        f"chemistry={data.metadata.get('ligand_chemistry_source', 'unknown')}, "
        f"elements={','.join(pdb_elements[:6])}{'...' if len(pdb_elements) > 6 else ''}"
    )


def coords_close(coord_a: tuple[float, float, float], coord_b: list[float], tol: float) -> bool:
    return all(abs(left - right) <= tol for left, right in zip(coord_a, coord_b))


if __name__ == "__main__":
    main()
