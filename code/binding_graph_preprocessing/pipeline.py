"""Dataset-level preprocessing pipeline for the confirmed v1 specification."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

from .constants import DEFAULT_VALID_SAMPLE_IDS, SKIP_SAMPLE_IDS
from .graph import build_complex_graph, save_graph
from .labels import parse_experimental_bd_table, parse_gb_aux_labels
from .structure import build_graph_components, parse_pdb_file


@dataclass
class ComplexPreprocessorConfig:
    ligand_resname: str = "J5W"
    ligand_resid: int | None = 139
    ligand_chain: str | None = None
    pocket_cutoff: float = 5.0
    protein_edge_cutoff: float = 4.5
    ligand_protein_edge_cutoff: float = 5.0
    keep_hydrogens: bool = True
    remove_water: bool = True
    keep_metals: bool = True


class ComplexGraphPreprocessor:
    """Preprocess one sample directory into one merged PyG-ready graph."""

    def __init__(self, config: ComplexPreprocessorConfig | None = None) -> None:
        self.config = config or ComplexPreprocessorConfig()

    def process_sample(
        self,
        sample_id: str,
        sample_dir: str | Path,
        y_exp: float,
    ) -> dict[str, object]:
        sample_path = Path(sample_dir)
        pdb_path = sample_path / "complex.pdb"
        mmpbsa_path = sample_path / "mmpbsa.out"

        parsed = parse_pdb_file(pdb_path)
        components = build_graph_components(
            parsed=parsed,
            ligand_resname=self.config.ligand_resname,
            ligand_resid=self.config.ligand_resid,
            ligand_chain=self.config.ligand_chain,
            pocket_cutoff=self.config.pocket_cutoff,
            protein_edge_cutoff=self.config.protein_edge_cutoff,
            keep_hydrogens=self.config.keep_hydrogens,
            remove_water=self.config.remove_water,
            keep_metals=self.config.keep_metals,
        )
        aux = parse_gb_aux_labels(mmpbsa_path)
        return build_complex_graph(
            components=components,
            sample_id=sample_id,
            pdb_path=str(pdb_path),
            mmpbsa_path=str(mmpbsa_path),
            y_exp=y_exp,
            y_vdw=aux["y_vdw"],
            y_elec=aux["y_elec"],
            y_polar=aux["y_polar"],
            y_nonpolar=aux["y_nonpolar"],
            delta_total_kcal=aux["delta_total_kcal"],
            protein_edge_cutoff=self.config.protein_edge_cutoff,
            ligand_protein_edge_cutoff=self.config.ligand_protein_edge_cutoff,
        )


def preprocess_dataset(
    root_dir: str | Path,
    output_dir: str | Path,
    config: ComplexPreprocessorConfig | None = None,
    sample_ids: list[str] | None = None,
) -> dict[str, object]:
    """Scan the dataset root, build graphs, and save metadata plus failures."""
    root = Path(root_dir)
    if not root.exists():
        raise FileNotFoundError(f"Dataset root not found: {root}")

    output_root = Path(output_dir)
    graphs_dir = output_root / "graphs"
    graphs_dir.mkdir(parents=True, exist_ok=True)

    preprocessor = ComplexGraphPreprocessor(config=config)
    experimental_values = parse_experimental_bd_table(root / "bd")
    requested_ids = sample_ids or DEFAULT_VALID_SAMPLE_IDS
    selected_ids = [sample_id for sample_id in requested_ids if sample_id not in SKIP_SAMPLE_IDS]

    metadata_rows: list[dict[str, object]] = []
    failures: list[dict[str, str]] = []

    for sample_id in selected_ids:
        sample_dir = root / sample_id
        try:
            if not sample_dir.is_dir():
                raise FileNotFoundError(f"Sample directory not found: {sample_dir}")
            if sample_id not in experimental_values:
                raise KeyError(f"Experimental label missing for {sample_id}")

            graph = preprocessor.process_sample(
                sample_id=sample_id,
                sample_dir=sample_dir,
                y_exp=experimental_values[sample_id],
            )
            output_path = graphs_dir / f"{sample_id}.pt"
            save_graph(graph, str(output_path))
            metadata_rows.append(build_metadata_row(graph, output_path))
        except Exception as exc:  # noqa: BLE001
            failures.append({"sample_id": sample_id, "error": str(exc)})

    write_metadata_csv(output_root / "metadata.csv", metadata_rows)
    write_failures_json(output_root / "failures.json", failures)

    return {
        "graphs_dir": str(graphs_dir),
        "metadata_csv": str(output_root / "metadata.csv"),
        "failures_json": str(output_root / "failures.json"),
        "num_processed": len(metadata_rows),
        "num_failed": len(failures),
    }


def build_metadata_row(graph: dict[str, object], output_path: Path) -> dict[str, object]:
    return {
        "sample_id": graph["sample_id"],
        "pdb_id": graph["pdb_id"],
        "graph_path": str(output_path),
        "pdb_path": graph["pdb_path"],
        "mmpbsa_path": graph["mmpbsa_path"],
        "ligand_resname": graph["ligand_resname"],
        "ligand_resid": graph["ligand_resid"],
        "ligand_chain": graph["ligand_chain"],
        "num_nodes": len(graph["x"]),
        "num_edges": len(graph["edge_attr"]),
        "num_ligand_atoms": int(sum(graph["ligand_mask"])),
        "num_protein_atoms": int(sum(graph["protein_mask"])),
        "num_metal_atoms": int(sum(graph["metal_mask"])),
        "y_exp": graph["y_exp"][0],
        "y_vdw": graph["y_vdw"][0],
        "y_elec": graph["y_elec"][0],
        "y_polar": graph["y_polar"][0],
        "y_nonpolar": graph["y_nonpolar"][0],
        "delta_total_kcal": graph["delta_total_kcal"][0],
        "energy_unit": graph["energy_unit"],
    }


def write_metadata_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = [
        "sample_id",
        "pdb_id",
        "graph_path",
        "pdb_path",
        "mmpbsa_path",
        "ligand_resname",
        "ligand_resid",
        "ligand_chain",
        "num_nodes",
        "num_edges",
        "num_ligand_atoms",
        "num_protein_atoms",
        "num_metal_atoms",
        "y_exp",
        "y_vdw",
        "y_elec",
        "y_polar",
        "y_nonpolar",
        "delta_total_kcal",
        "energy_unit",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_failures_json(path: Path, failures: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(failures, handle, indent=2)
