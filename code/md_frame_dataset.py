"""Baseline 3 dataset that combines crystal graphs with MD-frame supervision."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import Dataset
from torch_geometric.data import Batch

from binding_graph_preprocessing import ComplexPreprocessorConfig
from binding_graph_preprocessing.constants import DEFAULT_VALID_SAMPLE_IDS
from binding_graph_preprocessing.graph import build_complex_graph, to_pyg_data
from binding_graph_preprocessing.structure import build_graph_components, parse_pdb_file
from dataset import load_graph
from md_frame_labels import PB_TARGET_KEYS, parse_average_pb_labels, parse_frame_pb_labels

DEFAULT_LIGAND_MAPPING: dict[str, tuple[str, int, str | None]] = {
    "6QLN": ("LIG", 139, None),
    "6QLO": ("LIG", 139, None),
    "6QLP": ("LIG", 139, None),
    "6QLR": ("LIG", 139, None),
    "6QLT": ("J5W", 139, None),
}
FRAME_SNAPSHOTS = ["frame_200.pdb", "frame_250.pdb", "frame_300.pdb", "frame_350.pdb", "frame_400.pdb"]


@dataclass
class SampleRecord:
    sample_id: str
    crystal_graph: object
    frame_graphs: list[object]
    frame_paths: list[str]
    y_exp: float
    y_avg_pb: torch.Tensor
    y_frame_pb: torch.Tensor
    avg_mmpbsa_path: str
    frame_summary_path: str
    debug_rows: list[dict[str, object]]


def build_frame_graph(
    sample_id: str,
    frame_path: Path,
    ligand_mapping: tuple[str, int, str | None],
    label_source_path: Path,
    config: ComplexPreprocessorConfig,
):
    ligand_resname, ligand_resid, ligand_chain = ligand_mapping
    parsed = parse_pdb_file(frame_path)
    components = build_graph_components(
        parsed=parsed,
        ligand_resname=ligand_resname,
        ligand_resid=ligand_resid,
        ligand_chain=ligand_chain,
        pocket_cutoff=config.pocket_cutoff,
        protein_edge_cutoff=config.protein_edge_cutoff,
        keep_hydrogens=config.keep_hydrogens,
        remove_water=config.remove_water,
        keep_metals=config.keep_metals,
    )
    graph_dict = build_complex_graph(
        components=components,
        sample_id=sample_id,
        pdb_path=str(frame_path),
        mmpbsa_path=str(label_source_path),
        y_exp=0.0,
        y_vdw=0.0,
        y_elec=0.0,
        y_polar=0.0,
        y_nonpolar=0.0,
        delta_total_kcal=0.0,
        protein_edge_cutoff=config.protein_edge_cutoff,
        ligand_protein_edge_cutoff=config.ligand_protein_edge_cutoff,
    )
    return to_pyg_data(graph_dict)


class Baseline3Dataset(Dataset):
    def __init__(
        self,
        graph_dir: str | Path,
        raw_root_dir: str | Path,
        frame_root_dir: str | Path,
        sample_ids: list[str] | None = None,
        load_frames: bool = True,
        ligand_mapping: dict[str, tuple[str, int, str | None]] | None = None,
        config: ComplexPreprocessorConfig | None = None,
    ) -> None:
        self.graph_dir = Path(graph_dir)
        self.raw_root_dir = Path(raw_root_dir)
        self.frame_root_dir = Path(frame_root_dir)
        self.sample_ids = sample_ids or list(DEFAULT_VALID_SAMPLE_IDS)
        self.load_frames = load_frames
        self.ligand_mapping = ligand_mapping or DEFAULT_LIGAND_MAPPING
        self.config = config or ComplexPreprocessorConfig(ligand_resname="J5W", ligand_resid=139, ligand_chain=None)
        self.records = [self._load_record(sample_id) for sample_id in self.sample_ids]

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, object]:
        record = self.records[index]
        return {
            "sample_id": record.sample_id,
            "crystal_graph": record.crystal_graph,
            "frame_graphs": record.frame_graphs,
            "frame_paths": record.frame_paths,
            "y_exp": torch.tensor([[record.y_exp]], dtype=torch.float32),
            "y_avg_pb": record.y_avg_pb.unsqueeze(0),
            "y_frame_pb": record.y_frame_pb,
            "avg_mmpbsa_path": record.avg_mmpbsa_path,
            "frame_summary_path": record.frame_summary_path,
        }

    def _load_record(self, sample_id: str) -> SampleRecord:
        if sample_id not in self.ligand_mapping:
            raise ValueError(f"Missing ligand mapping for {sample_id}")

        crystal_graph = load_graph(self.graph_dir / f"{sample_id}.pt")
        sample_raw_dir = self.raw_root_dir / sample_id
        avg_mmpbsa_path = sample_raw_dir / "mmpbsa.out"
        crystal_pdb_path = sample_raw_dir / "complex.pdb"
        avg_labels = parse_average_pb_labels(avg_mmpbsa_path)
        frame_dir = self.frame_root_dir / sample_id
        summary_path = frame_dir / "snapshot_energy_summary.csv"
        frame_labels = parse_frame_pb_labels(summary_path)

        debug_rows = [self._debug_row(sample_id, "crystal", crystal_graph, str(crystal_pdb_path), self.ligand_mapping[sample_id][0])]
        frame_graphs: list[object] = []
        frame_paths: list[str] = []
        frame_targets: list[torch.Tensor] = []

        if self.load_frames:
            for snapshot_name in FRAME_SNAPSHOTS:
                if snapshot_name not in frame_labels:
                    raise FileNotFoundError(f"Missing frame label for {sample_id}: {snapshot_name}")
                frame_path = frame_dir / snapshot_name
                if not frame_path.exists():
                    raise FileNotFoundError(f"Missing frame file for {sample_id}: {frame_path}")
                frame_graph = build_frame_graph(
                    sample_id=sample_id,
                    frame_path=frame_path,
                    ligand_mapping=self.ligand_mapping[sample_id],
                    label_source_path=summary_path,
                    config=self.config,
                )
                frame_graphs.append(frame_graph)
                frame_paths.append(str(frame_path))
                frame_targets.append(torch.tensor([frame_labels[snapshot_name][key] for key in PB_TARGET_KEYS], dtype=torch.float32))
                debug_rows.append(self._debug_row(sample_id, snapshot_name, frame_graph, str(frame_path), self.ligand_mapping[sample_id][0]))

        return SampleRecord(
            sample_id=sample_id,
            crystal_graph=crystal_graph,
            frame_graphs=frame_graphs,
            frame_paths=frame_paths,
            y_exp=float(crystal_graph.y_exp.view(-1)[0].item()),
            y_avg_pb=torch.tensor([avg_labels[key] for key in PB_TARGET_KEYS], dtype=torch.float32),
            y_frame_pb=torch.stack(frame_targets) if frame_targets else torch.empty((0, len(PB_TARGET_KEYS)), dtype=torch.float32),
            avg_mmpbsa_path=str(avg_mmpbsa_path),
            frame_summary_path=str(summary_path),
            debug_rows=debug_rows,
        )

    @staticmethod
    def _debug_row(sample_id: str, structure_name: str, graph, file_path: str, ligand_resname: str) -> dict[str, object]:
        edge_attr = getattr(graph, "edge_attr", None)
        edge_dim = int(edge_attr.size(-1)) if edge_attr is not None and edge_attr.numel() > 0 else 0
        return {
            "sample_id": sample_id,
            "structure_name": structure_name,
            "file_path": file_path,
            "ligand_resname": ligand_resname,
            "num_ligand_atoms": int(graph.ligand_mask.sum().item()),
            "num_pocket_atoms": int(graph.protein_mask.sum().item()),
            "num_retained_metals": int(graph.metal_mask.sum().item()),
            "num_nodes": int(graph.x.size(0)),
            "num_edges": int(graph.edge_index.size(1)),
            "node_dim": int(graph.x.size(1)),
            "edge_dim": edge_dim,
        }

    def debug_rows(self) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        for record in self.records:
            rows.extend(record.debug_rows)
        return rows

    def record_lookup(self) -> dict[str, SampleRecord]:
        return {record.sample_id: record for record in self.records}


def collate_baseline3_batch(items: list[dict[str, object]]) -> dict[str, object]:
    crystal_graphs = [item["crystal_graph"] for item in items]
    crystal_batch = Batch.from_data_list(crystal_graphs)

    frame_graphs = [graph for item in items for graph in item["frame_graphs"]]
    frame_batch = Batch.from_data_list(frame_graphs) if frame_graphs else None
    y_frame_pb = torch.cat([item["y_frame_pb"] for item in items], dim=0) if frame_graphs else None

    return {
        "sample_ids": [item["sample_id"] for item in items],
        "crystal_batch": crystal_batch,
        "frame_batch": frame_batch,
        "y_exp": torch.cat([item["y_exp"] for item in items], dim=0),
        "y_avg_pb": torch.cat([item["y_avg_pb"] for item in items], dim=0),
        "y_frame_pb": y_frame_pb,
        "frame_paths": [path for item in items for path in item["frame_paths"]],
    }
