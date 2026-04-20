"""Structured records used by the v1 preprocessing pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class ResidueId:
    chain_id: str
    residue_number: int
    insertion_code: str
    residue_name: str


@dataclass
class AtomRecord:
    serial: int
    atom_name: str
    element: str
    x: float
    y: float
    z: float
    residue_id: ResidueId
    record_type: str
    altloc: str = ""
    formal_charge: int = 0
    raw_charge: str = ""
    occupancy: float = 1.0
    b_factor: float = 0.0

    @property
    def coord(self) -> tuple[float, float, float]:
        return (self.x, self.y, self.z)

    @property
    def is_hydrogen(self) -> bool:
        return self.element == "H"


@dataclass
class ParsedComplex:
    pdb_path: Path
    atoms: list[AtomRecord]
    conect_records: dict[int, set[int]]


@dataclass
class GraphComponents:
    ligand_atoms: list[AtomRecord]
    protein_atoms: list[AtomRecord]
    metal_atoms: list[AtomRecord]
    atoms: list[AtomRecord]
    node_types: list[int]
    metadata: dict[str, object] = field(default_factory=dict)
    ligand_bonds: dict[frozenset[int], dict[str, object]] = field(default_factory=dict)
    inferred_neighbors: dict[int, set[int]] = field(default_factory=dict)
    ligand_atom_features: dict[int, dict[str, object]] = field(default_factory=dict)
    ligand_chemistry_source: str = "geometry"
