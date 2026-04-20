"""Feature extraction for graph nodes and edges."""

from __future__ import annotations

from .constants import (
    BACKBONE_ATOM_NAMES,
    BOND_TYPE_TO_INDEX,
    COMMON_ELEMENTS,
    EDGE_TYPE_TO_INDEX,
    ELEMENT_TO_ATOMIC_NUMBER,
    HYBRIDIZATION_TO_INDEX,
    PROTEIN_AROMATIC_ATOMS,
    RESIDUE_TO_INDEX,
    STANDARD_AMINO_ACIDS,
    UNK_RESIDUE_INDEX,
)
from .models import AtomRecord


def build_node_feature_vector(
    atom: AtomRecord,
    node_type: int,
    inferred_degree: int,
    inferred_total_hydrogens: int,
    ligand_feature_overrides: dict[str, object] | None = None,
) -> tuple[list[float], int, bool]:
    """Construct a practical baseline node feature vector plus residue helpers."""
    ligand_feature_overrides = ligand_feature_overrides or {}
    element = atom.element if atom.element in COMMON_ELEMENTS else "OTHER"
    residue_name = atom.residue_id.residue_name
    is_ligand = node_type == 0
    is_protein = node_type == 1
    is_metal = node_type == 2
    aromatic_flag = bool(
        ligand_feature_overrides.get("is_aromatic", _is_protein_aromatic(atom) if is_protein else False)
    )
    hybridization_name = str(ligand_feature_overrides.get("hybridization", "unspecified")).lower()
    donor = bool(ligand_feature_overrides.get("donor", False))
    acceptor = bool(ligand_feature_overrides.get("acceptor", False))
    formal_charge = int(ligand_feature_overrides.get("formal_charge", atom.formal_charge))
    ring_flag = bool(ligand_feature_overrides.get("is_ring", aromatic_flag))
    total_hydrogens = int(ligand_feature_overrides.get("total_hydrogens", inferred_total_hydrogens))
    degree = int(ligand_feature_overrides.get("degree", inferred_degree))
    residue_type_index = RESIDUE_TO_INDEX.get(residue_name, UNK_RESIDUE_INDEX) if is_protein else UNK_RESIDUE_INDEX
    is_backbone = atom.atom_name in BACKBONE_ATOM_NAMES and is_protein

    features = [
        float(ELEMENT_TO_ATOMIC_NUMBER.get(atom.element, 0)),
        float(formal_charge),
        float(aromatic_flag),
        float(HYBRIDIZATION_TO_INDEX.get(hybridization_name, 0)),
        float(degree),
        float(ring_flag),
        float(total_hydrogens),
        float(is_ligand),
        float(is_protein),
        float(is_metal),
        float(residue_type_index),
        float(is_backbone),
        float(donor),
        float(acceptor),
    ]
    features.extend(one_hot(element, COMMON_ELEMENTS))
    return features, residue_type_index, is_backbone


def build_edge_feature_vector(
    edge_type: str,
    distance: float,
    bond_type: str = "unknown",
    aromatic: bool = False,
    conjugated: bool = False,
) -> list[float]:
    """Encode edge type, distance, and bond annotations when available."""
    features = [
        float(EDGE_TYPE_TO_INDEX[edge_type]),
        float(distance),
        float(edge_type == "ligand_covalent"),
        float(edge_type == "protein_spatial"),
        float(edge_type == "ligand_protein_spatial"),
        float(aromatic),
        float(conjugated),
    ]
    features.extend(one_hot(bond_type, list(BOND_TYPE_TO_INDEX)))
    return features


def node_feature_names() -> list[str]:
    return [
        "atomic_number",
        "formal_charge",
        "is_aromatic",
        "hybridization_index",
        "degree",
        "is_ring",
        "total_hydrogens",
        "is_ligand",
        "is_protein",
        "is_metal",
        "residue_type_index",
        "is_backbone",
        "is_donor",
        "is_acceptor",
        *[f"element_{element.lower()}" for element in COMMON_ELEMENTS],
    ]


def edge_feature_names() -> list[str]:
    return [
        "edge_type_index",
        "distance",
        "is_ligand_covalent",
        "is_protein_spatial",
        "is_ligand_protein_spatial",
        "bond_is_aromatic",
        "bond_is_conjugated",
        *[f"bond_type_{name}" for name in BOND_TYPE_TO_INDEX],
    ]


def one_hot(value: str, categories: list[str]) -> list[float]:
    normalized = value.upper()
    return [1.0 if category.upper() == normalized else 0.0 for category in categories]


def _is_protein_aromatic(atom: AtomRecord) -> bool:
    residue_name = atom.residue_id.residue_name
    return residue_name in STANDARD_AMINO_ACIDS and atom.atom_name in PROTEIN_AROMATIC_ATOMS.get(residue_name, set())
