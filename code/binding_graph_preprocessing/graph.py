"""Graph assembly and PyTorch Geometric conversion."""

from __future__ import annotations

from typing import Any

from .constants import EDGE_TYPE_TO_INDEX
from .featurizer import build_edge_feature_vector, build_node_feature_vector, edge_feature_names, node_feature_names
from .models import GraphComponents
from .structure import distance, distance_sq


def build_complex_graph(
    components: GraphComponents,
    sample_id: str,
    pdb_path: str,
    mmpbsa_path: str,
    y_exp: float,
    y_vdw: float,
    y_elec: float,
    y_polar: float,
    y_nonpolar: float,
    delta_total_kcal: float,
    protein_edge_cutoff: float,
    ligand_protein_edge_cutoff: float,
) -> dict[str, Any]:
    """Build one merged graph for ligand + pocket protein + pocket metals."""
    atoms = components.atoms
    positions = [[atom.x, atom.y, atom.z] for atom in atoms]
    serial_to_index = {atom.serial: index for index, atom in enumerate(atoms)}
    ligand_feature_map = components.ligand_atom_features
    ligand_bonds = components.ligand_bonds

    node_features: list[list[float]] = []
    residue_type: list[int] = []
    is_backbone: list[bool] = []
    ligand_mask: list[bool] = []
    protein_mask: list[bool] = []
    metal_mask: list[bool] = []

    for atom, node_type in zip(atoms, components.node_types):
        neighbor_serials = components.inferred_neighbors.get(atom.serial, set())
        inferred_degree = len(neighbor_serials)
        inferred_total_hydrogens = sum(1 for serial in neighbor_serials if atoms[serial_to_index[serial]].element == "H")
        feature_vector, residue_index, backbone_flag = build_node_feature_vector(
            atom=atom,
            node_type=node_type,
            inferred_degree=inferred_degree,
            inferred_total_hydrogens=inferred_total_hydrogens,
            ligand_feature_overrides=ligand_feature_map.get(atom.serial),
        )
        node_features.append(feature_vector)
        residue_type.append(residue_index)
        is_backbone.append(backbone_flag)
        ligand_mask.append(node_type == 0)
        protein_mask.append(node_type == 1)
        metal_mask.append(node_type == 2)

    edge_pairs: list[tuple[int, int]] = []
    edge_attr: list[list[float]] = []
    edge_type: list[int] = []
    seen_edges: set[tuple[int, int, str]] = set()

    for bond_pair, bond_info in ligand_bonds.items():
        serial_i, serial_j = tuple(bond_pair)
        if serial_i not in serial_to_index or serial_j not in serial_to_index:
            continue
        index_i = serial_to_index[serial_i]
        index_j = serial_to_index[serial_j]
        add_bidirectional_edge(
            edge_pairs=edge_pairs,
            edge_attr=edge_attr,
            edge_type=edge_type,
            seen_edges=seen_edges,
            index_i=index_i,
            index_j=index_j,
            edge_type_name="ligand_covalent",
            distance_value=distance(atoms[index_i].coord, atoms[index_j].coord),
            bond_type=str(bond_info.get("bond_type", "unknown")),
            aromatic=bool(bond_info.get("aromatic", False)),
            conjugated=bool(bond_info.get("conjugated", False)),
        )

    protein_indices = [index for index, node_type in enumerate(components.node_types) if node_type in {1, 2}]
    protein_cutoff_sq = protein_edge_cutoff * protein_edge_cutoff
    for left_offset, left_index in enumerate(protein_indices):
        for right_index in protein_indices[left_offset + 1 :]:
            if distance_sq(atoms[left_index].coord, atoms[right_index].coord) <= protein_cutoff_sq:
                add_bidirectional_edge(
                    edge_pairs=edge_pairs,
                    edge_attr=edge_attr,
                    edge_type=edge_type,
                    seen_edges=seen_edges,
                    index_i=left_index,
                    index_j=right_index,
                    edge_type_name="protein_spatial",
                    distance_value=distance(atoms[left_index].coord, atoms[right_index].coord),
                )

    ligand_indices = [index for index, node_type in enumerate(components.node_types) if node_type == 0]
    ligand_protein_cutoff_sq = ligand_protein_edge_cutoff * ligand_protein_edge_cutoff
    for ligand_index in ligand_indices:
        for protein_index in protein_indices:
            if distance_sq(atoms[ligand_index].coord, atoms[protein_index].coord) <= ligand_protein_cutoff_sq:
                add_bidirectional_edge(
                    edge_pairs=edge_pairs,
                    edge_attr=edge_attr,
                    edge_type=edge_type,
                    seen_edges=seen_edges,
                    index_i=ligand_index,
                    index_j=protein_index,
                    edge_type_name="ligand_protein_spatial",
                    distance_value=distance(atoms[ligand_index].coord, atoms[protein_index].coord),
                )

    return {
        "x": node_features,
        "pos": positions,
        "edge_index": [list(pair) for pair in zip(*edge_pairs)] if edge_pairs else [[], []],
        "edge_attr": edge_attr,
        "edge_type": edge_type,
        "node_type": components.node_types,
        "ligand_mask": ligand_mask,
        "protein_mask": protein_mask,
        "metal_mask": metal_mask,
        "residue_type": residue_type,
        "is_backbone": is_backbone,
        "y_exp": [float(y_exp)],
        "y_vdw": [float(y_vdw)],
        "y_elec": [float(y_elec)],
        "y_polar": [float(y_polar)],
        "y_nonpolar": [float(y_nonpolar)],
        "y_aux": [float(y_vdw), float(y_elec), float(y_polar), float(y_nonpolar)],
        "delta_total_kcal": [float(delta_total_kcal)],
        "sample_id": sample_id,
        "pdb_id": sample_id,
        "pdb_path": pdb_path,
        "mmpbsa_path": mmpbsa_path,
        "ligand_resname": components.metadata["ligand_resname"],
        "ligand_resid": components.metadata["ligand_resid"],
        "ligand_chain": components.metadata["ligand_chain"],
        "energy_unit": "kcal/mol",
        "metadata": {
            **components.metadata,
            "sample_id": sample_id,
            "pdb_id": sample_id,
            "pdb_path": pdb_path,
            "mmpbsa_path": mmpbsa_path,
            "energy_unit": "kcal/mol",
            "ligand_protein_edge_cutoff": ligand_protein_edge_cutoff,
            "ligand_chemistry_source": components.ligand_chemistry_source,
            "node_feature_names": node_feature_names(),
            "edge_feature_names": edge_feature_names(),
        },
    }


def to_pyg_data(graph_dict: dict[str, Any]):
    """Convert a graph dictionary into a torch_geometric Data object."""
    try:
        import torch
    except ImportError as exc:
        raise ImportError("PyTorch is required to create a PyG Data object") from exc

    try:
        from torch_geometric.data import Data
    except ImportError as exc:
        raise ImportError("torch_geometric is required to create a PyG Data object") from exc

    return Data(
        x=torch.tensor(graph_dict["x"], dtype=torch.float32),
        pos=torch.tensor(graph_dict["pos"], dtype=torch.float32),
        edge_index=torch.tensor(graph_dict["edge_index"], dtype=torch.long),
        edge_attr=torch.tensor(graph_dict["edge_attr"], dtype=torch.float32),
        edge_type=torch.tensor(graph_dict["edge_type"], dtype=torch.long),
        node_type=torch.tensor(graph_dict["node_type"], dtype=torch.long),
        ligand_mask=torch.tensor(graph_dict["ligand_mask"], dtype=torch.bool),
        protein_mask=torch.tensor(graph_dict["protein_mask"], dtype=torch.bool),
        metal_mask=torch.tensor(graph_dict["metal_mask"], dtype=torch.bool),
        residue_type=torch.tensor(graph_dict["residue_type"], dtype=torch.long),
        is_backbone=torch.tensor(graph_dict["is_backbone"], dtype=torch.bool),
        y_exp=torch.tensor(graph_dict["y_exp"], dtype=torch.float32),
        y_vdw=torch.tensor(graph_dict["y_vdw"], dtype=torch.float32),
        y_elec=torch.tensor(graph_dict["y_elec"], dtype=torch.float32),
        y_polar=torch.tensor(graph_dict["y_polar"], dtype=torch.float32),
        y_nonpolar=torch.tensor(graph_dict["y_nonpolar"], dtype=torch.float32),
        y_aux=torch.tensor(graph_dict["y_aux"], dtype=torch.float32),
        delta_total_kcal=torch.tensor(graph_dict["delta_total_kcal"], dtype=torch.float32),
        sample_id=graph_dict["sample_id"],
        pdb_id=graph_dict["pdb_id"],
        pdb_path=graph_dict["pdb_path"],
        mmpbsa_path=graph_dict["mmpbsa_path"],
        ligand_resname=graph_dict["ligand_resname"],
        ligand_resid=graph_dict["ligand_resid"],
        ligand_chain=graph_dict["ligand_chain"],
        energy_unit=graph_dict["energy_unit"],
        metadata=graph_dict["metadata"],
    )


def save_graph(graph_dict: dict[str, Any], output_path: str) -> None:
    """Persist the graph as a .pt file."""
    try:
        import torch
    except ImportError as exc:
        raise ImportError("PyTorch is required to save .pt graph files") from exc

    torch.save(to_pyg_data(graph_dict), output_path)


def add_bidirectional_edge(
    edge_pairs: list[tuple[int, int]],
    edge_attr: list[list[float]],
    edge_type: list[int],
    seen_edges: set[tuple[int, int, str]],
    index_i: int,
    index_j: int,
    edge_type_name: str,
    distance_value: float,
    bond_type: str = "unknown",
    aromatic: bool = False,
    conjugated: bool = False,
) -> None:
    forward_key = (index_i, index_j, edge_type_name)
    backward_key = (index_j, index_i, edge_type_name)
    if forward_key in seen_edges or backward_key in seen_edges:
        return

    attributes = build_edge_feature_vector(
        edge_type=edge_type_name,
        distance=distance_value,
        bond_type=bond_type,
        aromatic=aromatic,
        conjugated=conjugated,
    )
    edge_pairs.append((index_i, index_j))
    edge_pairs.append((index_j, index_i))
    edge_attr.append(attributes)
    edge_attr.append(attributes[:])
    edge_type.append(EDGE_TYPE_TO_INDEX[edge_type_name])
    edge_type.append(EDGE_TYPE_TO_INDEX[edge_type_name])
    seen_edges.add(forward_key)
    seen_edges.add(backward_key)
