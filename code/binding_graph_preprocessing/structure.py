"""PDB parsing, ligand selection, and pocket extraction utilities."""

from __future__ import annotations

import math
from pathlib import Path

from .constants import COVALENT_RADII, HALOGEN_ELEMENTS, METAL_RESNAMES, STANDARD_AMINO_ACIDS, WATER_RESNAMES
from .models import AtomRecord, GraphComponents, ParsedComplex, ResidueId


def parse_pdb_file(pdb_path: str | Path) -> ParsedComplex:
    """Parse a PDB file into atom records and optional CONECT adjacency."""
    path = Path(pdb_path)
    if not path.exists():
        raise FileNotFoundError(f"PDB file not found: {path}")

    atoms: list[AtomRecord] = []
    conect_records: dict[int, set[int]] = {}
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            record = line[:6].strip()
            if record in {"ATOM", "HETATM"}:
                atom = _parse_atom_line(line, record)
                if atom.altloc not in {"", "A"}:
                    continue
                atoms.append(atom)
            elif record == "CONECT":
                _parse_conect_line(line, conect_records)

    if not atoms:
        raise ValueError(f"No atoms found in {path}")
    return ParsedComplex(pdb_path=path, atoms=atoms, conect_records=conect_records)


def build_graph_components(
    parsed: ParsedComplex,
    ligand_resname: str,
    ligand_resid: int | None,
    ligand_chain: str | None,
    pocket_cutoff: float,
    protein_edge_cutoff: float,
    keep_hydrogens: bool,
    remove_water: bool,
    keep_metals: bool,
) -> GraphComponents:
    """Preprocess ligand and pocket separately, then merge into one final graph."""
    filtered_atoms = [
        atom
        for atom in parsed.atoms
        if not (remove_water and atom.residue_id.residue_name in WATER_RESNAMES)
    ]

    ligand_atoms = select_ligand_atoms(
        atoms=filtered_atoms,
        ligand_resname=ligand_resname,
        ligand_resid=ligand_resid,
        ligand_chain=ligand_chain,
    )
    ligand_residue_ids = {atom.residue_id for atom in ligand_atoms}

    protein_atoms_all = [
        atom
        for atom in filtered_atoms
        if atom.residue_id.residue_name in STANDARD_AMINO_ACIDS and atom.residue_id not in ligand_residue_ids
    ]
    metal_atoms_all = [
        atom
        for atom in filtered_atoms
        if is_metal_atom(atom) and atom.residue_id not in ligand_residue_ids
    ]

    protein_pocket_atoms = select_pocket_protein_atoms(ligand_atoms, protein_atoms_all, pocket_cutoff)
    metal_pocket_atoms = select_pocket_metal_atoms(ligand_atoms, metal_atoms_all, pocket_cutoff) if keep_metals else []

    if not keep_hydrogens:
        ligand_atoms = [atom for atom in ligand_atoms if not atom.is_hydrogen]
        protein_pocket_atoms = [atom for atom in protein_pocket_atoms if not atom.is_hydrogen]
        metal_pocket_atoms = [atom for atom in metal_pocket_atoms if not atom.is_hydrogen]

    atoms = ligand_atoms + protein_pocket_atoms + metal_pocket_atoms
    node_types = (
        [0] * len(ligand_atoms)
        + [1] * len(protein_pocket_atoms)
        + [2] * len(metal_pocket_atoms)
    )

    ligand_atom_features, ligand_bonds, ligand_chemistry_source = resolve_ligand_chemistry(parsed, ligand_atoms)
    inferred_neighbors = infer_local_covalent_neighbors(ligand_atoms, protein_pocket_atoms, metal_pocket_atoms)

    metadata = {
        "ligand_resname": ligand_atoms[0].residue_id.residue_name if ligand_atoms else ligand_resname,
        "ligand_resid": ligand_atoms[0].residue_id.residue_number if ligand_atoms else ligand_resid,
        "ligand_chain": ligand_atoms[0].residue_id.chain_id if ligand_atoms else ligand_chain,
        "num_ligand_atoms": len(ligand_atoms),
        "num_protein_atoms": len(protein_pocket_atoms),
        "num_metal_atoms": len(metal_pocket_atoms),
        "pocket_cutoff": pocket_cutoff,
        "protein_edge_cutoff": protein_edge_cutoff,
        "ligand_chemistry_source": ligand_chemistry_source,
    }
    return GraphComponents(
        ligand_atoms=ligand_atoms,
        protein_atoms=protein_pocket_atoms,
        metal_atoms=metal_pocket_atoms,
        atoms=atoms,
        node_types=node_types,
        metadata=metadata,
        ligand_bonds=ligand_bonds,
        inferred_neighbors=inferred_neighbors,
        ligand_atom_features=ligand_atom_features,
        ligand_chemistry_source=ligand_chemistry_source,
    )


def select_ligand_atoms(
    atoms: list[AtomRecord],
    ligand_resname: str,
    ligand_resid: int | None,
    ligand_chain: str | None,
) -> list[AtomRecord]:
    """Select ligand atoms using explicit identifiers with safe fallbacks for processed PDBs."""
    target_resname = ligand_resname.upper() if ligand_resname else None
    target_chain = ligand_chain.strip() if ligand_chain else None

    exact = [
        atom
        for atom in atoms
        if _matches_residue(atom, target_resname, ligand_resid, target_chain)
    ]
    if exact:
        return exact

    resid_fallback = [
        atom
        for atom in atoms
        if ligand_resid is not None
        and atom.residue_id.residue_number == ligand_resid
        and (target_chain is None or atom.residue_id.chain_id == target_chain)
        and atom.residue_id.residue_name not in STANDARD_AMINO_ACIDS
        and atom.residue_id.residue_name not in WATER_RESNAMES
        and not is_metal_atom(atom)
    ]
    if resid_fallback:
        return resid_fallback

    resname_fallback = [
        atom
        for atom in atoms
        if target_resname is not None
        and atom.residue_id.residue_name == target_resname
        and atom.residue_id.residue_name not in WATER_RESNAMES
        and not is_metal_atom(atom)
    ]
    if resname_fallback:
        return resname_fallback

    raise ValueError(
        f"Could not identify ligand atoms with ligand_resname={ligand_resname}, "
        f"ligand_resid={ligand_resid}, ligand_chain={ligand_chain}"
    )


def select_pocket_protein_atoms(
    ligand_atoms: list[AtomRecord],
    protein_atoms: list[AtomRecord],
    pocket_cutoff: float,
) -> list[AtomRecord]:
    """Keep complete protein residues whose any atom is within the cutoff of any ligand atom."""
    selected_residues: set[ResidueId] = set()
    cutoff_sq = pocket_cutoff * pocket_cutoff
    ligand_coords = [atom.coord for atom in ligand_atoms]
    for atom in protein_atoms:
        if any(distance_sq(atom.coord, ligand_coord) <= cutoff_sq for ligand_coord in ligand_coords):
            selected_residues.add(atom.residue_id)
    return [atom for atom in protein_atoms if atom.residue_id in selected_residues]


def select_pocket_metal_atoms(
    ligand_atoms: list[AtomRecord],
    metal_atoms: list[AtomRecord],
    pocket_cutoff: float,
) -> list[AtomRecord]:
    """Keep metals only if they are inside the ligand pocket range."""
    cutoff_sq = pocket_cutoff * pocket_cutoff
    ligand_coords = [atom.coord for atom in ligand_atoms]
    return [
        atom
        for atom in metal_atoms
        if any(distance_sq(atom.coord, ligand_coord) <= cutoff_sq for ligand_coord in ligand_coords)
    ]


def resolve_ligand_chemistry(
    parsed: ParsedComplex,
    ligand_atoms: list[AtomRecord],
) -> tuple[dict[int, dict[str, object]], dict[frozenset[int], dict[str, object]], str]:
    """Resolve ligand chemistry with a fixed fallback order: RDKit -> CONECT -> geometry."""
    rdkit_features, rdkit_bonds = try_build_rdkit_ligand_features(ligand_atoms)
    if rdkit_bonds:
        return rdkit_features, rdkit_bonds, "rdkit"

    serials = {atom.serial for atom in ligand_atoms}
    bonds: dict[frozenset[int], dict[str, object]] = {}
    for serial, neighbors in parsed.conect_records.items():
        for neighbor in neighbors:
            if serial in serials and neighbor in serials and serial != neighbor:
                bonds[frozenset((serial, neighbor))] = {
                    "bond_type": "unknown",
                    "aromatic": False,
                    "conjugated": False,
                }
    if bonds:
        return {}, bonds, "conect"

    for left_index, atom_i in enumerate(ligand_atoms):
        for atom_j in ligand_atoms[left_index + 1 :]:
            if likely_covalent(atom_i, atom_j):
                bonds[frozenset((atom_i.serial, atom_j.serial))] = {
                    "bond_type": "unknown",
                    "aromatic": False,
                    "conjugated": False,
                }
    return {}, bonds, "geometry"


def infer_local_covalent_neighbors(
    ligand_atoms: list[AtomRecord],
    protein_atoms: list[AtomRecord],
    metal_atoms: list[AtomRecord],
) -> dict[int, set[int]]:
    """Infer local covalent neighborhoods for node-level features only."""
    neighbors: dict[int, set[int]] = {}
    for atom in ligand_atoms + protein_atoms + metal_atoms:
        neighbors.setdefault(atom.serial, set())

    ligand_bonds = infer_geometry_bonds(ligand_atoms)
    for atom_pair in ligand_bonds:
        serial_i, serial_j = tuple(atom_pair)
        neighbors[serial_i].add(serial_j)
        neighbors[serial_j].add(serial_i)

    by_residue: dict[ResidueId, list[AtomRecord]] = {}
    for atom in protein_atoms:
        by_residue.setdefault(atom.residue_id, []).append(atom)

    for residue_atoms in by_residue.values():
        for atom_pair in infer_geometry_bonds(residue_atoms):
            serial_i, serial_j = tuple(atom_pair)
            neighbors[serial_i].add(serial_j)
            neighbors[serial_j].add(serial_i)

    sorted_residues = sorted(
        by_residue,
        key=lambda rid: (rid.chain_id, rid.residue_number, rid.insertion_code),
    )
    for left_residue, right_residue in zip(sorted_residues, sorted_residues[1:]):
        if left_residue.chain_id != right_residue.chain_id:
            continue
        if right_residue.residue_number - left_residue.residue_number != 1:
            continue
        carbonyl_c = find_atom(by_residue[left_residue], "C")
        amide_n = find_atom(by_residue[right_residue], "N")
        if carbonyl_c and amide_n and distance(carbonyl_c.coord, amide_n.coord) <= 1.8:
            neighbors[carbonyl_c.serial].add(amide_n.serial)
            neighbors[amide_n.serial].add(carbonyl_c.serial)

    return neighbors


def infer_geometry_bonds(atoms: list[AtomRecord]) -> dict[frozenset[int], dict[str, object]]:
    bonds: dict[frozenset[int], dict[str, object]] = {}
    for left_index, atom_i in enumerate(atoms):
        for atom_j in atoms[left_index + 1 :]:
            if likely_covalent(atom_i, atom_j):
                bonds[frozenset((atom_i.serial, atom_j.serial))] = {
                    "bond_type": "unknown",
                    "aromatic": False,
                    "conjugated": False,
                }
    return bonds


def try_build_rdkit_ligand_features(
    ligand_atoms: list[AtomRecord],
) -> tuple[dict[int, dict[str, object]], dict[frozenset[int], dict[str, object]]]:
    """Try to recover richer ligand chemistry with RDKit, but fail gracefully."""
    try:
        from rdkit import Chem
    except ImportError:
        return {}, {}

    pdb_block = ligand_atoms_to_pdb_block(ligand_atoms)
    mol = Chem.MolFromPDBBlock(pdb_block, removeHs=False, sanitize=True, proximityBonding=True)
    if mol is None or mol.GetNumAtoms() != len(ligand_atoms):
        return {}, {}

    features: dict[int, dict[str, object]] = {}
    for atom_record, rd_atom in zip(ligand_atoms, mol.GetAtoms()):
        hybridization_name = str(rd_atom.GetHybridization()).lower()
        features[atom_record.serial] = {
            "formal_charge": rd_atom.GetFormalCharge(),
            "is_aromatic": bool(rd_atom.GetIsAromatic()),
            "hybridization": hybridization_name,
            "degree": int(rd_atom.GetDegree()),
            "is_ring": bool(rd_atom.IsInRing()),
            "total_hydrogens": int(rd_atom.GetTotalNumHs(includeNeighbors=True)),
            "donor": bool(rd_atom.GetAtomicNum() in {7, 8, 16} and rd_atom.GetTotalNumHs(includeNeighbors=True) > 0),
            "acceptor": bool(rd_atom.GetAtomicNum() in {7, 8, 9, 15, 16, 17, 35, 53}),
        }

    bonds: dict[frozenset[int], dict[str, object]] = {}
    for rd_bond in mol.GetBonds():
        begin_atom = ligand_atoms[rd_bond.GetBeginAtomIdx()]
        end_atom = ligand_atoms[rd_bond.GetEndAtomIdx()]
        bond_type_name = str(rd_bond.GetBondType()).lower()
        normalized_bond_type = {
            "single": "single",
            "double": "double",
            "triple": "triple",
            "aromatic": "aromatic",
        }.get(bond_type_name, "unknown")
        bonds[frozenset((begin_atom.serial, end_atom.serial))] = {
            "bond_type": normalized_bond_type,
            "aromatic": bool(rd_bond.GetIsAromatic()),
            "conjugated": bool(rd_bond.GetIsConjugated()),
        }
    return features, bonds


def ligand_atoms_to_pdb_block(ligand_atoms: list[AtomRecord]) -> str:
    lines = []
    for index, atom in enumerate(ligand_atoms, start=1):
        line = (
            f"HETATM{index:5d} "
            f"{atom.atom_name:<4}"
            f"{atom.altloc or ' ':1}"
            f"{atom.residue_id.residue_name:>3} "
            f"{(atom.residue_id.chain_id or ' '):1}"
            f"{atom.residue_id.residue_number:4d}"
            f"{atom.residue_id.insertion_code or ' ':1}   "
            f"{atom.x:8.3f}{atom.y:8.3f}{atom.z:8.3f}"
            f"{atom.occupancy:6.2f}{atom.b_factor:6.2f}          "
            f"{atom.element:>2}{atom.raw_charge:>2}"
        )
        lines.append(line)
    lines.append("END")
    return "\n".join(lines)


def is_metal_atom(atom: AtomRecord) -> bool:
    residue_name = atom.residue_id.residue_name.upper()
    atom_name = "".join(character for character in atom.atom_name.upper() if character.isalpha())
    return residue_name in METAL_RESNAMES and atom_name.startswith(residue_name)


def likely_covalent(atom_i: AtomRecord, atom_j: AtomRecord) -> bool:
    if is_metal_atom(atom_i) or is_metal_atom(atom_j):
        return False
    cutoff = 1.25 * (COVALENT_RADII.get(atom_i.element, 0.77) + COVALENT_RADII.get(atom_j.element, 0.77))
    d = distance(atom_i.coord, atom_j.coord)
    return 0.4 < d <= cutoff


def find_atom(atoms: list[AtomRecord], atom_name: str) -> AtomRecord | None:
    for atom in atoms:
        if atom.atom_name == atom_name:
            return atom
    return None


def distance(point_a: tuple[float, float, float], point_b: tuple[float, float, float]) -> float:
    return math.sqrt(distance_sq(point_a, point_b))


def distance_sq(point_a: tuple[float, float, float], point_b: tuple[float, float, float]) -> float:
    dx = point_a[0] - point_b[0]
    dy = point_a[1] - point_b[1]
    dz = point_a[2] - point_b[2]
    return dx * dx + dy * dy + dz * dz


def _parse_atom_line(line: str, record_type: str) -> AtomRecord:
    residue_id = ResidueId(
        chain_id=line[21].strip() or "",
        residue_number=int(line[22:26]),
        insertion_code=line[26].strip(),
        residue_name=line[17:20].strip().upper(),
    )
    return AtomRecord(
        serial=int(line[6:11]),
        atom_name=line[12:16].strip(),
        altloc=line[16].strip(),
        residue_id=residue_id,
        x=float(line[30:38]),
        y=float(line[38:46]),
        z=float(line[46:54]),
        occupancy=float(line[54:60].strip() or 1.0),
        b_factor=float(line[60:66].strip() or 0.0),
        element=infer_element(line[76:78].strip(), line[12:16]),
        record_type=record_type,
        raw_charge=line[78:80].strip(),
        formal_charge=parse_charge(line[78:80].strip()),
    )


def parse_charge(charge_text: str) -> int:
    if not charge_text:
        return 0
    sign = 1 if charge_text.endswith("+") else -1 if charge_text.endswith("-") else 0
    magnitude_text = charge_text[:-1] if sign else charge_text
    try:
        magnitude = int(magnitude_text) if magnitude_text else 0
    except ValueError:
        return 0
    return sign * magnitude


def infer_element(element_field: str, atom_name_field: str) -> str:
    if element_field:
        return element_field.strip().upper()

    stripped = "".join(character for character in atom_name_field.strip() if character.isalpha()).upper()
    if not stripped:
        return "OTHER"
    if stripped[:2] in HALOGEN_ELEMENTS:
        return stripped[:2]
    if atom_name_field and atom_name_field[0].isdigit() and len(stripped) > 1:
        return stripped[1]
    return stripped[0]


def _parse_conect_line(line: str, conect_records: dict[int, set[int]]) -> None:
    fields = line.split()
    if len(fields) < 3:
        return
    source = int(fields[1])
    for neighbor_text in fields[2:]:
        if not neighbor_text.isdigit():
            continue
        neighbor = int(neighbor_text)
        conect_records.setdefault(source, set()).add(neighbor)
        conect_records.setdefault(neighbor, set()).add(source)


def _matches_residue(
    atom: AtomRecord,
    ligand_resname: str | None,
    ligand_resid: int | None,
    ligand_chain: str | None,
) -> bool:
    if ligand_resname is not None and atom.residue_id.residue_name != ligand_resname:
        return False
    if ligand_resid is not None and atom.residue_id.residue_number != ligand_resid:
        return False
    if ligand_chain is not None and atom.residue_id.chain_id != ligand_chain:
        return False
    return True
