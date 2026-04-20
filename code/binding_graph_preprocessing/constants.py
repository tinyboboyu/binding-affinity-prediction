"""Project-wide constants and categorical mappings for the v1 pipeline."""

from __future__ import annotations

COMMON_ELEMENTS = [
    "H",
    "C",
    "N",
    "O",
    "S",
    "P",
    "F",
    "CL",
    "BR",
    "I",
    "MG",
    "ZN",
    "CA",
    "FE",
    "CU",
    "MN",
    "NA",
    "K",
    "OTHER",
]

ELEMENT_TO_ATOMIC_NUMBER = {
    "H": 1,
    "C": 6,
    "N": 7,
    "O": 8,
    "F": 9,
    "NA": 11,
    "MG": 12,
    "P": 15,
    "S": 16,
    "CL": 17,
    "K": 19,
    "CA": 20,
    "MN": 25,
    "FE": 26,
    "CU": 29,
    "ZN": 30,
    "BR": 35,
    "I": 53,
}

COVALENT_RADII = {
    "H": 0.31,
    "C": 0.76,
    "N": 0.71,
    "O": 0.66,
    "F": 0.57,
    "P": 1.07,
    "S": 1.05,
    "CL": 1.02,
    "BR": 1.20,
    "I": 1.39,
    "NA": 1.66,
    "MG": 1.41,
    "K": 2.03,
    "CA": 1.76,
    "MN": 1.39,
    "FE": 1.32,
    "CU": 1.32,
    "ZN": 1.22,
}

STANDARD_AMINO_ACIDS = [
    "ALA",
    "ARG",
    "ASN",
    "ASP",
    "CYS",
    "GLN",
    "GLU",
    "GLY",
    "HIS",
    "ILE",
    "LEU",
    "LYS",
    "MET",
    "PHE",
    "PRO",
    "SER",
    "THR",
    "TRP",
    "TYR",
    "VAL",
]

RESIDUE_TO_INDEX = {name: index for index, name in enumerate(STANDARD_AMINO_ACIDS)}
UNK_RESIDUE_INDEX = len(RESIDUE_TO_INDEX)

WATER_RESNAMES = {"HOH", "WAT", "H2O"}
METAL_RESNAMES = {"LI", "NA", "K", "RB", "CS", "MG", "CA", "SR", "BA", "ZN", "MN", "FE", "CO", "NI", "CU", "CD"}
HALOGEN_ELEMENTS = {"CL", "BR"}
BACKBONE_ATOM_NAMES = {"N", "CA", "C", "O", "OXT"}

PROTEIN_AROMATIC_ATOMS = {
    "PHE": {"CG", "CD1", "CD2", "CE1", "CE2", "CZ"},
    "TYR": {"CG", "CD1", "CD2", "CE1", "CE2", "CZ"},
    "TRP": {"CG", "CD1", "CD2", "NE1", "CE2", "CE3", "CZ2", "CZ3", "CH2"},
    "HIS": {"CG", "ND1", "CD2", "CE1", "NE2"},
}

EDGE_TYPE_TO_INDEX = {
    "ligand_covalent": 0,
    "protein_spatial": 1,
    "ligand_protein_spatial": 2,
}

BOND_TYPE_TO_INDEX = {
    "unknown": 0,
    "single": 1,
    "double": 2,
    "triple": 3,
    "aromatic": 4,
}

HYBRIDIZATION_TO_INDEX = {
    "unspecified": 0,
    "sp": 1,
    "sp2": 2,
    "sp3": 3,
    "sp3d": 4,
    "sp3d2": 5,
}

SKIP_SAMPLE_IDS = {"6QLQ", "6QLU"}
DEFAULT_VALID_SAMPLE_IDS = ["6QLN", "6QLO", "6QLP", "6QLR", "6QLT"]
