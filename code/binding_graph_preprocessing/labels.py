"""Parsers for experimental and MMPBSA labels."""

from __future__ import annotations

import csv
from pathlib import Path


def parse_experimental_bd_table(bd_path: str | Path) -> dict[str, float]:
    """Read experimental delta G values in kJ/mol and convert them to kcal/mol."""
    path = Path(bd_path)
    if not path.exists():
        raise FileNotFoundError(f"Experimental label file not found: {path}")

    values: dict[str, float] = {}
    with path.open("r", encoding="utf-8", errors="ignore", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        header_seen = False
        for row in reader:
            clean_row = [cell.strip() for cell in row if cell.strip()]
            if not clean_row:
                continue
            if not header_seen:
                header_seen = True
                continue
            if len(clean_row) < 2:
                continue
            pdb_id = clean_row[0].upper()
            values[pdb_id] = float(clean_row[1]) / 4.184
    return values


def parse_gb_aux_labels(mmpbsa_path: str | Path) -> dict[str, float]:
    """Parse GB auxiliary labels from the first Differences block under GENERALIZED BORN."""
    path = Path(mmpbsa_path)
    if not path.exists():
        raise FileNotFoundError(f"MMPBSA output not found: {path}")

    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    gb_start = next((index for index, line in enumerate(lines) if "GENERALIZED BORN" in line), None)
    if gb_start is None:
        raise ValueError(f"GENERALIZED BORN section not found in {path}")

    diff_start = next(
        (index for index in range(gb_start, len(lines)) if "Differences (Complex - Receptor - Ligand)" in lines[index]),
        None,
    )
    if diff_start is None:
        raise ValueError(f"GB differences block not found in {path}")

    target_keys = {"VDWAALS", "EEL", "EGB", "ESURF", "DELTA TOTAL"}
    parsed: dict[str, float] = {}
    for line in lines[diff_start + 1 :]:
        stripped = line.strip()
        if not stripped:
            if "DELTA TOTAL" in parsed:
                break
            continue
        if stripped.startswith("Using") or stripped.startswith("POISSON BOLTZMANN"):
            break
        tokens = stripped.split()
        key = " ".join(tokens[:2]) if len(tokens) >= 2 and tokens[0] == "DELTA" and tokens[1] == "TOTAL" else tokens[0]
        if key not in target_keys:
            continue
        value_index = 2 if key == "DELTA TOTAL" else 1
        parsed[key] = float(tokens[value_index])
        if key == "DELTA TOTAL":
            break

    missing = target_keys - set(parsed)
    if missing:
        raise ValueError(f"Missing GB label keys {sorted(missing)} in {path}")

    return {
        "y_vdw": parsed["VDWAALS"],
        "y_elec": parsed["EEL"],
        "y_polar": parsed["EGB"],
        "y_nonpolar": parsed["ESURF"],
        "delta_total_kcal": parsed["DELTA TOTAL"],
    }
