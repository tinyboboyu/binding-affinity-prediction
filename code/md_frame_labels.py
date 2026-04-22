"""Baseline 3 label parsers for average and frame-level PB/MMPBSA supervision.

Average PB labels come from the POISSON BOLTZMANN Differences block in
``mmpbsa.out`` and use ``ENPOLAR`` for the nonpolar solvation term.

Frame-level PB labels come from ``snapshot_energy_summary.csv`` and use
``delta_ecavity`` for the corresponding nonpolar/cavity contribution.

Baseline 3 maps both naming schemes into one unified target dimension named
``nonpolar_solv`` while keeping ``EDISPER`` as a separate dispersion term.
"""

from __future__ import annotations

import csv
from pathlib import Path

PB_TARGET_KEYS = [
    "vdw",
    "elec",
    "polar_solv",
    "nonpolar_solv",
    "dispersion",
    "total",
]


def parse_average_pb_labels(mmpbsa_path: str | Path) -> dict[str, float]:
    """Parse binding-level average PB labels from ``mmpbsa.out``."""
    path = Path(mmpbsa_path)
    if not path.exists():
        raise FileNotFoundError(f"MMPBSA output not found: {path}")

    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    pb_start = next((index for index, line in enumerate(lines) if "POISSON BOLTZMANN" in line), None)
    if pb_start is None:
        raise ValueError(f"POISSON BOLTZMANN section not found in {path}")

    diff_start = next(
        (
            index
            for index in range(pb_start, len(lines))
            if "Differences (Complex - Receptor - Ligand)" in lines[index]
        ),
        None,
    )
    if diff_start is None:
        raise ValueError(f"PB differences block not found in {path}")

    parsed: dict[str, float] = {}
    target_keys = {"VDWAALS", "EEL", "EPB", "ENPOLAR", "EDISPER", "TOTAL", "DELTA TOTAL"}
    for line in lines[diff_start + 1 :]:
        stripped = line.strip()
        if not stripped:
            if "TOTAL" in parsed or "DELTA TOTAL" in parsed:
                break
            continue
        if stripped.startswith("Using") or stripped.startswith("GENERALIZED BORN"):
            break
        tokens = stripped.split()
        if len(tokens) < 2:
            continue

        if tokens[0] == "DELTA" and len(tokens) >= 2 and tokens[1] == "TOTAL":
            key = "DELTA TOTAL"
            value = float(tokens[2])
        else:
            key = tokens[0]
            if key not in target_keys:
                continue
            value = float(tokens[1])

        if key in target_keys:
            parsed[key] = value

    total_value = parsed.get("TOTAL", parsed.get("DELTA TOTAL"))
    required = {"VDWAALS", "EEL", "EPB", "ENPOLAR", "EDISPER"}
    missing = required - set(parsed)
    if missing or total_value is None:
        raise ValueError(f"Missing PB label keys in {path}: {sorted(missing)} total={total_value is None}")

    return {
        "vdw": parsed["VDWAALS"],
        "elec": parsed["EEL"],
        "polar_solv": parsed["EPB"],
        "nonpolar_solv": parsed["ENPOLAR"],
        "dispersion": parsed["EDISPER"],
        "total": total_value,
    }


def parse_frame_pb_labels(summary_csv_path: str | Path) -> dict[str, dict[str, float]]:
    """Parse frame-level binding delta PB labels from ``snapshot_energy_summary.csv``."""
    path = Path(summary_csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Frame summary CSV not found: {path}")

    by_snapshot: dict[str, dict[str, float]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row.get("method") != "PB":
                continue
            snapshot = row["snapshot"]
            by_snapshot[snapshot] = {
                "vdw": float(row["delta_vdwaals"]),
                "elec": float(row["delta_eel"]),
                "polar_solv": float(row["delta_epb"]),
                "nonpolar_solv": float(row["delta_ecavity"]),
                "dispersion": float(row["delta_edisper"]),
                "total": float(row["delta_g_total"]),
            }

    if not by_snapshot:
        raise ValueError(f"No PB rows found in {path}")
    return by_snapshot

