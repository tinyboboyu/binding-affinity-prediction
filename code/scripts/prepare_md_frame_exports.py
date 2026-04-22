#!/usr/bin/env python3
"""Export selected MD frames and per-frame MMPBSA summaries for baseline 3/4."""

from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

FRAME_IDS = [200, 250, 300, 350, 400]
GB_SURFTEN = 0.0072
DEFAULT_OUTPUT_ROOT = Path("/lunarc/nobackup/projects/teobio/Xiaofan/binding_affinity_prediction/data/md_frame_exports")
DEFAULT_SOURCE_DIRS = {
    "6QLT": Path("/home/yuxiaofan/lu2025-17-57/Xiaofan/Day6/MMGBSA"),
    "6QLN": Path("/home/yuxiaofan/lu2025-17-57/diletta/Day6_7_8_9/Ex"),
    "6QLR": Path("/home/yuxiaofan/lu2025-17-57/George/Task_8"),
    "6QLO": Path("/home/yuxiaofan/lu2025-17-57/wilma/Part2/Ex5"),
    "6QLP": Path("/home/yuxiaofan/lu2025-17-57/Anna/day7"),
}
REQUIRED_FILES = [
    "nowat.prmtop",
    "nowat.mdcrd5",
    "mmpbsa.out",
    "_MMPBSA_complex_gb.mdout.0",
    "_MMPBSA_receptor_gb.mdout.0",
    "_MMPBSA_ligand_gb.mdout.0",
    "_MMPBSA_complex_gb_surf.dat.0",
    "_MMPBSA_receptor_gb_surf.dat.0",
    "_MMPBSA_ligand_gb_surf.dat.0",
    "_MMPBSA_complex_pb.mdout.0",
    "_MMPBSA_receptor_pb.mdout.0",
    "_MMPBSA_ligand_pb.mdout.0",
]
CSV_COLUMNS = [
    "method",
    "snapshot",
    "delta_vdwaals",
    "delta_eel",
    "delta_egb",
    "delta_esurf",
    "delta_epb",
    "delta_ecavity",
    "delta_edisper",
    "delta_g_gas",
    "delta_g_solv",
    "delta_g_total",
]


class ExportError(RuntimeError):
    """Raised when one export target cannot be processed."""


@dataclass(frozen=True)
class ComponentEnergies:
    vdwaals: float
    eel: float
    egb: float | None = None
    esurf: float | None = None
    epb: float | None = None
    ecavity: float | None = None
    edisper: float | None = None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export selected MD frames and per-frame MMPBSA summaries.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Directory that will contain one subdirectory per PDB ID.",
    )
    parser.add_argument(
        "--pdb-ids",
        nargs="+",
        default=list(DEFAULT_SOURCE_DIRS),
        help="Subset of PDB IDs to process. Defaults to all configured IDs.",
    )
    parser.add_argument(
        "--frames",
        nargs="+",
        type=int,
        default=FRAME_IDS,
        help="Frame indices to export and summarize.",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Validate required files and report status without writing outputs.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip an ID when all expected output files already exist.",
    )
    return parser


def ensure_known_ids(pdb_ids: Iterable[str]) -> list[str]:
    resolved = []
    unknown = []
    for pdb_id in pdb_ids:
        if pdb_id not in DEFAULT_SOURCE_DIRS:
            unknown.append(pdb_id)
        else:
            resolved.append(pdb_id)
    if unknown:
        raise SystemExit(f"Unknown PDB ID(s): {unknown}. Known IDs: {list(DEFAULT_SOURCE_DIRS)}")
    return resolved


def validate_source_dir(source_dir: Path) -> None:
    missing = [name for name in REQUIRED_FILES if not (source_dir / name).exists()]
    if missing:
        raise ExportError(f"Missing required file(s) in {source_dir}: {missing}")


def parse_mdout_component_file(mdout_path: Path) -> dict[int, ComponentEnergies]:
    frames: dict[int, dict[str, float | None]] = {}
    current_frame: int | None = None

    with mdout_path.open("r", encoding="utf-8", errors="replace") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if line.startswith("Processing frame"):
                current_frame = int(line.split()[-1])
                frames[current_frame] = {
                    "vdwaals": None,
                    "eel": None,
                    "egb": None,
                    "epb": None,
                    "ecavity": None,
                    "edisper": None,
                }
                continue

            if current_frame is None:
                continue

            if "VDWAALS =" in line:
                parts = line.replace("=", " = ").split()
                mapping = _parse_named_values(parts)
                for key in ("VDWAALS", "EEL", "EGB", "EPB"):
                    if key in mapping:
                        frames[current_frame][key.lower()] = mapping[key]
            elif "ECAVITY =" in line:
                parts = line.replace("=", " = ").split()
                mapping = _parse_named_values(parts)
                for key in ("ECAVITY", "EDISPER"):
                    if key in mapping:
                        frames[current_frame][key.lower()] = mapping[key]

    return {
        frame_id: ComponentEnergies(
            vdwaals=_require_float(values["vdwaals"], mdout_path, frame_id, "VDWAALS"),
            eel=_require_float(values["eel"], mdout_path, frame_id, "EEL"),
            egb=values["egb"],
            epb=values["epb"],
            ecavity=values["ecavity"],
            edisper=values["edisper"],
        )
        for frame_id, values in frames.items()
    }


def _parse_named_values(parts: list[str]) -> dict[str, float]:
    values: dict[str, float] = {}
    for index, token in enumerate(parts):
        if token == "=" and index > 0 and index + 1 < len(parts):
            name = parts[index - 1]
            try:
                values[name] = float(parts[index + 1])
            except ValueError:
                continue
    return values


def _require_float(value: float | None, path: Path, frame_id: int, field: str) -> float:
    if value is None:
        raise ExportError(f"Could not parse {field} for frame {frame_id} from {path}")
    return value


def parse_gb_surf_file(surf_path: Path) -> dict[int, float]:
    frames: dict[int, float] = {}
    with surf_path.open("r", encoding="utf-8", errors="replace") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            frame_id = int(parts[0])
            surface_area = float(parts[1])
            frames[frame_id] = surface_area * GB_SURFTEN
    return frames


def compute_method_rows(
    frames: list[int],
    complex_terms: dict[int, ComponentEnergies],
    receptor_terms: dict[int, ComponentEnergies],
    ligand_terms: dict[int, ComponentEnergies],
    method: str,
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for frame_id in frames:
        complex_frame = complex_terms[frame_id]
        receptor_frame = receptor_terms[frame_id]
        ligand_frame = ligand_terms[frame_id]
        delta_vdwaals = complex_frame.vdwaals - receptor_frame.vdwaals - ligand_frame.vdwaals
        delta_eel = complex_frame.eel - receptor_frame.eel - ligand_frame.eel
        delta_g_gas = delta_vdwaals + delta_eel
        snapshot = f"frame_{frame_id}.pdb"

        if method == "GB":
            delta_egb = _delta_optional(complex_frame.egb, receptor_frame.egb, ligand_frame.egb)
            delta_esurf = _delta_optional(
                complex_frame.esurf,
                receptor_frame.esurf,
                ligand_frame.esurf,
            )
            delta_g_solv = delta_egb + delta_esurf
            delta_g_total = delta_g_gas + delta_g_solv
            rows.append(
                {
                    "method": "GB",
                    "snapshot": snapshot,
                    "delta_vdwaals": _fmt(delta_vdwaals),
                    "delta_eel": _fmt(delta_eel),
                    "delta_egb": _fmt(delta_egb),
                    "delta_esurf": _fmt(delta_esurf),
                    "delta_epb": "",
                    "delta_ecavity": "",
                    "delta_edisper": "",
                    "delta_g_gas": _fmt(delta_g_gas),
                    "delta_g_solv": _fmt(delta_g_solv),
                    "delta_g_total": _fmt(delta_g_total),
                }
            )
        elif method == "PB":
            delta_epb = _delta_optional(complex_frame.epb, receptor_frame.epb, ligand_frame.epb)
            delta_ecavity = _delta_optional(
                complex_frame.ecavity,
                receptor_frame.ecavity,
                ligand_frame.ecavity,
            )
            delta_edisper = _delta_optional(
                complex_frame.edisper,
                receptor_frame.edisper,
                ligand_frame.edisper,
            )
            delta_g_solv = delta_epb + delta_ecavity + delta_edisper
            delta_g_total = delta_g_gas + delta_g_solv
            rows.append(
                {
                    "method": "PB",
                    "snapshot": snapshot,
                    "delta_vdwaals": _fmt(delta_vdwaals),
                    "delta_eel": _fmt(delta_eel),
                    "delta_egb": "",
                    "delta_esurf": "",
                    "delta_epb": _fmt(delta_epb),
                    "delta_ecavity": _fmt(delta_ecavity),
                    "delta_edisper": _fmt(delta_edisper),
                    "delta_g_gas": _fmt(delta_g_gas),
                    "delta_g_solv": _fmt(delta_g_solv),
                    "delta_g_total": _fmt(delta_g_total),
                }
            )
        else:
            raise ExportError(f"Unsupported method: {method}")
    return rows


def _delta_optional(complex_value: float | None, receptor_value: float | None, ligand_value: float | None) -> float:
    if complex_value is None or receptor_value is None or ligand_value is None:
        raise ExportError("Encountered missing component while computing delta energies")
    return complex_value - receptor_value - ligand_value


def _fmt(value: float) -> str:
    return f"{value:.4f}"


def build_markdown(rows: list[dict[str, str]], frames: list[int]) -> str:
    gb_rows = [row for row in rows if row["method"] == "GB"]
    pb_rows = [row for row in rows if row["method"] == "PB"]
    if len(frames) == 1:
        frame_list = f"`frame_{frames[0]}.pdb`"
    else:
        leading = ", ".join(f"`frame_{frame}.pdb`" for frame in frames[:-1])
        frame_list = f"{leading}, and `frame_{frames[-1]}.pdb`"

    lines = [
        "# Snapshot Energy Summary",
        "",
        f"Frames correspond to {frame_list}.",
        "",
        "All values are in kcal/mol. `Delta` values are calculated as `complex - receptor - ligand` for the same frame.",
        "",
        "## GB Per-Frame Delta G",
        "",
        "| Snapshot | Delta_VDWAALS | Delta_EEL | Delta_EGB | Delta_ESURF | Delta_G_gas | Delta_G_solv | Delta_G_total |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    lines.extend(
        f"| {row['snapshot']} | {row['delta_vdwaals']} | {row['delta_eel']} | {row['delta_egb']} | {row['delta_esurf']} | {row['delta_g_gas']} | {row['delta_g_solv']} | {row['delta_g_total']} |"
        for row in gb_rows
    )
    lines.extend(
        [
            "",
            "## PB Per-Frame Delta G",
            "",
            "| Snapshot | Delta_VDWAALS | Delta_EEL | Delta_EPB | Delta_ECAVITY | Delta_EDISPER | Delta_G_gas | Delta_G_solv | Delta_G_total |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    lines.extend(
        f"| {row['snapshot']} | {row['delta_vdwaals']} | {row['delta_eel']} | {row['delta_epb']} | {row['delta_ecavity']} | {row['delta_edisper']} | {row['delta_g_gas']} | {row['delta_g_solv']} | {row['delta_g_total']} |"
        for row in pb_rows
    )
    lines.append("")
    return "\n".join(lines)


def load_component_frames(source_dir: Path, prefix: str, method: str) -> dict[int, ComponentEnergies]:
    mdout_path = source_dir / f"_MMPBSA_{prefix}_{method.lower()}.mdout.0"
    frames = parse_mdout_component_file(mdout_path)
    if method == "GB":
        surf_values = parse_gb_surf_file(source_dir / f"_MMPBSA_{prefix}_gb_surf.dat.0")
        for frame_id, energies in list(frames.items()):
            if frame_id not in surf_values:
                raise ExportError(f"Missing ESURF entry for frame {frame_id} in {prefix} GB surf data")
            frames[frame_id] = ComponentEnergies(
                vdwaals=energies.vdwaals,
                eel=energies.eel,
                egb=energies.egb,
                esurf=surf_values[frame_id],
            )
    return frames


def validate_frame_coverage(frames: list[int], source_dir: Path) -> None:
    max_frame = max(frames)
    counts = {
        name: _count_entries(source_dir / name)
        for name in [
            "_MMPBSA_complex_gb.mdout.0",
            "_MMPBSA_receptor_gb.mdout.0",
            "_MMPBSA_ligand_gb.mdout.0",
            "_MMPBSA_complex_gb_surf.dat.0",
            "_MMPBSA_receptor_gb_surf.dat.0",
            "_MMPBSA_ligand_gb_surf.dat.0",
            "_MMPBSA_complex_pb.mdout.0",
            "_MMPBSA_receptor_pb.mdout.0",
            "_MMPBSA_ligand_pb.mdout.0",
        ]
    }
    too_short = {name: count for name, count in counts.items() if count < max_frame}
    if too_short:
        raise ExportError(f"Requested frame {max_frame}, but some files are too short: {too_short}")


def _count_entries(path: Path) -> int:
    if path.name.endswith(".mdout.0"):
        return path.read_text(encoding="utf-8", errors="replace").count("Processing frame")
    count = 0
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if line and not line.startswith("#"):
                count += 1
    return count


def run_cpptraj_extract(source_dir: Path, output_dir: Path, frames: list[int]) -> None:
    cpptraj_input = "\n".join(
        [
            "parm nowat.prmtop",
            "",
            *[
                f"trajin nowat.mdcrd5 {frame} {frame}\ntrajout {output_dir / f'frame_{frame}.pdb'} pdb\nrun\nclear trajin\nclear trajout"
                for frame in frames[:-1]
            ],
            f"trajin nowat.mdcrd5 {frames[-1]} {frames[-1]}",
            f"trajout {output_dir / f'frame_{frames[-1]}.pdb'} pdb",
            "run",
            "exit",
            "",
        ]
    )

    with tempfile.NamedTemporaryFile("w", suffix=".in", delete=False, dir=output_dir) as handle:
        handle.write(cpptraj_input)
        input_path = Path(handle.name)

    try:
        command = resolve_cpptraj_command(input_path)
        completed = subprocess.run(
            command,
            cwd=source_dir,
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode != 0:
            raise ExportError(
                "cpptraj extraction failed.\n"
                f"stdout:\n{completed.stdout}\n\nstderr:\n{completed.stderr}"
            )
    finally:
        input_path.unlink(missing_ok=True)


def resolve_cpptraj_command(input_path: Path) -> list[str]:
    if shutil.which("cpptraj"):
        return ["cpptraj", "-i", str(input_path)]

    module_script = Path("/etc/profile.d/modules.sh")
    if module_script.exists():
        shell_cmd = (
            f". {module_script} && "
            "module add GCC/11.2.0 OpenMPI/4.1.1 Amber/22.0-AmberTools-22.3 && "
            f"cpptraj -i {input_path}"
        )
        return ["bash", "-lc", shell_cmd]

    raise ExportError("cpptraj is not available and environment modules could not be loaded")


def write_csv(rows: list[dict[str, str]], output_path: Path) -> None:
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def outputs_exist(output_dir: Path, frames: list[int]) -> bool:
    expected = [output_dir / f"frame_{frame}.pdb" for frame in frames]
    expected.extend(
        [
            output_dir / "snapshot_energy_summary.csv",
            output_dir / "snapshot_energy_summary.md",
        ]
    )
    return all(path.exists() for path in expected)


def export_one(pdb_id: str, source_dir: Path, output_root: Path, frames: list[int], check_only: bool, skip_existing: bool) -> str:
    validate_source_dir(source_dir)
    validate_frame_coverage(frames, source_dir)

    output_dir = output_root / pdb_id
    if check_only:
        return f"[OK] {pdb_id}: all required inputs found in {source_dir}"

    output_dir.mkdir(parents=True, exist_ok=True)
    if skip_existing and outputs_exist(output_dir, frames):
        return f"[SKIP] {pdb_id}: outputs already exist in {output_dir}"

    run_cpptraj_extract(source_dir, output_dir, frames)

    complex_gb = load_component_frames(source_dir, "complex", "GB")
    receptor_gb = load_component_frames(source_dir, "receptor", "GB")
    ligand_gb = load_component_frames(source_dir, "ligand", "GB")
    complex_pb = load_component_frames(source_dir, "complex", "PB")
    receptor_pb = load_component_frames(source_dir, "receptor", "PB")
    ligand_pb = load_component_frames(source_dir, "ligand", "PB")

    rows = compute_method_rows(frames, complex_gb, receptor_gb, ligand_gb, method="GB")
    rows.extend(compute_method_rows(frames, complex_pb, receptor_pb, ligand_pb, method="PB"))
    write_csv(rows, output_dir / "snapshot_energy_summary.csv")
    (output_dir / "snapshot_energy_summary.md").write_text(build_markdown(rows, frames), encoding="utf-8")

    return f"[OK] {pdb_id}: exported {len(frames)} frame(s) to {output_dir}"


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    pdb_ids = ensure_known_ids(args.pdb_ids)
    frames = sorted(dict.fromkeys(args.frames))

    results: list[str] = []
    failures: list[str] = []
    for pdb_id in pdb_ids:
        source_dir = DEFAULT_SOURCE_DIRS[pdb_id]
        try:
            results.append(
                export_one(
                    pdb_id=pdb_id,
                    source_dir=source_dir,
                    output_root=args.output_root,
                    frames=frames,
                    check_only=args.check_only,
                    skip_existing=args.skip_existing,
                )
            )
        except Exception as exc:  # noqa: BLE001
            failures.append(f"[FAIL] {pdb_id}: {exc}")

    for line in results + failures:
        print(line)

    if failures:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
