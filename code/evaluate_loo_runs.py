"""Evaluate the fixed leave-one-out runs and generate publication-style plots."""

from __future__ import annotations

import argparse
import csv
import math
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-codex")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

try:
    import numpy as np
except ImportError as exc:  # pragma: no cover - dependency failure path
    raise SystemExit(
        "numpy is required for evaluate_loo_runs.py. "
        "Load a conda/module environment first, for example: "
        "'source /etc/profile.d/modules.sh && module load Anaconda3/2024.06-1'."
    ) from exc

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError as exc:  # pragma: no cover - dependency failure path
    raise SystemExit(
        "matplotlib is required for evaluate_loo_runs.py. "
        "Load a conda/module environment first, for example: "
        "'source /etc/profile.d/modules.sh && module load Anaconda3/2024.06-1'."
    ) from exc


DEFAULT_RUN_DIRS = [
    "leave_one_out_val_multi_gb_loo_6QLN_val_6QLO",
    "leave_one_out_val_multi_gb_loo_6QLO_val_6QLP",
    "leave_one_out_val_multi_gb_loo_6QLP_val_6QLR",
    "leave_one_out_val_multi_gb_loo_6QLR_val_6QLT",
    "leave_one_out_val_multi_gb_loo_6QLT_val_6QLN",
]
EXPECTED_SAMPLE_IDS = ["6QLN", "6QLO", "6QLP", "6QLR", "6QLT"]
TARGET_SPECS = [
    ("exp", "pred_exp", "true_exp"),
    ("vdw", "pred_vdw", "true_vdw"),
    ("elec", "pred_elec", "true_elec"),
    ("polar", "pred_polar", "true_polar"),
    ("nonpolar", "pred_nonpolar", "true_nonpolar"),
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate the five fixed leave-one-out validation-selected runs."
    )
    parser.add_argument(
        "--output_dir",
        default="../results/training_runs",
        help="Directory containing run outputs and receiving evaluation CSVs and parity plot.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="PNG resolution for the generated parity plot.",
    )
    return parser


def load_predictions(results_root: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    seen_sample_ids: set[str] = set()

    for run_dir in DEFAULT_RUN_DIRS:
        csv_path = results_root / run_dir / "best_predictions.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"Missing predictions file: {csv_path}")

        with csv_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            file_rows = list(reader)

        if len(file_rows) != 1:
            raise ValueError(f"Expected exactly one test row in {csv_path}, found {len(file_rows)}")

        row = file_rows[0]
        sample_id = row["sample_id"]
        if sample_id in seen_sample_ids:
            raise ValueError(f"Duplicate sample_id found in merged predictions: {sample_id}")
        seen_sample_ids.add(sample_id)

        merged_row: dict[str, object] = {"source_run": run_dir, "sample_id": sample_id}
        for _, pred_key, true_key in TARGET_SPECS:
            merged_row[pred_key] = float(row[pred_key])
            merged_row[true_key] = float(row[true_key])
        rows.append(merged_row)

    observed_sample_ids = sorted(str(row["sample_id"]) for row in rows)
    if observed_sample_ids != sorted(EXPECTED_SAMPLE_IDS):
        raise ValueError(
            f"Merged sample IDs do not match expected set. "
            f"Observed={observed_sample_ids}, expected={sorted(EXPECTED_SAMPLE_IDS)}"
        )

    rows.sort(key=lambda row: EXPECTED_SAMPLE_IDS.index(str(row["sample_id"])))
    return rows


def compute_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_pred - y_true) ** 2)))


def compute_pearson_r(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if np.allclose(np.std(y_true), 0.0) or np.allclose(np.std(y_pred), 0.0):
        raise ValueError("Pearson r is undefined because one series has zero variance")
    return float(np.corrcoef(y_true, y_pred)[0, 1])


def compute_summary(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    summary_rows: list[dict[str, object]] = []
    for target, pred_key, true_key in TARGET_SPECS:
        y_pred = np.array([float(row[pred_key]) for row in rows], dtype=float)
        y_true = np.array([float(row[true_key]) for row in rows], dtype=float)
        summary_rows.append(
            {
                "target": target,
                "n": len(rows),
                "rmse": compute_rmse(y_true, y_pred),
                "pearson_r": compute_pearson_r(y_true, y_pred),
            }
        )
    return summary_rows


def save_predictions_csv(rows: list[dict[str, object]], output_path: Path) -> None:
    fieldnames = [
        "source_run",
        "sample_id",
        "pred_exp",
        "true_exp",
        "pred_vdw",
        "true_vdw",
        "pred_elec",
        "true_elec",
        "pred_polar",
        "true_polar",
        "pred_nonpolar",
        "true_nonpolar",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def save_summary_csv(summary_rows: list[dict[str, object]], output_path: Path) -> None:
    fieldnames = ["target", "n", "rmse", "pearson_r"]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(row)


def plot_exp_parity(
    rows: list[dict[str, object]],
    exp_metrics: dict[str, object],
    output_path: Path,
    dpi: int,
) -> None:
    true_values = np.array([float(row["true_exp"]) for row in rows], dtype=float)
    pred_values = np.array([float(row["pred_exp"]) for row in rows], dtype=float)

    all_values = np.concatenate([true_values, pred_values])
    data_min = float(np.min(all_values))
    data_max = float(np.max(all_values))
    value_range = data_max - data_min
    margin = max(0.7, value_range * 0.18)
    axis_min = math.floor((data_min - margin) * 2.0) / 2.0
    axis_max = math.ceil((data_max + margin) * 2.0) / 2.0

    x_line = np.linspace(axis_min, axis_max, 300)

    plt.style.use("default")
    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.labelsize": 12,
            "axes.titlesize": 13,
            "axes.linewidth": 1.6,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "xtick.major.width": 1.2,
            "ytick.major.width": 1.2,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "savefig.bbox": "tight",
        }
    )

    fig, ax = plt.subplots(figsize=(5.6, 5.4), dpi=dpi)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#fffdf5")

    ax.fill_between(
        x_line,
        x_line - 2.0,
        x_line + 2.0,
        color="#fde68a",
        alpha=0.28,
        linewidth=0,
        zorder=0,
        label=r"$\pm 2.0$ kcal/mol",
    )
    ax.fill_between(
        x_line,
        x_line - 1.0,
        x_line + 1.0,
        color="#fbbf24",
        alpha=0.45,
        linewidth=0,
        zorder=1,
        label=r"$\pm 1.0$ kcal/mol",
    )
    ax.plot(
        x_line,
        x_line,
        color="#4b5563",
        linewidth=2.0,
        zorder=2,
        label="Ideal parity",
    )
    ax.scatter(
        true_values,
        pred_values,
        s=90,
        marker="D",
        facecolor="#0f766e",
        edgecolor="#111827",
        linewidth=1.0,
        alpha=0.95,
        zorder=4,
    )

    for row in rows:
        ax.annotate(
            str(row["sample_id"]),
            (float(row["true_exp"]), float(row["pred_exp"])),
            xytext=(6, 6),
            textcoords="offset points",
            fontsize=8.5,
            color="#1f2937",
        )

    metrics_text = "\n".join(
        [
            f"RMSE = {float(exp_metrics['rmse']):.2f} kcal/mol",
            f"n = {int(exp_metrics['n'])}",
        ]
    )
    ax.text(
        0.04,
        0.96,
        metrics_text,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=10.5,
        color="#111827",
        bbox={
            "boxstyle": "round,pad=0.35",
            "facecolor": (1.0, 1.0, 1.0, 0.85),
            "edgecolor": "#d1d5db",
            "linewidth": 0.8,
        },
    )

    ax.set_xlim(axis_min, axis_max)
    ax.set_ylim(axis_min, axis_max)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(r"$\Delta G_{\mathrm{exp}}$ (kcal/mol)")
    ax.set_ylabel(r"$\Delta G_{\mathrm{pred}}$ (kcal/mol)")
    ax.set_title("LOO Parity Plot for Experimental Binding Free Energy", pad=12)
    ax.grid(True, color="#d1d5db", linewidth=0.8, linestyle=":", alpha=0.8)
    ax.legend(loc="lower right", frameon=False, fontsize=9)

    for spine in ax.spines.values():
        spine.set_color("#4b5563")

    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def main() -> None:
    args = build_parser().parse_args()

    repo_root = Path(__file__).resolve().parent
    output_dir = (repo_root / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = load_predictions(output_dir)
    summary_rows = compute_summary(rows)

    predictions_path = output_dir / "loo_eval_predictions.csv"
    summary_path = output_dir / "loo_eval_summary.csv"
    plot_path = output_dir / "loo_exp_parity_plot.png"

    save_predictions_csv(rows, predictions_path)
    save_summary_csv(summary_rows, summary_path)

    exp_metrics = next(row for row in summary_rows if row["target"] == "exp")
    plot_exp_parity(rows, exp_metrics, plot_path, dpi=args.dpi)

    print("Evaluation finished")
    print(f"  merged_predictions: {predictions_path}")
    print(f"  summary_metrics: {summary_path}")
    print(f"  exp_parity_plot: {plot_path}")


if __name__ == "__main__":
    main()
