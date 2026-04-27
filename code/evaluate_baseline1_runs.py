"""Aggregate Baseline 1 run predictions and generate an experimental Delta G parity plot."""

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
        "numpy is required for evaluate_baseline1_runs.py. "
        "Load a conda/module environment first, for example: "
        "'source /etc/profile.d/modules.sh && module load Anaconda3/2024.06-1'."
    ) from exc

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError as exc:  # pragma: no cover - dependency failure path
    raise SystemExit(
        "matplotlib is required for evaluate_baseline1_runs.py. "
        "Load a conda/module environment first, for example: "
        "'source /etc/profile.d/modules.sh && module load Anaconda3/2024.06-1'."
    ) from exc


EXPECTED_SAMPLE_IDS = ["6QLN", "6QLO", "6QLP", "6QLR", "6QLT"]
DEFAULT_RESULTS_ROOT = Path(__file__).resolve().parent.parent / "results" / "training_runs"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize Baseline 1 prediction CSVs and plot exp parity.")
    parser.add_argument("--results_root", default=str(DEFAULT_RESULTS_ROOT))
    parser.add_argument("--run_dirs", nargs="*", default=None)
    parser.add_argument("--run_prefix", default="baseline1_")
    parser.add_argument("--split", choices=["test", "val", "train", "all"], default="test")
    parser.add_argument("--dpi", type=int, default=300)
    return parser


def compute_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_pred - y_true)))


def compute_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_pred - y_true) ** 2)))


def compute_pearson_r(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) < 2 or np.allclose(np.std(y_true), 0.0) or np.allclose(np.std(y_pred), 0.0):
        return float("nan")
    return float(np.corrcoef(y_true, y_pred)[0, 1])


def resolve_run_dirs(results_root: Path, requested_run_dirs: list[str] | None, run_prefix: str) -> list[str]:
    if requested_run_dirs:
        return requested_run_dirs
    return sorted(
        path.name
        for path in results_root.iterdir()
        if path.is_dir() and path.name.startswith(run_prefix) and (path / "best_predictions.csv").exists()
    )


def load_predictions(results_root: Path, run_dirs: list[str], split_filter: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    seen_test_sample_ids: set[str] = set()

    for run_dir in run_dirs:
        csv_path = results_root / run_dir / "best_predictions.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"Missing predictions file: {csv_path}")

        with csv_path.open("r", encoding="utf-8", newline="") as handle:
            file_rows = list(csv.DictReader(handle))

        if not file_rows:
            raise ValueError(f"No prediction rows found in {csv_path}")

        for row in file_rows:
            split = row["split"]
            if split_filter != "all" and split != split_filter:
                continue
            sample_id = row["sample_id"]
            if split_filter == "test":
                if sample_id in seen_test_sample_ids:
                    raise ValueError(f"Duplicate test sample_id found in merged predictions: {sample_id}")
                seen_test_sample_ids.add(sample_id)
            rows.append(
                {
                    "run_dir": run_dir,
                    "sample_id": sample_id,
                    "split": split,
                    "true_exp": float(row["true_exp"]),
                    "pred_exp": float(row["pred_exp"]),
                    "abs_error": float(row["abs_error"]),
                }
            )

    if split_filter == "test" and sorted(EXPECTED_SAMPLE_IDS) == sorted(str(row["sample_id"]) for row in rows):
        rows.sort(key=lambda row: EXPECTED_SAMPLE_IDS.index(str(row["sample_id"])))
    else:
        rows.sort(key=lambda row: (str(row["split"]), str(row["sample_id"]), str(row["run_dir"])))
    return rows


def compute_summary(rows: list[dict[str, object]]) -> dict[str, object]:
    y_true = np.array([float(row["true_exp"]) for row in rows], dtype=float)
    y_pred = np.array([float(row["pred_exp"]) for row in rows], dtype=float)
    return {
        "target": "exp",
        "mae": compute_mae(y_true, y_pred),
        "rmse": compute_rmse(y_true, y_pred),
        "pearson_r": compute_pearson_r(y_true, y_pred),
        "n": len(rows),
    }


def save_predictions_csv(rows: list[dict[str, object]], output_path: Path) -> None:
    fieldnames = ["run_dir", "sample_id", "split", "true_exp", "pred_exp", "abs_error"]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_summary_csv(summary: dict[str, object], output_path: Path) -> None:
    fieldnames = ["target", "mae", "rmse", "pearson_r", "n"]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(summary)


def plot_exp_parity(rows: list[dict[str, object]], metrics: dict[str, object], output_path: Path, dpi: int) -> None:
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
    ax.fill_between(x_line, x_line - 2.0, x_line + 2.0, color="#fde68a", alpha=0.28, linewidth=0, zorder=0)
    ax.fill_between(x_line, x_line - 1.0, x_line + 1.0, color="#fbbf24", alpha=0.45, linewidth=0, zorder=1)
    ax.plot(x_line, x_line, color="#4b5563", linewidth=2.0, zorder=2, label="Ideal parity")
    ax.scatter(true_values, pred_values, s=90, marker="D", facecolor="#0f766e", edgecolor="#111827", linewidth=1.0, alpha=0.95, zorder=4)

    for row in rows:
        ax.annotate(
            str(row["sample_id"]),
            (float(row["true_exp"]), float(row["pred_exp"])),
            xytext=(6, 6),
            textcoords="offset points",
            fontsize=8.5,
            color="#1f2937",
        )

    pearson_text = "nan" if math.isnan(float(metrics["pearson_r"])) else f"{float(metrics['pearson_r']):.2f}"
    metrics_text = "\n".join(
        [
            f"MAE = {float(metrics['mae']):.2f} kcal/mol",
            f"RMSE = {float(metrics['rmse']):.2f} kcal/mol",
            f"r = {pearson_text}",
            f"n = {int(metrics['n'])}",
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
    ax.set_title("Baseline 1 Parity Plot for Experimental Binding Free Energy", pad=12)
    ax.grid(True, color="#d1d5db", linewidth=0.8, linestyle=":", alpha=0.8)
    ax.legend(loc="lower right", frameon=False, fontsize=9)
    for spine in ax.spines.values():
        spine.set_color("#4b5563")

    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def main() -> None:
    args = build_parser().parse_args()
    results_root = Path(args.results_root).expanduser().resolve()
    run_dirs = resolve_run_dirs(results_root, args.run_dirs, args.run_prefix)
    if not run_dirs:
        raise SystemExit(f"No Baseline 1 run directories found under {results_root}")

    rows = load_predictions(results_root, run_dirs, split_filter=args.split)
    if not rows:
        raise SystemExit(f"No prediction rows found for split={args.split}")

    summary = compute_summary(rows)
    suffix = "" if args.split == "test" else f"_{args.split}"
    merged_path = results_root / f"baseline1{suffix}_merged_predictions.csv"
    summary_path = results_root / f"baseline1{suffix}_summary_metrics.csv"
    plot_path = results_root / f"baseline1{suffix}_exp_parity_plot.png"

    save_predictions_csv(rows, merged_path)
    save_summary_csv(summary, summary_path)
    plot_exp_parity(rows, summary, plot_path, dpi=args.dpi)

    pearson_text = "nan" if math.isnan(float(summary["pearson_r"])) else f"{float(summary['pearson_r']):.4f}"
    print("Baseline 1 evaluation finished")
    print(f"  split: {args.split}")
    print(f"  mae={float(summary['mae']):.4f} rmse={float(summary['rmse']):.4f} pearson_r={pearson_text} n={int(summary['n'])}")
    print(f"  merged_predictions: {merged_path}")
    print(f"  summary_metrics: {summary_path}")
    print(f"  exp_parity_plot: {plot_path}")


if __name__ == "__main__":
    main()
