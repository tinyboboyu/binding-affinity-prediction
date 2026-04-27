"""Aggregate Baseline 4 run predictions and generate publication-style plots."""

from __future__ import annotations

from evaluate_baseline3_runs import (
    build_parser,
    compute_summary,
    load_predictions,
    plot_exp_parity,
    resolve_run_dirs,
    save_predictions_csv,
    save_summary_csv,
)


def main() -> None:
    parser = build_parser()
    parser.description = "Summarize Baseline 4 prediction CSVs and plot parity figures."
    parser.set_defaults(run_prefix="baseline4_")
    args = parser.parse_args()

    from pathlib import Path

    results_root = Path(args.results_root).expanduser().resolve()
    run_dirs = resolve_run_dirs(results_root, args.run_dirs, args.run_prefix)
    if not run_dirs:
        raise SystemExit(f"No Baseline 4 run directories found under {results_root}")

    rows = load_predictions(results_root, run_dirs)
    summary_rows = compute_summary(rows)

    merged_path = results_root / "baseline4_merged_predictions.csv"
    summary_path = results_root / "baseline4_summary_metrics.csv"
    plot_path = results_root / "baseline4_exp_parity_plot.png"

    save_predictions_csv(rows, merged_path)
    save_summary_csv(summary_rows, summary_path)

    exp_metrics = next(row for row in summary_rows if row["target"] == "exp")
    plot_exp_parity(rows, exp_metrics, plot_path, dpi=args.dpi)

    print("Baseline 4 evaluation finished")
    print(f"  runs: {run_dirs}")
    print(f"  merged_predictions: {merged_path}")
    print(f"  summary_metrics: {summary_path}")
    print(f"  exp_parity_plot: {plot_path}")


if __name__ == "__main__":
    main()
