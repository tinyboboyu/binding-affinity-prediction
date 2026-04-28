"""Crystal-only inference for a new protein-ligand complex across Baselines 1-4."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean, pstdev

import torch
from torch_geometric.data import Batch

from binding_graph_preprocessing import ComplexPreprocessorConfig
from binding_graph_preprocessing.graph import build_complex_graph, to_pyg_data
from binding_graph_preprocessing.structure import build_graph_components, parse_pdb_file
from md_frame_labels import PB_TARGET_KEYS
from model_baseline1 import Baseline1ExpModel
from model_baseline2_pb import Baseline2PBModel
from model_baseline3 import Baseline3PBModel
from model_baseline4 import Baseline4PBModel


BASELINE_CHOICES = ["baseline1", "baseline2_pb", "baseline3", "baseline4"]
MODEL_TYPE_ALIASES = {
    "baseline1": "baseline1",
    "b1": "baseline1",
    "baseline2": "baseline2_pb",
    "baseline2_pb": "baseline2_pb",
    "b2": "baseline2_pb",
    "b2pb": "baseline2_pb",
    "baseline3": "baseline3",
    "b3": "baseline3",
    "baseline4": "baseline4",
    "b4": "baseline4",
}
PB_OUTPUT_COLUMNS = [f"pred_avg_pb_{key}" for key in PB_TARGET_KEYS]
R_KCAL_PER_MOL_K = 0.0019872041
DEFAULT_TEMPERATURE_K = 298.15


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Predict experimental Delta G for a new crystal complex using saved baseline checkpoints."
    )
    parser.add_argument("--pdb_path", required=True, help="Path to the new crystal complex PDB file.")
    parser.add_argument("--sample_id", required=True, help="Identifier to use in output files.")
    parser.add_argument("--ligand_resname", required=True, help="Ligand residue name in the PDB file.")
    parser.add_argument("--ligand_resid", type=int, required=True, help="Ligand residue number in the PDB file.")
    parser.add_argument("--ligand_chain", default=None, help="Optional ligand chain identifier.")
    parser.add_argument("--output_dir", required=True, help="Directory receiving the graph and prediction CSVs.")
    parser.add_argument("--results_root", default="../results/training_runs", help="Directory containing training runs.")
    parser.add_argument("--baselines", nargs="+", choices=BASELINE_CHOICES, default=BASELINE_CHOICES)
    parser.add_argument(
        "--checkpoint_dirs",
        nargs="*",
        default=None,
        help="Optional explicit checkpoint directories. Baseline names are inferred from directory names.",
    )
    parser.add_argument("--pocket_cutoff", type=float, default=5.0)
    parser.add_argument("--protein_edge_cutoff", type=float, default=4.5)
    parser.add_argument("--ligand_protein_edge_cutoff", type=float, default=5.0)
    parser.add_argument("--device", default=None)
    return parser


def select_device(device_arg: str | None) -> torch.device:
    if device_arg is not None:
        return torch.device(device_arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def normalize_model_type(model_type: str) -> str:
    normalized = model_type.strip().lower().replace("-", "_")
    if normalized not in MODEL_TYPE_ALIASES:
        raise ValueError(f"Unsupported model_type={model_type!r}. Supported values: {sorted(MODEL_TYPE_ALIASES)}")
    return MODEL_TYPE_ALIASES[normalized]


def build_unlabeled_graph_from_pdb(
    pdb_path: str | Path,
    sample_id: str,
    ligand_resname: str,
    ligand_resid: int,
    ligand_chain: str | None = None,
    pocket_cutoff: float = 5.0,
    protein_edge_cutoff: float = 4.5,
    ligand_protein_edge_cutoff: float = 5.0,
):
    config = ComplexPreprocessorConfig(
        ligand_resname=ligand_resname,
        ligand_resid=ligand_resid,
        ligand_chain=ligand_chain,
        pocket_cutoff=pocket_cutoff,
        protein_edge_cutoff=protein_edge_cutoff,
        ligand_protein_edge_cutoff=ligand_protein_edge_cutoff,
        keep_hydrogens=True,
        remove_water=True,
        keep_metals=True,
    )
    resolved_pdb_path = Path(pdb_path).expanduser().resolve()
    parsed = parse_pdb_file(resolved_pdb_path)
    components = build_graph_components(
        parsed=parsed,
        ligand_resname=config.ligand_resname,
        ligand_resid=config.ligand_resid,
        ligand_chain=config.ligand_chain,
        pocket_cutoff=config.pocket_cutoff,
        protein_edge_cutoff=config.protein_edge_cutoff,
        keep_hydrogens=config.keep_hydrogens,
        remove_water=config.remove_water,
        keep_metals=config.keep_metals,
    )
    graph_dict = build_complex_graph(
        components=components,
        sample_id=sample_id,
        pdb_path=str(resolved_pdb_path),
        mmpbsa_path="not_used_for_inference",
        y_exp=0.0,
        y_vdw=0.0,
        y_elec=0.0,
        y_polar=0.0,
        y_nonpolar=0.0,
        delta_total_kcal=0.0,
        protein_edge_cutoff=config.protein_edge_cutoff,
        ligand_protein_edge_cutoff=config.ligand_protein_edge_cutoff,
    )
    return to_pyg_data(graph_dict)


def build_unlabeled_graph(args: argparse.Namespace):
    return build_unlabeled_graph_from_pdb(
        pdb_path=args.pdb_path,
        sample_id=args.sample_id,
        ligand_resname=args.ligand_resname,
        ligand_resid=args.ligand_resid,
        ligand_chain=args.ligand_chain,
        pocket_cutoff=args.pocket_cutoff,
        protein_edge_cutoff=args.protein_edge_cutoff,
        ligand_protein_edge_cutoff=args.ligand_protein_edge_cutoff,
    )


def save_graph(graph, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(graph, output_path)


def infer_baseline_from_path(path: Path) -> str | None:
    name = path.name
    for baseline in ["baseline2_pb", "baseline1", "baseline3", "baseline4"]:
        if name.startswith(f"{baseline}_"):
            return baseline
    return None


def resolve_checkpoint_dirs(results_root: Path, baselines: list[str], explicit_dirs: list[str] | None) -> dict[str, list[Path]]:
    resolved = {baseline: [] for baseline in baselines}
    if explicit_dirs:
        for text in explicit_dirs:
            path = Path(text).expanduser().resolve()
            baseline = infer_baseline_from_path(path)
            if baseline is None or baseline not in resolved:
                raise ValueError(f"Could not infer a selected baseline from checkpoint directory: {path}")
            if not (path / "best_model.pt").exists():
                raise FileNotFoundError(f"Missing best_model.pt in checkpoint directory: {path}")
            resolved[baseline].append(path)
        return resolved

    for baseline in baselines:
        pattern = f"{baseline}_rotating_train_val_test_round_*"
        dirs = [
            path
            for path in sorted(results_root.glob(pattern))
            if path.is_dir() and (path / "best_model.pt").exists()
        ]
        if not dirs:
            raise FileNotFoundError(f"No checkpoint directories found for {baseline} under {results_root}")
        resolved[baseline] = dirs
    return resolved


def load_normalization_stats(checkpoint_dir: Path) -> dict[str, object]:
    stats_path = checkpoint_dir / "label_normalization_stats.json"
    return load_normalization_stats_file(stats_path)


def load_normalization_stats_file(stats_path: str | Path) -> dict[str, object]:
    stats_path = Path(stats_path).expanduser().resolve()
    if not stats_path.exists():
        raise FileNotFoundError(f"Missing normalization stats: {stats_path}")
    return json.loads(stats_path.read_text(encoding="utf-8"))


def require_stats_keys(stats: dict[str, object], keys: list[str], stats_path: str | Path) -> None:
    missing = [key for key in keys if key not in stats]
    if missing:
        raise KeyError(f"Normalization stats file {stats_path} is missing required keys: {missing}")


def denormalize_exp(tensor: torch.Tensor, stats: dict[str, object]) -> torch.Tensor:
    if not bool(stats.get("enabled", True)):
        return tensor
    mean_tensor = torch.tensor(stats["exp_mean"], dtype=tensor.dtype, device=tensor.device)
    std_tensor = torch.tensor(stats["exp_std"], dtype=tensor.dtype, device=tensor.device)
    return tensor * std_tensor + mean_tensor


def denormalize_avg_pb(tensor: torch.Tensor, stats: dict[str, object]) -> torch.Tensor:
    if not bool(stats.get("enabled", True)):
        return tensor
    mean_tensor = torch.tensor(stats["avg_pb_mean"], dtype=tensor.dtype, device=tensor.device)
    std_tensor = torch.tensor(stats["avg_pb_std"], dtype=tensor.dtype, device=tensor.device)
    return tensor * std_tensor + mean_tensor


def build_model(baseline: str, model_config: dict[str, object]):
    kwargs = {
        "in_dim": int(model_config["in_dim"]),
        "hidden_dim": int(model_config.get("hidden_dim", 64)),
        "num_layers": int(model_config.get("num_layers", 2)),
        "dropout": float(model_config.get("dropout", 0.0)),
    }
    if baseline == "baseline1":
        return Baseline1ExpModel(**kwargs)
    if baseline == "baseline2_pb":
        return Baseline2PBModel(**kwargs)
    if baseline == "baseline3":
        return Baseline3PBModel(**kwargs)
    if baseline == "baseline4":
        return Baseline4PBModel(**kwargs)
    raise ValueError(f"Unsupported baseline: {baseline}")


def predict_graph_with_checkpoint(
    model_type: str,
    checkpoint_path: str | Path,
    graph,
    device: torch.device,
    normalization_stats_path: str | Path | None = None,
) -> dict[str, object]:
    baseline = normalize_model_type(model_type)
    checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = build_model(baseline, checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    batch = Batch.from_data_list([graph]).to(device)
    row: dict[str, object] = {
        "baseline": baseline,
        "checkpoint_dir": str(checkpoint_path.parent),
        "checkpoint_path": str(checkpoint_path),
        "normalization_stats_path": "" if normalization_stats_path is None else str(Path(normalization_stats_path).expanduser().resolve()),
        "best_epoch": checkpoint.get("best_epoch", ""),
    }
    for column in PB_OUTPUT_COLUMNS:
        row[column] = ""

    with torch.no_grad():
        if baseline == "baseline1":
            outputs = model(batch)
            row["pred_exp"] = float(outputs["pred_exp"].detach().cpu().view(-1)[0].item())
        elif baseline == "baseline2_pb":
            if normalization_stats_path is None:
                raise FileNotFoundError("Baseline 2-PB requires label_normalization_stats.json for de-normalization.")
            outputs = model(batch)
            stats = load_normalization_stats_file(normalization_stats_path)
            require_stats_keys(stats, ["exp_mean", "exp_std", "avg_pb_mean", "avg_pb_std"], normalization_stats_path)
            pred_exp = denormalize_exp(outputs["pred_exp_norm"], stats)
            pred_avg_pb = denormalize_avg_pb(outputs["pred_avg_pb_norm"], stats)
            row["pred_exp"] = float(pred_exp.detach().cpu().view(-1)[0].item())
            for index, key in enumerate(PB_TARGET_KEYS):
                row[f"pred_avg_pb_{key}"] = float(pred_avg_pb.detach().cpu().view(-1, len(PB_TARGET_KEYS))[0, index].item())
        elif baseline == "baseline3":
            outputs = model(batch, frame_batch=None)
            row["pred_exp"] = float(outputs["pred_exp"].detach().cpu().view(-1)[0].item())
            pred_avg_pb = outputs["pred_avg_pb"].detach().cpu().view(-1, len(PB_TARGET_KEYS))
            for index, key in enumerate(PB_TARGET_KEYS):
                row[f"pred_avg_pb_{key}"] = float(pred_avg_pb[0, index].item())
        elif baseline == "baseline4":
            outputs = model(batch, frame_batch=None)
            row["pred_exp"] = float(outputs["pred_exp"].detach().cpu().view(-1)[0].item())
            pred_avg_pb = outputs["pred_avg_pb"].detach().cpu().view(-1, len(PB_TARGET_KEYS))
            for index, key in enumerate(PB_TARGET_KEYS):
                row[f"pred_avg_pb_{key}"] = float(pred_avg_pb[0, index].item())
    return row


def predict_one_checkpoint(baseline: str, checkpoint_dir: Path, graph, device: torch.device) -> dict[str, object]:
    checkpoint_path = checkpoint_dir / "best_model.pt"
    stats_path = checkpoint_dir / "label_normalization_stats.json"
    normalization_stats_path = stats_path if stats_path.exists() else None
    return predict_graph_with_checkpoint(
        model_type=baseline,
        checkpoint_path=checkpoint_path,
        graph=graph,
        device=device,
        normalization_stats_path=normalization_stats_path,
    )


def delta_g_kcal_to_kj(delta_g_kcal: float) -> float:
    return float(delta_g_kcal) * 4.184


def delta_g_kcal_to_kd(delta_g_kcal: float, temperature: float = DEFAULT_TEMPERATURE_K) -> float:
    try:
        return math.exp(float(delta_g_kcal) / (R_KCAL_PER_MOL_K * temperature))
    except OverflowError:
        return float("inf")


def format_kd(kd_molar: float) -> str:
    if not math.isfinite(kd_molar):
        return "not finite"
    if kd_molar < 1e-6:
        return f"{kd_molar * 1e9:.3g} nM"
    if kd_molar < 1e-3:
        return f"{kd_molar * 1e6:.3g} µM"
    if kd_molar < 1.0:
        return f"{kd_molar * 1e3:.3g} mM"
    return f"{kd_molar:.3g} M"


def summarize_graph(graph) -> dict[str, int]:
    edge_attr = getattr(graph, "edge_attr", None)
    edge_dim = int(edge_attr.size(-1)) if edge_attr is not None and edge_attr.numel() > 0 else 0
    return {
        "num_nodes": int(graph.x.size(0)),
        "num_edges": int(graph.edge_index.size(1)),
        "num_ligand_atoms": int(graph.ligand_mask.sum().item()),
        "num_protein_atoms": int(graph.protein_mask.sum().item()),
        "num_metal_atoms": int(graph.metal_mask.sum().item()),
        "node_feature_dim": int(graph.x.size(1)),
        "edge_feature_dim": edge_dim,
    }


def row_to_pred_avg_pb(row: dict[str, object]) -> dict[str, float] | None:
    values: dict[str, float] = {}
    for key in PB_TARGET_KEYS:
        value = row.get(f"pred_avg_pb_{key}", "")
        if value == "":
            return None
        values[key] = float(value)
    return values


def predict_from_pdb(
    pdb_path: str | Path,
    ligand_resname: str,
    ligand_resid: int,
    ligand_chain: str | None,
    model_type: str,
    checkpoint_path: str | Path,
    normalization_stats_path: str | Path | None = None,
    device: str | torch.device | None = None,
    sample_id: str = "uploaded_complex",
    pocket_cutoff: float = 5.0,
    protein_edge_cutoff: float = 4.5,
    ligand_protein_edge_cutoff: float = 5.0,
    save_graph_path: str | Path | None = None,
) -> dict[str, object]:
    selected_device = device if isinstance(device, torch.device) else select_device(device)
    baseline = normalize_model_type(model_type)
    graph = build_unlabeled_graph_from_pdb(
        pdb_path=pdb_path,
        sample_id=sample_id,
        ligand_resname=ligand_resname,
        ligand_resid=ligand_resid,
        ligand_chain=ligand_chain,
        pocket_cutoff=pocket_cutoff,
        protein_edge_cutoff=protein_edge_cutoff,
        ligand_protein_edge_cutoff=ligand_protein_edge_cutoff,
    )
    if save_graph_path is not None:
        save_graph(graph, Path(save_graph_path).expanduser().resolve())

    row = predict_graph_with_checkpoint(
        model_type=baseline,
        checkpoint_path=checkpoint_path,
        graph=graph,
        device=selected_device,
        normalization_stats_path=normalization_stats_path,
    )
    pred_exp_kcal = float(row["pred_exp"])
    pred_exp_kj = delta_g_kcal_to_kj(pred_exp_kcal)
    kd_molar = delta_g_kcal_to_kd(pred_exp_kcal)
    if not math.isfinite(pred_exp_kcal):
        raise ValueError(f"Non-finite predicted experimental Delta G: {pred_exp_kcal}")
    return {
        "sample_id": sample_id,
        "model_type": baseline,
        "pred_exp_kcal": pred_exp_kcal,
        "pred_exp_kj": pred_exp_kj,
        "estimated_kd_molar": kd_molar,
        "estimated_kd_display": format_kd(kd_molar),
        "pred_avg_pb": row_to_pred_avg_pb(row),
        "graph_summary": summarize_graph(graph),
        "best_epoch": row.get("best_epoch", ""),
        "checkpoint_path": str(Path(checkpoint_path).expanduser().resolve()),
        "normalization_stats_path": "" if normalization_stats_path is None else str(Path(normalization_stats_path).expanduser().resolve()),
    }


def write_checkpoint_predictions(path: Path, sample_id: str, rows: list[dict[str, object]]) -> None:
    fieldnames = ["sample_id", "baseline", "checkpoint_dir", "best_epoch", "pred_exp", *PB_OUTPUT_COLUMNS]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            output_row = {"sample_id": sample_id, **row}
            writer.writerow(output_row)


def build_ensemble_summary(sample_id: str, rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[str, list[float]] = {}
    for row in rows:
        grouped.setdefault(str(row["baseline"]), []).append(float(row["pred_exp"]))

    summary_rows: list[dict[str, object]] = []
    for baseline in sorted(grouped):
        values = grouped[baseline]
        summary_rows.append(
            {
                "sample_id": sample_id,
                "baseline": baseline,
                "n_checkpoints": len(values),
                "pred_exp_mean": mean(values),
                "pred_exp_std": pstdev(values) if len(values) > 1 else 0.0,
                "pred_exp_min": min(values),
                "pred_exp_max": max(values),
            }
        )
    return summary_rows


def write_ensemble_summary(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = [
        "sample_id",
        "baseline",
        "n_checkpoints",
        "pred_exp_mean",
        "pred_exp_std",
        "pred_exp_min",
        "pred_exp_max",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def validate_predictions(rows: list[dict[str, object]]) -> None:
    for row in rows:
        value = float(row["pred_exp"])
        if not math.isfinite(value):
            raise ValueError(f"Non-finite pred_exp from {row['checkpoint_dir']}: {value}")


def main() -> None:
    args = build_parser().parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    results_root = Path(args.results_root).expanduser().resolve()
    device = select_device(args.device)

    graph = build_unlabeled_graph(args)
    graph_path = output_dir / f"{args.sample_id}.pt"
    save_graph(graph, graph_path)

    checkpoint_dirs = resolve_checkpoint_dirs(results_root, args.baselines, args.checkpoint_dirs)
    prediction_rows: list[dict[str, object]] = []
    for baseline in args.baselines:
        for checkpoint_dir in checkpoint_dirs[baseline]:
            prediction_rows.append(predict_one_checkpoint(baseline, checkpoint_dir, graph, device))

    validate_predictions(prediction_rows)
    checkpoint_csv = output_dir / "checkpoint_predictions.csv"
    ensemble_csv = output_dir / "ensemble_summary.csv"
    summary_rows = build_ensemble_summary(args.sample_id, prediction_rows)
    write_checkpoint_predictions(checkpoint_csv, args.sample_id, prediction_rows)
    write_ensemble_summary(ensemble_csv, summary_rows)

    print("New complex inference finished")
    print(f"  sample_id: {args.sample_id}")
    print(f"  graph: {graph_path}")
    print(
        "  graph_summary: "
        f"nodes={int(graph.x.size(0))} edges={int(graph.edge_index.size(1))} "
        f"ligand_atoms={int(graph.ligand_mask.sum().item())} "
        f"protein_atoms={int(graph.protein_mask.sum().item())} "
        f"metal_atoms={int(graph.metal_mask.sum().item())}"
    )
    print(f"  checkpoint_predictions: {checkpoint_csv}")
    print(f"  ensemble_summary: {ensemble_csv}")
    print("  predictions:")
    for row in summary_rows:
        print(
            f"    {row['baseline']}: "
            f"mean={float(row['pred_exp_mean']):.4f} "
            f"std={float(row['pred_exp_std']):.4f} "
            f"n={int(row['n_checkpoints'])}"
        )


if __name__ == "__main__":
    main()
