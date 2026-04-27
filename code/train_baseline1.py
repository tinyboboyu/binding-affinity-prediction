"""Baseline 1 training: crystal-only experimental binding free energy prediction."""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import torch
from torch import nn
from torch.optim import Adam
from torch_geometric.loader import DataLoader

from dataset import MMPBSAGraphDataset
from model_baseline1 import Baseline1ExpModel
from normalization_baseline3 import ExpLabelNormalizer
from splits_baseline3 import resolve_baseline3_split


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train Baseline 1 crystal-only experimental Delta G model.")
    parser.add_argument("--graph_dir", default="../data/MMPBSA/processed/graphs")
    parser.add_argument("--save_dir", default=None)
    parser.add_argument("--split_mode", choices=["rotating_train_val_test", "leave_one_out"], default="rotating_train_val_test")
    parser.add_argument("--split_round", type=int, default=1)
    parser.add_argument("--test_sample_id", default=None)
    parser.add_argument("--val_mode", choices=["none", "deterministic", "explicit"], default="deterministic")
    parser.add_argument("--val_sample_id", default=None)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--normalize_labels", type=str, default="true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--print_every", type=int, default=10)
    parser.add_argument("--device", default=None)
    return parser


def parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y"}:
        return True
    if normalized in {"0", "false", "no", "n"}:
        return False
    raise ValueError(f"Cannot parse boolean value from {value!r}")


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_device(device_arg: str | None) -> torch.device:
    if device_arg is not None:
        return torch.device(device_arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def resolve_default_save_dir(args: argparse.Namespace, split_info: dict[str, object]) -> Path:
    if args.save_dir:
        return Path(args.save_dir)
    if args.split_mode == "rotating_train_val_test":
        run_name = (
            f"round_{args.split_round}_"
            f"val_{split_info['val_sample_ids'][0]}_"
            f"test_{split_info['test_sample_ids'][0]}"
        )
    else:
        run_name = f"test_{args.test_sample_id}_{args.val_mode}"
        if split_info["val_sample_ids"]:
            run_name = f"{run_name}_val_{split_info['val_sample_ids'][0]}"
    return Path("../results/training_runs") / f"baseline1_{args.split_mode}_{run_name}"


def save_json(payload: dict[str, object], path: str | Path) -> None:
    Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def graph_debug_row(sample_id: str, graph, graph_path: Path) -> dict[str, object]:
    edge_attr = getattr(graph, "edge_attr", None)
    edge_dim = int(edge_attr.size(-1)) if edge_attr is not None and edge_attr.numel() > 0 else 0
    return {
        "sample_id": sample_id,
        "structure_name": "crystal",
        "file_path": str(graph_path),
        "num_ligand_atoms": int(graph.ligand_mask.sum().item()),
        "num_pocket_atoms": int(graph.protein_mask.sum().item()),
        "num_retained_metals": int(graph.metal_mask.sum().item()),
        "num_nodes": int(graph.x.size(0)),
        "num_edges": int(graph.edge_index.size(1)),
        "node_dim": int(graph.x.size(1)),
        "edge_dim": edge_dim,
    }


def save_debug_summary(dataset: MMPBSAGraphDataset, csv_path: str | Path) -> None:
    fieldnames = [
        "sample_id",
        "structure_name",
        "file_path",
        "num_ligand_atoms",
        "num_pocket_atoms",
        "num_retained_metals",
        "num_nodes",
        "num_edges",
        "node_dim",
        "edge_dim",
    ]
    rows = [
        graph_debug_row(sample_id, dataset[index], graph_path)
        for index, (sample_id, graph_path) in enumerate(zip(dataset.sample_ids, dataset.graph_paths))
    ]
    with Path(csv_path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_debug_summary(dataset: MMPBSAGraphDataset) -> None:
    print("Graph construction debug summary")
    for index, (sample_id, graph_path) in enumerate(zip(dataset.sample_ids, dataset.graph_paths)):
        row = graph_debug_row(sample_id, dataset[index], graph_path)
        print(
            f"  sample={row['sample_id']} structure={row['structure_name']} "
            f"path={row['file_path']} ligand_atoms={row['num_ligand_atoms']} "
            f"pocket_atoms={row['num_pocket_atoms']} metals={row['num_retained_metals']} "
            f"nodes={row['num_nodes']} edges={row['num_edges']} "
            f"node_dim={row['node_dim']} edge_dim={row['edge_dim']}"
        )


def compute_loss(outputs: dict[str, torch.Tensor], batch, normalizer: ExpLabelNormalizer) -> tuple[torch.Tensor, dict[str, float]]:
    mse = nn.MSELoss()
    target_exp = batch.y_exp.view(-1, 1).float()
    loss_exp = mse(normalizer.normalize_exp(outputs["pred_exp"]), normalizer.normalize_exp(target_exp))
    return loss_exp, {
        "L_exp": float(loss_exp.detach().cpu().item()),
        "L_total": float(loss_exp.detach().cpu().item()),
    }


def train_one_epoch(model, loader, optimizer, device, normalizer: ExpLabelNormalizer) -> dict[str, float]:
    model.train()
    totals = {"L_exp": 0.0, "L_total": 0.0}
    total_graphs = 0

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        outputs = model(batch)
        loss, diagnostics = compute_loss(outputs, batch, normalizer)
        loss.backward()
        optimizer.step()

        num_graphs = int(batch.num_graphs)
        total_graphs += num_graphs
        for key in totals:
            totals[key] += diagnostics[key] * num_graphs

    return {key: value / total_graphs for key, value in totals.items()}


def evaluate(model, loader, device, normalizer: ExpLabelNormalizer, split_name: str) -> dict[str, object]:
    model.eval()
    total_loss = 0.0
    total_graphs = 0
    sample_ids: list[str] = []
    pred_exp: list[float] = []
    true_exp: list[float] = []

    mse = nn.MSELoss()
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            outputs = model(batch)
            target_exp = batch.y_exp.view(-1, 1).float()
            pred_exp_norm = normalizer.normalize_exp(outputs["pred_exp"])
            true_exp_norm = normalizer.normalize_exp(target_exp)
            loss_exp = mse(pred_exp_norm, true_exp_norm)

            denorm_pred = normalizer.denormalize_exp(pred_exp_norm)
            denorm_true = normalizer.denormalize_exp(true_exp_norm)

            num_graphs = int(batch.num_graphs)
            total_graphs += num_graphs
            total_loss += float(loss_exp.detach().cpu().item()) * num_graphs
            sample_ids.extend(list(batch.sample_id))
            pred_exp.extend(denorm_pred.detach().cpu().view(-1).tolist())
            true_exp.extend(denorm_true.detach().cpu().view(-1).tolist())

    return {
        "split": split_name,
        "L_exp": total_loss / total_graphs,
        "L_total": total_loss / total_graphs,
        "sample_ids": sample_ids,
        "pred_exp": pred_exp,
        "true_exp": true_exp,
    }


def save_train_log(rows: list[dict[str, object]], csv_path: str | Path) -> None:
    fieldnames = ["epoch", "train_L_exp", "val_L_exp", "best_val_L_exp", "best_epoch"]
    with Path(csv_path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_predictions(results_by_split: list[dict[str, object]], csv_path: str | Path) -> None:
    fieldnames = ["sample_id", "split", "true_exp", "pred_exp", "abs_error"]
    with Path(csv_path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for results in results_by_split:
            for sample_id, true_exp, pred_exp in zip(results["sample_ids"], results["true_exp"], results["pred_exp"]):
                writer.writerow(
                    {
                        "sample_id": sample_id,
                        "split": results["split"],
                        "true_exp": true_exp,
                        "pred_exp": pred_exp,
                        "abs_error": abs(pred_exp - true_exp),
                    }
                )


def main() -> None:
    args = build_parser().parse_args()
    normalize_labels = parse_bool(args.normalize_labels)
    set_seed(args.seed)
    device = select_device(args.device)

    split_info = resolve_baseline3_split(
        split_mode=args.split_mode,
        split_round=args.split_round,
        test_sample_id=args.test_sample_id,
        val_mode=args.val_mode,
        val_sample_id=args.val_sample_id,
    )

    graph_dir = Path(args.graph_dir)
    all_sample_ids = list(dict.fromkeys(split_info["train_sample_ids"] + split_info["val_sample_ids"] + split_info["test_sample_ids"]))
    debug_dataset = MMPBSAGraphDataset(graph_dir, sample_ids=all_sample_ids)
    train_dataset = MMPBSAGraphDataset(graph_dir, sample_ids=split_info["train_sample_ids"])
    val_dataset = MMPBSAGraphDataset(graph_dir, sample_ids=split_info["val_sample_ids"]) if split_info["val_sample_ids"] else None
    test_dataset = MMPBSAGraphDataset(graph_dir, sample_ids=split_info["test_sample_ids"])

    print_debug_summary(debug_dataset)

    normalizer = ExpLabelNormalizer.from_training_graphs(
        [train_dataset[index] for index in range(len(train_dataset))],
        enabled=normalize_labels,
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False) if val_dataset is not None else None
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    in_dim = int(train_dataset[0].x.size(-1))
    model = Baseline1ExpModel(
        in_dim=in_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)
    optimizer = Adam(model.parameters(), lr=args.lr)

    save_dir = resolve_default_save_dir(args, split_info)
    save_dir.mkdir(parents=True, exist_ok=True)
    normalizer.save(save_dir / "label_normalization_stats.json")
    save_debug_summary(debug_dataset, save_dir / "graph_debug_summary.csv")
    save_json(split_info, save_dir / "split_info.json")
    save_json(
        {
            "graph_dir": str(graph_dir),
            "split_mode": args.split_mode,
            "split_round": args.split_round,
            "test_sample_id": args.test_sample_id,
            "val_mode": args.val_mode,
            "val_sample_id": args.val_sample_id,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "hidden_dim": args.hidden_dim,
            "num_layers": args.num_layers,
            "dropout": args.dropout,
            "normalize_labels": normalize_labels,
            "seed": args.seed,
            "target": "experimental_delta_g_only",
            "inference_inputs": ["crystal_graph"],
        },
        save_dir / "run_config.json",
    )

    train_log_rows: list[dict[str, object]] = []
    best_val_loss = float("inf")
    best_epoch = -1

    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(model, train_loader, optimizer, device=device, normalizer=normalizer)
        eval_loader = val_loader if val_loader is not None else train_loader
        val_results = evaluate(model, eval_loader, device=device, normalizer=normalizer, split_name="val")

        if val_results["L_exp"] < best_val_loss:
            best_val_loss = float(val_results["L_exp"])
            best_epoch = epoch
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "model_config": {
                        "in_dim": in_dim,
                        "hidden_dim": args.hidden_dim,
                        "num_layers": args.num_layers,
                        "dropout": args.dropout,
                    },
                    "split_info": split_info,
                    "best_epoch": best_epoch,
                    "best_val_L_exp": best_val_loss,
                    "normalize_labels": normalize_labels,
                },
                save_dir / "best_model.pt",
            )

        train_log_rows.append(
            {
                "epoch": epoch,
                "train_L_exp": train_metrics["L_exp"],
                "val_L_exp": val_results["L_exp"],
                "best_val_L_exp": best_val_loss,
                "best_epoch": best_epoch,
            }
        )

        if epoch == 1 or epoch % args.print_every == 0 or epoch == args.epochs:
            print(
                f"Epoch {epoch:4d} | "
                f"train_L_exp={train_metrics['L_exp']:.6f} "
                f"val_L_exp={val_results['L_exp']:.6f} "
                f"best_val_L_exp={best_val_loss:.6f} "
                f"best_epoch={best_epoch}"
            )

    save_train_log(train_log_rows, save_dir / "train_log.csv")

    checkpoint = torch.load(save_dir / "best_model.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    prediction_results: list[dict[str, object]] = []
    prediction_results.append(evaluate(model, train_loader, device=device, normalizer=normalizer, split_name="train"))
    if val_loader is not None:
        prediction_results.append(evaluate(model, val_loader, device=device, normalizer=normalizer, split_name="val"))
    prediction_results.append(evaluate(model, test_loader, device=device, normalizer=normalizer, split_name="test"))
    save_predictions(prediction_results, save_dir / "best_predictions.csv")

    print()
    print("Baseline 1 training finished")
    print(f"  split_mode: {args.split_mode}")
    print(f"  train_ids: {split_info['train_sample_ids']}")
    print(f"  val_ids: {split_info['val_sample_ids']}")
    print(f"  test_ids: {split_info['test_sample_ids']}")
    print(f"  normalize_labels: {normalize_labels}")
    print(f"  best_epoch: {best_epoch}")
    print(f"  best_val_L_exp: {best_val_loss:.6f}")
    print(f"  save_dir: {save_dir}")


if __name__ == "__main__":
    main()
