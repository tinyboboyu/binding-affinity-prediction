"""Baseline 2-PB training: crystal prediction with average PB auxiliary supervision."""

from __future__ import annotations

import argparse
import csv
import json
import random
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from torch_geometric.data import Batch

from dataset import load_graph
from md_frame_labels import (
    PB_AVERAGE_SOURCE_BLOCK,
    PB_AVERAGE_SOURCE_SECTION,
    PB_TARGET_KEYS,
    parse_average_pb_labels,
)
from model_baseline2_pb import Baseline2PBModel
from normalization_baseline3 import ExpAvgPBLabelNormalizer
from splits_baseline3 import resolve_baseline3_split


PB_CSV_NAMES = {
    "vdw": "vdw",
    "elec": "elec",
    "polar_solv": "polar_solv",
    "nonpolar_solv": "nonpolar_solv",
    "dispersion": "dispersion",
    "total": "total",
}


@dataclass
class Baseline2PBRecord:
    sample_id: str
    crystal_graph: object
    y_exp: float
    y_avg_pb: torch.Tensor
    graph_path: str
    mmpbsa_path: str
    source_section: str
    source_block: str


class Baseline2PBDataset(Dataset):
    def __init__(self, graph_dir: str | Path, mmpbsa_root: str | Path, sample_ids: list[str]) -> None:
        self.graph_dir = Path(graph_dir)
        self.mmpbsa_root = Path(mmpbsa_root)
        self.sample_ids = list(sample_ids)
        self.records = [self._load_record(sample_id) for sample_id in self.sample_ids]

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, object]:
        record = self.records[index]
        return {
            "sample_id": record.sample_id,
            "crystal_graph": record.crystal_graph,
            "y_exp": torch.tensor([[record.y_exp]], dtype=torch.float32),
            "y_avg_pb": record.y_avg_pb,
            "graph_path": record.graph_path,
            "mmpbsa_path": record.mmpbsa_path,
            "source_section": record.source_section,
            "source_block": record.source_block,
        }

    def _load_record(self, sample_id: str) -> Baseline2PBRecord:
        graph_path = self.graph_dir / f"{sample_id}.pt"
        mmpbsa_path = self.mmpbsa_root / sample_id / "mmpbsa.out"
        graph = load_graph(graph_path)
        avg_pb_labels = parse_average_pb_labels(mmpbsa_path)
        return Baseline2PBRecord(
            sample_id=sample_id,
            crystal_graph=graph,
            y_exp=float(graph.y_exp.view(-1)[0].item()),
            y_avg_pb=torch.tensor([avg_pb_labels[key] for key in PB_TARGET_KEYS], dtype=torch.float32),
            graph_path=str(graph_path),
            mmpbsa_path=str(mmpbsa_path),
            source_section=PB_AVERAGE_SOURCE_SECTION,
            source_block=PB_AVERAGE_SOURCE_BLOCK,
        )

    def record_lookup(self) -> dict[str, Baseline2PBRecord]:
        return {record.sample_id: record for record in self.records}

    def debug_rows(self) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        for record in self.records:
            graph = record.crystal_graph
            edge_attr = getattr(graph, "edge_attr", None)
            edge_dim = int(edge_attr.size(-1)) if edge_attr is not None and edge_attr.numel() > 0 else 0
            row: dict[str, object] = {
                "sample_id": record.sample_id,
                "structure_name": "crystal",
                "graph_path": record.graph_path,
                "mmpbsa_path": record.mmpbsa_path,
                "source_section": record.source_section,
                "source_block": record.source_block,
                "num_ligand_atoms": int(graph.ligand_mask.sum().item()),
                "num_pocket_atoms": int(graph.protein_mask.sum().item()),
                "num_retained_metals": int(graph.metal_mask.sum().item()),
                "num_nodes": int(graph.x.size(0)),
                "num_edges": int(graph.edge_index.size(1)),
                "node_dim": int(graph.x.size(1)),
                "edge_dim": edge_dim,
            }
            for index, key in enumerate(PB_TARGET_KEYS):
                row[f"avg_pb_{key}"] = float(record.y_avg_pb[index].item())
            rows.append(row)
        return rows


def collate_baseline2_pb_batch(items: list[dict[str, object]]) -> dict[str, object]:
    return {
        "sample_ids": [item["sample_id"] for item in items],
        "crystal_batch": Batch.from_data_list([item["crystal_graph"] for item in items]),
        "y_exp": torch.cat([item["y_exp"] for item in items], dim=0),
        "y_avg_pb": torch.stack([item["y_avg_pb"] for item in items], dim=0),
        "graph_paths": [item["graph_path"] for item in items],
        "mmpbsa_paths": [item["mmpbsa_path"] for item in items],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train Baseline 2-PB with average PB auxiliary supervision.")
    parser.add_argument("--graph_dir", default="../data/MMPBSA/processed/graphs")
    parser.add_argument("--mmpbsa_root", default="../data/MMPBSA")
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
    parser.add_argument("--lambda_avg", type=float, default=0.1)
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
    return Path("../results/training_runs") / f"baseline2_pb_{args.split_mode}_{run_name}"


def move_batch_to_device(batch: dict[str, object], device: torch.device) -> dict[str, object]:
    return {
        "sample_ids": batch["sample_ids"],
        "crystal_batch": batch["crystal_batch"].to(device),
        "y_exp": batch["y_exp"].to(device),
        "y_avg_pb": batch["y_avg_pb"].to(device),
        "graph_paths": batch["graph_paths"],
        "mmpbsa_paths": batch["mmpbsa_paths"],
    }


def compute_losses(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, object],
    normalizer: ExpAvgPBLabelNormalizer,
    lambda_avg: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    mse = nn.MSELoss()
    target_exp_norm = normalizer.normalize_exp(batch["y_exp"])
    target_avg_pb_norm = normalizer.normalize_avg_pb(batch["y_avg_pb"])
    loss_exp = mse(outputs["pred_exp_norm"], target_exp_norm)
    loss_avg_pb = mse(outputs["pred_avg_pb_norm"], target_avg_pb_norm)
    weighted_avg = lambda_avg * loss_avg_pb
    loss_total = loss_exp + weighted_avg
    return loss_total, {
        "L_exp": float(loss_exp.detach().cpu().item()),
        "L_avg_pb": float(loss_avg_pb.detach().cpu().item()),
        "weighted_L_avg_pb": float(weighted_avg.detach().cpu().item()),
        "L_total": float(loss_total.detach().cpu().item()),
    }


def train_one_epoch(model, loader, optimizer, device, normalizer: ExpAvgPBLabelNormalizer, lambda_avg: float):
    model.train()
    totals = {key: 0.0 for key in ["L_exp", "L_avg_pb", "weighted_L_avg_pb", "L_total"]}
    total_graphs = 0

    for raw_batch in loader:
        batch = move_batch_to_device(raw_batch, device)
        optimizer.zero_grad()
        outputs = model(batch["crystal_batch"])
        loss, diagnostics = compute_losses(outputs, batch, normalizer, lambda_avg=lambda_avg)
        loss.backward()
        optimizer.step()

        num_graphs = int(batch["y_exp"].size(0))
        total_graphs += num_graphs
        for key in totals:
            totals[key] += diagnostics[key] * num_graphs

    return {key: value / total_graphs for key, value in totals.items()}


def evaluate(model, loader, device, normalizer: ExpAvgPBLabelNormalizer, lambda_avg: float, split_name: str):
    model.eval()
    totals = {key: 0.0 for key in ["L_exp", "L_avg_pb", "weighted_L_avg_pb", "L_total"]}
    total_graphs = 0
    sample_ids: list[str] = []
    pred_exp: list[float] = []
    true_exp: list[float] = []
    pred_avg_pb: list[list[float]] = []
    true_avg_pb: list[list[float]] = []

    with torch.no_grad():
        for raw_batch in loader:
            batch = move_batch_to_device(raw_batch, device)
            outputs = model(batch["crystal_batch"])
            _, diagnostics = compute_losses(outputs, batch, normalizer, lambda_avg=lambda_avg)

            denorm_pred_exp = normalizer.denormalize_exp(outputs["pred_exp_norm"])
            denorm_true_exp = normalizer.denormalize_exp(normalizer.normalize_exp(batch["y_exp"]))
            denorm_pred_avg_pb = normalizer.denormalize_avg_pb(outputs["pred_avg_pb_norm"])
            denorm_true_avg_pb = normalizer.denormalize_avg_pb(normalizer.normalize_avg_pb(batch["y_avg_pb"]))

            num_graphs = int(batch["y_exp"].size(0))
            total_graphs += num_graphs
            for key in totals:
                totals[key] += diagnostics[key] * num_graphs

            sample_ids.extend(batch["sample_ids"])
            pred_exp.extend(denorm_pred_exp.detach().cpu().view(-1).tolist())
            true_exp.extend(denorm_true_exp.detach().cpu().view(-1).tolist())
            pred_avg_pb.extend(denorm_pred_avg_pb.detach().cpu().tolist())
            true_avg_pb.extend(denorm_true_avg_pb.detach().cpu().tolist())

    averaged = {key: value / total_graphs for key, value in totals.items()}
    averaged.update(
        {
            "split": split_name,
            "sample_ids": sample_ids,
            "pred_exp": pred_exp,
            "true_exp": true_exp,
            "pred_avg_pb": pred_avg_pb,
            "true_avg_pb": true_avg_pb,
        }
    )
    return averaged


def save_train_log(rows: list[dict[str, object]], csv_path: str | Path) -> None:
    fieldnames = [
        "epoch",
        "train_L_exp",
        "train_L_avg_pb",
        "train_weighted_L_avg_pb",
        "train_L_total",
        "val_L_exp",
        "val_L_avg_pb",
        "val_weighted_L_avg_pb",
        "val_L_total",
        "best_val_L_exp",
        "best_epoch",
    ]
    with Path(csv_path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_predictions(results_by_split: list[dict[str, object]], csv_path: str | Path) -> None:
    fieldnames = ["sample_id", "split", "true_exp", "pred_exp", "abs_error"]
    fieldnames.extend([f"true_avg_pb_{PB_CSV_NAMES[key]}" for key in PB_TARGET_KEYS])
    fieldnames.extend([f"pred_avg_pb_{PB_CSV_NAMES[key]}" for key in PB_TARGET_KEYS])

    with Path(csv_path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for results in results_by_split:
            for index, sample_id in enumerate(results["sample_ids"]):
                row = {
                    "sample_id": sample_id,
                    "split": results["split"],
                    "true_exp": results["true_exp"][index],
                    "pred_exp": results["pred_exp"][index],
                    "abs_error": abs(results["pred_exp"][index] - results["true_exp"][index]),
                }
                for pb_index, key in enumerate(PB_TARGET_KEYS):
                    csv_name = PB_CSV_NAMES[key]
                    row[f"true_avg_pb_{csv_name}"] = results["true_avg_pb"][index][pb_index]
                    row[f"pred_avg_pb_{csv_name}"] = results["pred_avg_pb"][index][pb_index]
                writer.writerow(row)


def save_debug_summary(rows: list[dict[str, object]], csv_path: str | Path) -> None:
    fieldnames = [
        "sample_id",
        "structure_name",
        "graph_path",
        "mmpbsa_path",
        "source_section",
        "source_block",
        "num_ligand_atoms",
        "num_pocket_atoms",
        "num_retained_metals",
        "num_nodes",
        "num_edges",
        "node_dim",
        "edge_dim",
    ]
    fieldnames.extend([f"avg_pb_{key}" for key in PB_TARGET_KEYS])
    with Path(csv_path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_debug_summary(rows: list[dict[str, object]]) -> None:
    print("Baseline 2-PB graph and label debug summary")
    print(f"  PB labels source_section={PB_AVERAGE_SOURCE_SECTION}")
    print(f"  PB labels source_block={PB_AVERAGE_SOURCE_BLOCK}")
    print("  PB labels are binding-level Differences labels, not Complex block labels.")
    for row in rows:
        pb_terms = " ".join(f"{key}={float(row[f'avg_pb_{key}']):.4f}" for key in PB_TARGET_KEYS)
        print(
            f"  sample={row['sample_id']} structure={row['structure_name']} "
            f"graph={row['graph_path']} mmpbsa={row['mmpbsa_path']} "
            f"nodes={row['num_nodes']} edges={row['num_edges']} "
            f"node_dim={row['node_dim']} edge_dim={row['edge_dim']} {pb_terms}"
        )


def save_json(payload: dict[str, object], path: str | Path) -> None:
    Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")


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
    mmpbsa_root = Path(args.mmpbsa_root)
    all_sample_ids = list(dict.fromkeys(split_info["train_sample_ids"] + split_info["val_sample_ids"] + split_info["test_sample_ids"]))

    debug_dataset = Baseline2PBDataset(graph_dir=graph_dir, mmpbsa_root=mmpbsa_root, sample_ids=all_sample_ids)
    train_dataset = Baseline2PBDataset(graph_dir=graph_dir, mmpbsa_root=mmpbsa_root, sample_ids=split_info["train_sample_ids"])
    val_dataset = Baseline2PBDataset(graph_dir=graph_dir, mmpbsa_root=mmpbsa_root, sample_ids=split_info["val_sample_ids"]) if split_info["val_sample_ids"] else None
    test_dataset = Baseline2PBDataset(graph_dir=graph_dir, mmpbsa_root=mmpbsa_root, sample_ids=split_info["test_sample_ids"])

    debug_rows = debug_dataset.debug_rows()
    print_debug_summary(debug_rows)

    normalizer = ExpAvgPBLabelNormalizer.from_training_records(
        list(train_dataset.record_lookup().values()),
        enabled=normalize_labels,
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_baseline2_pb_batch)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_baseline2_pb_batch) if val_dataset is not None else None
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_baseline2_pb_batch)

    in_dim = int(train_dataset[0]["crystal_graph"].x.size(-1))
    model = Baseline2PBModel(
        in_dim=in_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)
    optimizer = Adam(model.parameters(), lr=args.lr)

    save_dir = resolve_default_save_dir(args, split_info)
    save_dir.mkdir(parents=True, exist_ok=True)
    normalizer.save(save_dir / "label_normalization_stats.json")
    save_debug_summary(debug_rows, save_dir / "graph_debug_summary.csv")
    save_json(split_info, save_dir / "split_info.json")
    save_json(
        {
            "graph_dir": str(graph_dir),
            "mmpbsa_root": str(mmpbsa_root),
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
            "lambda_avg": args.lambda_avg,
            "normalize_labels": normalize_labels,
            "seed": args.seed,
            "pb_target_keys": PB_TARGET_KEYS,
            "pb_source_section": PB_AVERAGE_SOURCE_SECTION,
            "pb_source_block": PB_AVERAGE_SOURCE_BLOCK,
        },
        save_dir / "run_config.json",
    )

    train_log_rows: list[dict[str, object]] = []
    best_val_loss = float("inf")
    best_epoch = -1

    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device=device,
            normalizer=normalizer,
            lambda_avg=args.lambda_avg,
        )
        eval_loader = val_loader if val_loader is not None else train_loader
        val_results = evaluate(
            model,
            eval_loader,
            device=device,
            normalizer=normalizer,
            lambda_avg=args.lambda_avg,
            split_name="val",
        )

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
                    "lambda_avg": args.lambda_avg,
                    "pb_target_keys": PB_TARGET_KEYS,
                },
                save_dir / "best_model.pt",
            )

        train_log_rows.append(
            {
                "epoch": epoch,
                "train_L_exp": train_metrics["L_exp"],
                "train_L_avg_pb": train_metrics["L_avg_pb"],
                "train_weighted_L_avg_pb": train_metrics["weighted_L_avg_pb"],
                "train_L_total": train_metrics["L_total"],
                "val_L_exp": val_results["L_exp"],
                "val_L_avg_pb": val_results["L_avg_pb"],
                "val_weighted_L_avg_pb": val_results["weighted_L_avg_pb"],
                "val_L_total": val_results["L_total"],
                "best_val_L_exp": best_val_loss,
                "best_epoch": best_epoch,
            }
        )

        if epoch == 1 or epoch % args.print_every == 0 or epoch == args.epochs:
            print(
                f"Epoch {epoch:4d} | "
                f"train_L_exp={train_metrics['L_exp']:.6f} "
                f"train_L_avg_pb={train_metrics['L_avg_pb']:.6f} "
                f"lambda_avg*train_L_avg_pb={train_metrics['weighted_L_avg_pb']:.6f} "
                f"train_L_total={train_metrics['L_total']:.6f} | "
                f"val_L_exp={val_results['L_exp']:.6f} "
                f"best_val_L_exp={best_val_loss:.6f} "
                f"best_epoch={best_epoch}"
            )

    save_train_log(train_log_rows, save_dir / "train_log.csv")

    checkpoint = torch.load(save_dir / "best_model.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    prediction_results: list[dict[str, object]] = [
        evaluate(model, train_loader, device=device, normalizer=normalizer, lambda_avg=args.lambda_avg, split_name="train")
    ]
    if val_loader is not None:
        prediction_results.append(
            evaluate(model, val_loader, device=device, normalizer=normalizer, lambda_avg=args.lambda_avg, split_name="val")
        )
    prediction_results.append(
        evaluate(model, test_loader, device=device, normalizer=normalizer, lambda_avg=args.lambda_avg, split_name="test")
    )
    save_predictions(prediction_results, save_dir / "best_predictions.csv")

    print()
    print("Baseline 2-PB training finished")
    print(f"  split_mode: {args.split_mode}")
    print(f"  train_ids: {split_info['train_sample_ids']}")
    print(f"  val_ids: {split_info['val_sample_ids']}")
    print(f"  test_ids: {split_info['test_sample_ids']}")
    print(f"  normalize_labels: {normalize_labels}")
    print(f"  lambda_avg: {args.lambda_avg}")
    print(f"  best_epoch: {best_epoch}")
    print(f"  best_val_L_exp: {best_val_loss:.6f}")
    print(f"  save_dir: {save_dir}")


if __name__ == "__main__":
    main()
