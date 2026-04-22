"""Baseline 3 training: crystal prediction with MD-frame PB auxiliary supervision."""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from md_frame_dataset import Baseline3Dataset, collate_baseline3_batch
from model_baseline3 import Baseline3PBModel
from normalization_baseline3 import LabelNormalizer
from splits_baseline3 import resolve_baseline3_split


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train Baseline 3 with PB/MMPBSA frame supervision.")
    parser.add_argument("--graph_dir", default="../data/MMPBSA/processed/graphs")
    parser.add_argument("--raw_root_dir", default="../data/MMPBSA")
    parser.add_argument("--frame_root_dir", default="../data/md_frame_exports")
    parser.add_argument("--save_dir", default=None)
    parser.add_argument("--split_mode", choices=["rotating_train_val_test", "leave_one_out"], default="rotating_train_val_test")
    parser.add_argument("--split_round", type=int, default=1)
    parser.add_argument("--test_sample_id", default=None)
    parser.add_argument("--val_mode", choices=["none", "deterministic", "explicit"], default="deterministic")
    parser.add_argument("--val_sample_id", default=None)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--lambda_avg", type=float, default=0.1)
    parser.add_argument("--lambda_frame", type=float, default=0.03)
    parser.add_argument("--normalize_labels", type=str, default="true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--print_every", type=int, default=20)
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
    return Path("../results/training_runs") / f"baseline3_{args.split_mode}_{run_name}"


def print_debug_summary(rows: list[dict[str, object]]) -> None:
    print("Graph construction debug summary")
    for row in rows:
        print(
            f"  sample={row['sample_id']} structure={row['structure_name']} "
            f"path={row['file_path']} ligand={row['ligand_resname']} "
            f"ligand_atoms={row['num_ligand_atoms']} pocket_atoms={row['num_pocket_atoms']} "
            f"metals={row['num_retained_metals']} nodes={row['num_nodes']} edges={row['num_edges']} "
            f"node_dim={row['node_dim']} edge_dim={row['edge_dim']}"
        )


def move_batch_to_device(batch: dict[str, object], device: torch.device) -> dict[str, object]:
    return {
        "sample_ids": batch["sample_ids"],
        "crystal_batch": batch["crystal_batch"].to(device),
        "frame_batch": None if batch["frame_batch"] is None else batch["frame_batch"].to(device),
        "y_exp": batch["y_exp"].to(device),
        "y_avg_pb": batch["y_avg_pb"].to(device),
        "y_frame_pb": None if batch["y_frame_pb"] is None else batch["y_frame_pb"].to(device),
        "frame_paths": batch["frame_paths"],
    }


def compute_losses(
    outputs: dict[str, torch.Tensor | None],
    batch: dict[str, object],
    normalizer: LabelNormalizer,
    lambda_avg: float,
    lambda_frame: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    mse = nn.MSELoss()
    pred_exp = outputs["pred_exp"]
    pred_avg_pb = outputs["pred_avg_pb"]
    pred_frame_pb = outputs["pred_frame_pb"]

    target_exp = batch["y_exp"]
    target_avg_pb = batch["y_avg_pb"]
    target_frame_pb = batch["y_frame_pb"]

    loss_exp = mse(normalizer.normalize_exp(pred_exp), normalizer.normalize_exp(target_exp))
    loss_avg_pb = mse(normalizer.normalize_avg_pb(pred_avg_pb), normalizer.normalize_avg_pb(target_avg_pb))

    if pred_frame_pb is None or target_frame_pb is None or target_frame_pb.numel() == 0:
        loss_frame_pb = torch.tensor(0.0, device=pred_exp.device)
    else:
        loss_frame_pb = mse(
            normalizer.normalize_frame_pb(pred_frame_pb),
            normalizer.normalize_frame_pb(target_frame_pb),
        )

    weighted_avg = lambda_avg * loss_avg_pb
    weighted_frame = lambda_frame * loss_frame_pb
    loss_total = loss_exp + weighted_avg + weighted_frame
    diagnostics = {
        "L_exp": float(loss_exp.detach().cpu().item()),
        "L_avg_pb": float(loss_avg_pb.detach().cpu().item()),
        "L_frame_pb": float(loss_frame_pb.detach().cpu().item()),
        "weighted_L_avg_pb": float(weighted_avg.detach().cpu().item()),
        "weighted_L_frame_pb": float(weighted_frame.detach().cpu().item()),
        "L_total": float(loss_total.detach().cpu().item()),
    }
    return loss_total, diagnostics


def train_one_epoch(model, loader, optimizer, device, normalizer, lambda_avg: float, lambda_frame: float):
    model.train()
    totals = {key: 0.0 for key in ["L_exp", "L_avg_pb", "L_frame_pb", "weighted_L_avg_pb", "weighted_L_frame_pb", "L_total"]}
    total_graphs = 0

    for raw_batch in loader:
        batch = move_batch_to_device(raw_batch, device)
        optimizer.zero_grad()
        outputs = model(batch["crystal_batch"], batch["frame_batch"])
        loss, diagnostics = compute_losses(outputs, batch, normalizer, lambda_avg=lambda_avg, lambda_frame=lambda_frame)
        loss.backward()
        optimizer.step()

        num_graphs = int(batch["y_exp"].size(0))
        total_graphs += num_graphs
        for key in totals:
            totals[key] += diagnostics[key] * num_graphs

    return {key: value / total_graphs for key, value in totals.items()}


def evaluate(model, loader, device, normalizer, lambda_avg: float):
    model.eval()
    totals = {key: 0.0 for key in ["L_exp", "L_avg_pb", "weighted_L_avg_pb", "L_total"]}
    total_graphs = 0
    pred_exp: list[float] = []
    true_exp: list[float] = []
    pred_avg_pb: list[list[float]] = []
    true_avg_pb: list[list[float]] = []
    sample_ids: list[str] = []

    mse = nn.MSELoss()
    with torch.no_grad():
        for raw_batch in loader:
            batch = move_batch_to_device(raw_batch, device)
            outputs = model(batch["crystal_batch"], frame_batch=None)

            loss_exp = mse(normalizer.normalize_exp(outputs["pred_exp"]), normalizer.normalize_exp(batch["y_exp"]))
            loss_avg_pb = mse(
                normalizer.normalize_avg_pb(outputs["pred_avg_pb"]),
                normalizer.normalize_avg_pb(batch["y_avg_pb"]),
            )
            weighted_avg = lambda_avg * loss_avg_pb
            loss_total = loss_exp + weighted_avg

            num_graphs = int(batch["y_exp"].size(0))
            total_graphs += num_graphs
            totals["L_exp"] += float(loss_exp.detach().cpu().item()) * num_graphs
            totals["L_avg_pb"] += float(loss_avg_pb.detach().cpu().item()) * num_graphs
            totals["weighted_L_avg_pb"] += float(weighted_avg.detach().cpu().item()) * num_graphs
            totals["L_total"] += float(loss_total.detach().cpu().item()) * num_graphs

            pred_exp.extend(outputs["pred_exp"].detach().cpu().view(-1).tolist())
            true_exp.extend(batch["y_exp"].detach().cpu().view(-1).tolist())
            pred_avg_pb.extend(outputs["pred_avg_pb"].detach().cpu().tolist())
            true_avg_pb.extend(batch["y_avg_pb"].detach().cpu().tolist())
            sample_ids.extend(batch["sample_ids"])

    averaged = {key: value / total_graphs for key, value in totals.items()}
    averaged.update(
        {
            "pred_exp": pred_exp,
            "true_exp": true_exp,
            "pred_avg_pb": pred_avg_pb,
            "true_avg_pb": true_avg_pb,
            "sample_ids": sample_ids,
        }
    )
    return averaged


def save_train_log(rows: list[dict[str, object]], csv_path: str | Path) -> None:
    fieldnames = [
        "epoch",
        "train_L_exp",
        "train_L_avg_pb",
        "train_L_frame_pb",
        "train_weighted_L_avg_pb",
        "train_weighted_L_frame_pb",
        "train_L_total",
        "eval_L_exp",
        "eval_L_avg_pb",
        "eval_weighted_L_avg_pb",
        "eval_L_total",
    ]
    with Path(csv_path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def save_predictions(results: dict[str, object], csv_path: str | Path) -> None:
    fieldnames = [
        "sample_id",
        "pred_exp",
        "true_exp",
        "pred_vdw",
        "true_vdw",
        "pred_elec",
        "true_elec",
        "pred_polar_solv",
        "true_polar_solv",
        "pred_nonpolar_solv",
        "true_nonpolar_solv",
        "pred_dispersion",
        "true_dispersion",
        "pred_total",
        "true_total",
    ]
    with Path(csv_path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for index, sample_id in enumerate(results["sample_ids"]):
            pred_avg = results["pred_avg_pb"][index]
            true_avg = results["true_avg_pb"][index]
            writer.writerow(
                {
                    "sample_id": sample_id,
                    "pred_exp": results["pred_exp"][index],
                    "true_exp": results["true_exp"][index],
                    "pred_vdw": pred_avg[0],
                    "true_vdw": true_avg[0],
                    "pred_elec": pred_avg[1],
                    "true_elec": true_avg[1],
                    "pred_polar_solv": pred_avg[2],
                    "true_polar_solv": true_avg[2],
                    "pred_nonpolar_solv": pred_avg[3],
                    "true_nonpolar_solv": true_avg[3],
                    "pred_dispersion": pred_avg[4],
                    "true_dispersion": true_avg[4],
                    "pred_total": pred_avg[5],
                    "true_total": true_avg[5],
                }
            )


def save_debug_summary(rows: list[dict[str, object]], csv_path: str | Path) -> None:
    fieldnames = [
        "sample_id",
        "structure_name",
        "file_path",
        "ligand_resname",
        "num_ligand_atoms",
        "num_pocket_atoms",
        "num_retained_metals",
        "num_nodes",
        "num_edges",
        "node_dim",
        "edge_dim",
    ]
    with Path(csv_path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


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
    raw_root_dir = Path(args.raw_root_dir)
    frame_root_dir = Path(args.frame_root_dir)

    debug_dataset = Baseline3Dataset(
        graph_dir=graph_dir,
        raw_root_dir=raw_root_dir,
        frame_root_dir=frame_root_dir,
        sample_ids=list(dict.fromkeys(split_info["train_sample_ids"] + split_info["val_sample_ids"] + split_info["test_sample_ids"])),
        load_frames=True,
    )
    debug_rows = debug_dataset.debug_rows()
    print_debug_summary(debug_rows)

    train_dataset = Baseline3Dataset(
        graph_dir=graph_dir,
        raw_root_dir=raw_root_dir,
        frame_root_dir=frame_root_dir,
        sample_ids=split_info["train_sample_ids"],
        load_frames=True,
    )
    val_dataset = Baseline3Dataset(
        graph_dir=graph_dir,
        raw_root_dir=raw_root_dir,
        frame_root_dir=frame_root_dir,
        sample_ids=split_info["val_sample_ids"],
        load_frames=False,
    )
    test_dataset = Baseline3Dataset(
        graph_dir=graph_dir,
        raw_root_dir=raw_root_dir,
        frame_root_dir=frame_root_dir,
        sample_ids=split_info["test_sample_ids"],
        load_frames=False,
    )

    normalizer = LabelNormalizer.from_training_records(
        list(train_dataset.record_lookup().values()),
        enabled=normalize_labels,
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_baseline3_batch)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_baseline3_batch) if len(val_dataset) > 0 else None
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_baseline3_batch)

    in_dim = int(train_dataset[0]["crystal_graph"].x.size(-1))
    model = Baseline3PBModel(
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
            "raw_root_dir": str(raw_root_dir),
            "frame_root_dir": str(frame_root_dir),
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
            "lambda_frame": args.lambda_frame,
            "normalize_labels": normalize_labels,
            "seed": args.seed,
        },
        save_dir / "run_config.json",
    )

    train_log_rows: list[dict[str, object]] = []
    best_eval_loss = float("inf")
    best_epoch = -1
    best_val_results: dict[str, object] | None = None

    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device=device,
            normalizer=normalizer,
            lambda_avg=args.lambda_avg,
            lambda_frame=args.lambda_frame,
        )

        if val_loader is not None:
            eval_results = evaluate(
                model,
                val_loader,
                device=device,
                normalizer=normalizer,
                lambda_avg=args.lambda_avg,
            )
        else:
            eval_results = {key: train_metrics[key] for key in ["L_exp", "L_avg_pb", "weighted_L_avg_pb", "L_total"]}
            eval_results.update({"pred_exp": [], "true_exp": [], "pred_avg_pb": [], "true_avg_pb": [], "sample_ids": []})

        train_log_rows.append(
            {
                "epoch": epoch,
                "train_L_exp": train_metrics["L_exp"],
                "train_L_avg_pb": train_metrics["L_avg_pb"],
                "train_L_frame_pb": train_metrics["L_frame_pb"],
                "train_weighted_L_avg_pb": train_metrics["weighted_L_avg_pb"],
                "train_weighted_L_frame_pb": train_metrics["weighted_L_frame_pb"],
                "train_L_total": train_metrics["L_total"],
                "eval_L_exp": eval_results["L_exp"],
                "eval_L_avg_pb": eval_results["L_avg_pb"],
                "eval_weighted_L_avg_pb": eval_results["weighted_L_avg_pb"],
                "eval_L_total": eval_results["L_total"],
            }
        )

        if eval_results["L_total"] < best_eval_loss:
            best_eval_loss = eval_results["L_total"]
            best_epoch = epoch
            best_val_results = eval_results
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
                    "best_eval_loss": best_eval_loss,
                    "normalize_labels": normalize_labels,
                },
                save_dir / "best_model.pt",
            )

        if epoch == 1 or epoch % args.print_every == 0 or epoch == args.epochs:
            print(
                f"Epoch {epoch:4d} | "
                f"L_exp={train_metrics['L_exp']:.6f} "
                f"L_avg_pb={train_metrics['L_avg_pb']:.6f} "
                f"L_frame_pb={train_metrics['L_frame_pb']:.6f} "
                f"lambda_avg*L_avg_pb={train_metrics['weighted_L_avg_pb']:.6f} "
                f"lambda_frame*L_frame_pb={train_metrics['weighted_L_frame_pb']:.6f} "
                f"L_total={train_metrics['L_total']:.6f} | "
                f"eval_total={eval_results['L_total']:.6f}"
            )

    save_train_log(train_log_rows, save_dir / "train_log.csv")

    checkpoint = torch.load(save_dir / "best_model.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_results = evaluate(
        model,
        test_loader,
        device=device,
        normalizer=normalizer,
        lambda_avg=args.lambda_avg,
    )

    if best_val_results and best_val_results["sample_ids"]:
        save_predictions(best_val_results, save_dir / "best_validation_predictions.csv")
    save_predictions(test_results, save_dir / "best_predictions.csv")

    print()
    print("Baseline 3 training finished")
    print(f"  split_mode: {args.split_mode}")
    print(f"  train_ids: {split_info['train_sample_ids']}")
    print(f"  val_ids: {split_info['val_sample_ids']}")
    print(f"  test_ids: {split_info['test_sample_ids']}")
    print(f"  normalize_labels: {normalize_labels}")
    print(f"  best_epoch: {best_epoch}")
    print(f"  best_eval_loss: {best_eval_loss:.6f}")
    print(f"  save_dir: {save_dir}")


if __name__ == "__main__":
    main()
