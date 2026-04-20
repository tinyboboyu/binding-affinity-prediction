"""First stable training baseline for processed protein-ligand complex graphs."""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

import torch
from torch import nn
from torch.optim import Adam
from torch_geometric.loader import DataLoader

from binding_graph_preprocessing.constants import DEFAULT_VALID_SAMPLE_IDS
from dataset import MMPBSAGraphDataset
from model import MultiTaskComplexGNN

VALID_SAMPLE_IDS = list(DEFAULT_VALID_SAMPLE_IDS)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a minimal multi-task GNN baseline.")
    parser.add_argument(
        "--graph_dir",
        default="../data/MMPBSA/processed/graphs",
        help="Directory containing .pt graph files.",
    )
    parser.add_argument("--save_dir", default=None, help="Directory to save outputs. Defaults to a run-specific path.")
    parser.add_argument("--split_mode", choices=["overfit_one", "overfit_all", "leave_one_out"], default="overfit_one")
    parser.add_argument(
        "--selection_mode",
        choices=["eval", "val"],
        default="eval",
        help="Model-selection set. Use 'val' to select best epoch on a validation sample and evaluate the test sample only once.",
    )
    parser.add_argument("--sample_id", default="6QLN", help="Sample ID for overfit_one mode.")
    parser.add_argument("--test_sample_id", default=None, help="Held-out sample ID for leave_one_out mode.")
    parser.add_argument(
        "--val_sample_id",
        default=None,
        help="Validation sample ID used only when split_mode='leave_one_out' and selection_mode='val'.",
    )
    parser.add_argument("--target_mode", choices=["exp", "aux_gb", "multi_gb"], default="multi_gb")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--lambda_aux", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--print_every", type=int, default=20)
    parser.add_argument("--device", default=None, help="Device to use. Defaults to cuda if available else cpu.")
    return parser


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_split(
    split_mode: str,
    selection_mode: str,
    sample_id: str | None = None,
    test_sample_id: str | None = None,
    val_sample_id: str | None = None,
) -> dict[str, object]:
    if split_mode == "overfit_one":
        train_id = sample_id or "6QLN"
        validate_sample_id(train_id, argument_name="sample_id")
        return {
            "split_mode": split_mode,
            "selection_mode": selection_mode,
            "train_sample_ids": [train_id],
            "val_sample_ids": [],
            "test_sample_ids": [],
        }

    if split_mode == "overfit_all":
        return {
            "split_mode": split_mode,
            "selection_mode": selection_mode,
            "train_sample_ids": list(VALID_SAMPLE_IDS),
            "val_sample_ids": [],
            "test_sample_ids": [],
        }

    if split_mode == "leave_one_out":
        if test_sample_id is None:
            raise ValueError("test_sample_id must be provided when split_mode='leave_one_out'")
        validate_sample_id(test_sample_id, argument_name="test_sample_id")
        candidate_sample_ids = [sample for sample in VALID_SAMPLE_IDS if sample != test_sample_id]
        if selection_mode == "val":
            resolved_val_sample_id = val_sample_id or candidate_sample_ids[0]
            validate_sample_id(resolved_val_sample_id, argument_name="val_sample_id")
            if resolved_val_sample_id == test_sample_id:
                raise ValueError("val_sample_id must be different from test_sample_id")
            if resolved_val_sample_id not in candidate_sample_ids:
                raise ValueError(
                    f"val_sample_id must come from the non-test samples: {candidate_sample_ids}"
                )
            train_sample_ids = [
                sample for sample in candidate_sample_ids if sample != resolved_val_sample_id
            ]
            val_sample_ids = [resolved_val_sample_id]
        else:
            train_sample_ids = candidate_sample_ids
            val_sample_ids = []
        return {
            "split_mode": split_mode,
            "selection_mode": selection_mode,
            "train_sample_ids": train_sample_ids,
            "val_sample_ids": val_sample_ids,
            "test_sample_ids": [test_sample_id],
        }

    raise ValueError(f"Unsupported split_mode: {split_mode}")


def validate_sample_id(sample_id: str, argument_name: str) -> None:
    if sample_id not in VALID_SAMPLE_IDS:
        raise ValueError(
            f"Invalid {argument_name}: {sample_id}. Valid sample IDs are {VALID_SAMPLE_IDS}"
        )


def build_loaders(
    graph_dir: str | Path,
    split_mode: str,
    selection_mode: str,
    sample_id: str | None = None,
    test_sample_id: str | None = None,
    val_sample_id: str | None = None,
    batch_size: int = 1,
):
    split_info = resolve_split(
        split_mode,
        selection_mode=selection_mode,
        sample_id=sample_id,
        test_sample_id=test_sample_id,
        val_sample_id=val_sample_id,
    )
    train_dataset = MMPBSAGraphDataset(graph_dir, sample_ids=split_info["train_sample_ids"])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    val_sample_ids = split_info["val_sample_ids"]
    if val_sample_ids:
        val_dataset = MMPBSAGraphDataset(graph_dir, sample_ids=val_sample_ids)
        val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    else:
        val_dataset = None
        val_loader = None

    test_sample_ids = split_info["test_sample_ids"]
    if test_sample_ids:
        test_dataset = MMPBSAGraphDataset(graph_dir, sample_ids=test_sample_ids)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    else:
        test_dataset = None
        test_loader = None

    return train_dataset, train_loader, val_dataset, val_loader, test_dataset, test_loader, split_info


def compute_loss(out, batch, target_mode: str, lambda_aux: float = 1.0):
    mse = nn.MSELoss()
    zero = torch.tensor(0.0, device=out["pred_exp"].device)
    target_aux = batch.y_aux.view(batch.num_graphs, -1).float()

    if target_mode == "exp":
        target_exp = batch.y_exp.view(-1, 1).float()
        loss_exp = mse(out["pred_exp"], target_exp)
        loss_aux = zero
        loss = loss_exp
    elif target_mode == "aux_gb":
        loss_aux = mse(out["pred_aux"], target_aux)
        loss_exp = zero
        loss = loss_aux
    elif target_mode == "multi_gb":
        target_exp = batch.y_exp.view(-1, 1).float()
        loss_exp = mse(out["pred_exp"], target_exp)
        loss_aux = mse(out["pred_aux"], target_aux)
        loss = loss_exp + lambda_aux * loss_aux
    else:
        raise ValueError(f"Unsupported target_mode: {target_mode}")

    loss_dict = {
        "loss_total": float(loss.detach().cpu().item()),
        "loss_exp": float(loss_exp.detach().cpu().item()),
        "loss_aux": float(loss_aux.detach().cpu().item()),
    }
    return loss, loss_dict


def train_one_epoch(model, loader, optimizer, device, target_mode: str, lambda_aux: float = 1.0):
    model.train()
    totals = {"loss_total": 0.0, "loss_exp": 0.0, "loss_aux": 0.0}
    total_graphs = 0

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch)
        loss, loss_dict = compute_loss(out, batch, target_mode=target_mode, lambda_aux=lambda_aux)
        loss.backward()
        optimizer.step()

        num_graphs = int(batch.num_graphs)
        total_graphs += num_graphs
        for key in totals:
            totals[key] += loss_dict[key] * num_graphs

    if total_graphs == 0:
        raise ValueError("Training loader produced zero graphs")

    return {key: value / total_graphs for key, value in totals.items()}


def evaluate(model, loader, device, target_mode: str, lambda_aux: float = 1.0):
    model.eval()
    totals = {"loss_total": 0.0, "loss_exp": 0.0, "loss_aux": 0.0}
    total_graphs = 0
    pred_exp: list[float] = []
    true_exp: list[float] = []
    pred_aux: list[list[float]] = []
    true_aux: list[list[float]] = []
    sample_ids: list[str] = []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch)
            _, loss_dict = compute_loss(out, batch, target_mode=target_mode, lambda_aux=lambda_aux)

            num_graphs = int(batch.num_graphs)
            total_graphs += num_graphs
            for key in totals:
                totals[key] += loss_dict[key] * num_graphs

            pred_exp.extend(out["pred_exp"].detach().cpu().view(-1).tolist())
            true_exp.extend(batch.y_exp.detach().cpu().view(-1).tolist())
            pred_aux.extend(out["pred_aux"].detach().cpu().tolist())
            true_aux.extend(batch.y_aux.detach().cpu().view(batch.num_graphs, -1).tolist())

            batch_sample_ids = batch.sample_id
            if isinstance(batch_sample_ids, str):
                sample_ids.append(batch_sample_ids)
            else:
                sample_ids.extend(list(batch_sample_ids))

    if total_graphs == 0:
        raise ValueError("Evaluation loader produced zero graphs")

    averaged = {key: value / total_graphs for key, value in totals.items()}
    averaged.update(
        {
            "pred_exp": pred_exp,
            "true_exp": true_exp,
            "pred_aux": pred_aux,
            "true_aux": true_aux,
            "sample_ids": sample_ids,
        }
    )
    return averaged


def save_train_log(rows: list[dict[str, object]], csv_path: str | Path) -> None:
    fieldnames = [
        "epoch",
        "train_loss_total",
        "train_loss_exp",
        "train_loss_aux",
        "eval_loss_total",
        "eval_loss_exp",
        "eval_loss_aux",
    ]
    with Path(csv_path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def save_predictions(results: dict[str, object], csv_path: str | Path, target_mode: str) -> None:
    include_aux = target_mode in {"aux_gb", "multi_gb"}
    fieldnames = ["sample_id", "pred_exp", "true_exp"]
    if include_aux:
        fieldnames.extend(
            [
                "pred_vdw",
                "true_vdw",
                "pred_elec",
                "true_elec",
                "pred_polar",
                "true_polar",
                "pred_nonpolar",
                "true_nonpolar",
            ]
        )

    with Path(csv_path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()

        for index, sample_id in enumerate(results["sample_ids"]):
            row = {
                "sample_id": sample_id,
                "pred_exp": results["pred_exp"][index] if target_mode != "aux_gb" else "",
                "true_exp": results["true_exp"][index] if target_mode != "aux_gb" else "",
            }
            if include_aux:
                pred_aux = results["pred_aux"][index]
                true_aux = results["true_aux"][index]
                row.update(
                    {
                        "pred_vdw": pred_aux[0],
                        "true_vdw": true_aux[0],
                        "pred_elec": pred_aux[1],
                        "true_elec": true_aux[1],
                        "pred_polar": pred_aux[2],
                        "true_polar": true_aux[2],
                        "pred_nonpolar": pred_aux[3],
                        "true_nonpolar": true_aux[3],
                    }
                )
            writer.writerow(row)


def resolve_default_save_dir(args: argparse.Namespace, split_info: dict[str, object]) -> Path:
    if args.save_dir:
        return Path(args.save_dir)

    if args.split_mode == "overfit_one":
        run_name = args.sample_id
    elif args.split_mode == "overfit_all":
        run_name = "all"
    else:
        run_name = f"loo_{args.test_sample_id}"
        if split_info["val_sample_ids"]:
            run_name = f"{run_name}_val_{split_info['val_sample_ids'][0]}"

    return Path("../results/training_runs") / f"{args.split_mode}_{args.selection_mode}_{args.target_mode}_{run_name}"


def select_device(device_arg: str | None) -> torch.device:
    if device_arg is not None:
        return torch.device(device_arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main() -> None:
    args = build_parser().parse_args()
    if args.selection_mode == "val" and args.split_mode != "leave_one_out":
        raise ValueError("selection_mode='val' is only supported with split_mode='leave_one_out'")

    set_seed(args.seed)
    device = select_device(args.device)

    train_dataset, train_loader, val_dataset, val_loader, test_dataset, test_loader, split_info = build_loaders(
        graph_dir=args.graph_dir,
        split_mode=args.split_mode,
        selection_mode=args.selection_mode,
        sample_id=args.sample_id,
        test_sample_id=args.test_sample_id,
        val_sample_id=args.val_sample_id,
        batch_size=args.batch_size,
    )
    if args.selection_mode == "val":
        selection_dataset = val_dataset
        selection_loader = val_loader
        final_test_loader = test_loader
    else:
        selection_dataset = test_dataset if args.split_mode == "leave_one_out" else train_dataset
        selection_loader = test_loader if args.split_mode == "leave_one_out" else DataLoader(
            selection_dataset,
            batch_size=1,
            shuffle=False,
        )
        final_test_loader = None

    first_sample = train_dataset[0]
    in_dim = int(first_sample.x.size(-1))
    model = MultiTaskComplexGNN(
        in_dim=in_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)
    optimizer = Adam(model.parameters(), lr=args.lr)

    save_dir = resolve_default_save_dir(args, split_info)
    save_dir.mkdir(parents=True, exist_ok=True)

    train_log_rows: list[dict[str, object]] = []
    best_eval_loss = float("inf")
    best_epoch = -1
    best_results: dict[str, object] | None = None
    final_test_results: dict[str, object] | None = None

    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device=device,
            target_mode=args.target_mode,
            lambda_aux=args.lambda_aux,
        )
        eval_results = evaluate(
            model,
            selection_loader,
            device=device,
            target_mode=args.target_mode,
            lambda_aux=args.lambda_aux,
        )

        train_log_rows.append(
            {
                "epoch": epoch,
                "train_loss_total": train_metrics["loss_total"],
                "train_loss_exp": train_metrics["loss_exp"],
                "train_loss_aux": train_metrics["loss_aux"],
                "eval_loss_total": eval_results["loss_total"],
                "eval_loss_exp": eval_results["loss_exp"],
                "eval_loss_aux": eval_results["loss_aux"],
            }
        )

        if eval_results["loss_total"] < best_eval_loss:
            best_eval_loss = eval_results["loss_total"]
            best_epoch = epoch
            best_results = eval_results
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
                    "target_mode": args.target_mode,
                    "best_epoch": best_epoch,
                    "best_eval_loss": best_eval_loss,
                },
                save_dir / "best_model.pt",
            )

        if epoch == 1 or epoch % args.print_every == 0 or epoch == args.epochs:
            print(
                f"Epoch {epoch:4d} | "
                f"train_total={train_metrics['loss_total']:.6f} "
                f"train_exp={train_metrics['loss_exp']:.6f} "
                f"train_aux={train_metrics['loss_aux']:.6f} | "
                f"eval_total={eval_results['loss_total']:.6f} "
                f"eval_exp={eval_results['loss_exp']:.6f} "
                f"eval_aux={eval_results['loss_aux']:.6f}"
            )

    save_train_log(train_log_rows, save_dir / "train_log.csv")
    if best_results is None:
        raise RuntimeError("No evaluation results were recorded during training")

    if final_test_loader is not None:
        checkpoint = torch.load(save_dir / "best_model.pt", map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        final_test_results = evaluate(
            model,
            final_test_loader,
            device=device,
            target_mode=args.target_mode,
            lambda_aux=args.lambda_aux,
        )
        save_predictions(
            best_results,
            save_dir / "best_validation_predictions.csv",
            target_mode=args.target_mode,
        )
        save_predictions(
            final_test_results,
            save_dir / "best_predictions.csv",
            target_mode=args.target_mode,
        )
    else:
        save_predictions(best_results, save_dir / "best_predictions.csv", target_mode=args.target_mode)

    print()
    print("Training finished")
    print(f"  split_mode: {args.split_mode}")
    print(f"  selection_mode: {args.selection_mode}")
    print(f"  target_mode: {args.target_mode}")
    print(f"  best_epoch: {best_epoch}")
    print(f"  best_eval_loss: {best_eval_loss:.6f}")
    if final_test_results is not None:
        print(f"  final_test_loss: {final_test_results['loss_total']:.6f}")
    print(f"  save_dir: {save_dir}")


if __name__ == "__main__":
    main()
