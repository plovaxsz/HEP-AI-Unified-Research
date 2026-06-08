"""Training loop + Cyber HUD telemetry server for TDAGATv2.

Implements the GPU-bound portion of the asymmetric framework:

- Loads a cached TDA dataset (CPU preprocessing already performed).
- Uses PyG DataLoader with a VRAM-constrained batch size.
- Mixed precision training with GradScaler on CUDA.
- Cosine Annealing learning rate schedule.
- VRAM defense: aggressively frees tensors and clears CUDA cache.
- Publishes metrics via FastAPI + WebSocket telemetry (telemetry.py).
"""

from __future__ import annotations

import argparse
import gc
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, roc_auc_score
from torch.cuda.amp import GradScaler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch_geometric.loader import DataLoader

from TDA_GATv2_Research.model_tdagatv2 import TDAGATv2
from TDA_GATv2_Research.tda_dataset import CanonicalJetTDADataset, TDAConfig
from TDA_GATv2_Research.telemetry import TelemetryRecord, TelemetryServer, build_runtime_snapshot
from TDA_GATv2_Research.build_canonical_data import build_canonical_shards, CanonicalDataConfig
from TDA_GATv2_Research.processed_cache_dataset import ProcessedGraphDataset




def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train TDAGATv2")
    p.add_argument("--project-dir", type=Path, default=Path(__file__).resolve().parent)
    p.add_argument("--data-dir", type=Path, default=Path("./data"))
    p.add_argument("--canonical-shard-dir", type=Path, default=Path("./data/canonical_shards"))
    p.add_argument("--processed-cache-dir", type=Path, default=Path("./data/processed_cache"))
    p.add_argument("--telemetry-host", type=str, default="127.0.0.1")
    p.add_argument("--telemetry-port", type=int, default=8765)

    # dataset source
    p.add_argument(
        "--use-processed-graphs-dir",
        type=Path,
        default=None,
        help="If set, bypass EnergyFlow + ripser and load cached PyG graph NPZs from this directory.",
    )
    p.add_argument(
        "--max-items",
        type=int,
        default=None,
        help="Limit number of samples (useful for smoke tests). Only applies to processed-graphs-dir mode.",
    )


    # dataset
    p.add_argument("--num-data", type=int, default=100_000)
    p.add_argument("--shard-size", type=int, default=2_000)
    p.add_argument("--generator", type=str, default="pythia")
    p.add_argument("--allow-synthetic", action="store_true")
    p.add_argument("--val-fraction", type=float, default=0.2)

    # model
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--lr", type=float, default=8e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--batch-size", type=int, default=224)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--top-k", type=int, default=64, help="Top N particles by descending pt")
    p.add_argument("--k-neighbors", type=int, default=8)
    p.add_argument("--eps-max", type=float, default=1.2)
    p.add_argument("--maxdim", type=int, default=1)

    # VRAM defense
    p.add_argument("--clear-every", type=int, default=0, help="If >0, clear CUDA cache every N steps")

    return p.parse_args()


def ensure_canonical_shards(args: argparse.Namespace) -> None:
    if not args.canonical_shard_dir.exists() or not any(args.canonical_shard_dir.glob("*.npz")):
        args.canonical_shard_dir.mkdir(parents=True, exist_ok=True)
        cfg = CanonicalDataConfig(
            num_data=args.num_data,
            shard_size=args.shard_size,
            generator=args.generator,
            allow_synthetic=args.allow_synthetic,
        )
        build_canonical_shards(
            output_dir=args.canonical_shard_dir,
            config=cfg,
            cache_dir=args.data_dir,
        )


def split_dataset(dataset: Any, val_fraction: float) -> tuple[Any, Any]:
    n_total = len(dataset)
    n_val = max(1, int(n_total * val_fraction))
    n_train = max(1, n_total - n_val)
    g = torch.Generator().manual_seed(42)
    return torch.utils.data.random_split(dataset, [n_train, n_val], generator=g)


@torch.no_grad()
def evaluate(model: TDAGATv2, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    y_true: list[int] = []
    y_prob: list[float] = []

    for batch in loader:
        batch = batch.to(device, non_blocking=True)
        logits, _, _ = model(batch, return_attention_weights=False)
        probs = F.softmax(logits, dim=-1)[:, 1]

        y_true.extend(batch.y.view(-1).detach().cpu().tolist())
        y_prob.extend(probs.detach().cpu().tolist())

    acc = accuracy_score(y_true, [int(p > 0.5) for p in y_prob]) if y_true else 0.0
    try:
        auc = roc_auc_score(y_true, y_prob) if len(set(y_true)) > 1 else 0.5
    except ValueError:
        auc = 0.5

    return {"accuracy": float(acc), "val_auc": float(auc)}


def build_loader(dataset: Any, *, batch_size: int, num_workers: int, shuffle: bool) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        drop_last=False,
    )


def train_one_epoch(
    model: TDAGATv2,
    loader: DataLoader,
    optimizer: AdamW,
    scaler: GradScaler,
    device: torch.device,
    epoch: int,
    telemetry_server: TelemetryServer | None,
    *,
    clear_every: int,
) -> dict[str, float]:
    model.train()

    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    for step, batch in enumerate(loader):
        batch = batch.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        if device.type == "cuda":
            autocast_ctx = torch.cuda.amp.autocast(dtype=torch.float16)
        else:
            autocast_ctx = nullcontext()

        with autocast_ctx:
            logits, telemetry, _ = model(batch, return_attention_weights=True)
            loss = F.cross_entropy(logits, batch.y.view(-1))

        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        preds = logits.argmax(dim=-1)
        correct = int((preds == batch.y.view(-1)).sum().item())
        bs = int(batch.y.numel())

        total_loss += float(loss.detach().cpu()) * bs
        total_correct += correct
        total_examples += bs

        runtime = build_runtime_snapshot()
        if telemetry_server is not None:
            telemetry_server.publish(
                TelemetryRecord(
                    epoch=epoch,
                    step=step,
                    phase="train",
                    loss=float(loss.detach().cpu()),
                    accuracy=float(correct / max(1, bs)),
                    attention_mean=float(telemetry["attention_mean"].detach().cpu().item()),
                    attention_peak=float(telemetry["attention_peak"].detach().cpu().item()),
                    learning_rate=float(optimizer.param_groups[0]["lr"]),
                    cpu_percent=runtime["cpu_percent"],
                    ram_percent=runtime["ram_percent"],
                    vram_allocated_mb=(torch.cuda.memory_allocated() / (1024**2)) if device.type == "cuda" else None,
                    vram_reserved_mb=(torch.cuda.memory_reserved() / (1024**2)) if device.type == "cuda" else None,
                )
            )

        del batch, logits, loss, telemetry
        gc.collect()
        if device.type == "cuda":
            if clear_every and (step + 1) % clear_every == 0:
                torch.cuda.empty_cache()

    return {
        "loss": total_loss / max(1, total_examples),
        "accuracy": total_correct / max(1, total_examples),
    }


def main() -> None:
    args = parse_args()

    args.project_dir.mkdir(parents=True, exist_ok=True)
    args.data_dir.mkdir(parents=True, exist_ok=True)
    args.canonical_shard_dir.mkdir(parents=True, exist_ok=True)
    args.processed_cache_dir.mkdir(parents=True, exist_ok=True)

    telemetry_server = TelemetryServer(host=args.telemetry_host, port=args.telemetry_port)
    telemetry_server.start()

    # dataset selection
    if args.use_processed_graphs_dir is not None:
        dataset = ProcessedGraphDataset(
            cache_dir=args.use_processed_graphs_dir,
            max_items=args.max_items,
        )
    else:
        ensure_canonical_shards(args)

        tda_cfg = TDAConfig(
            top_n=args.top_k,
            k_neighbors=args.k_neighbors,
            eps_max=args.eps_max,
            maxdim=args.maxdim,
            topo_bins=8,
        )

        dataset = CanonicalJetTDADataset(
            shard_dir=args.canonical_shard_dir,
            cache_dir=args.processed_cache_dir,
            tda_cfg=tda_cfg,
            allow_synthetic=args.allow_synthetic,
        )


    train_set, val_set = split_dataset(dataset, val_fraction=args.val_fraction)
    train_loader = build_loader(train_set, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)
    val_loader = build_loader(val_set, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Node features = [pt, eta, phi, E, pid] => 5
    # Graph summary u => 22
    model = TDAGATv2(node_dim=5, tda_dim=22, edge_dim=3, hidden_channels=32, heads=4, dropout=0.1).to(device)

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))
    scaler = GradScaler(enabled=(device.type == "cuda"))

    best_auc = -float("inf")
    checkpoint_path = args.project_dir / "best_model.pt"

    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            epoch=epoch,
            telemetry_server=telemetry_server,
            clear_every=args.clear_every,
        )
        val_metrics = evaluate(model, val_loader, device)
        scheduler.step()

        epoch_snapshot = build_runtime_snapshot()
        telemetry_server.publish(
            TelemetryRecord(
                epoch=epoch,
                step=len(train_loader),
                phase="epoch_end",
                loss=float(train_metrics["loss"]),
                accuracy=float(train_metrics["accuracy"]),
                val_auc=float(val_metrics["val_auc"]),
                learning_rate=float(optimizer.param_groups[0]["lr"]),
                cpu_percent=epoch_snapshot["cpu_percent"],
                ram_percent=epoch_snapshot["ram_percent"],
                vram_allocated_mb=(torch.cuda.memory_allocated() / (1024**2)) if device.type == "cuda" else None,
                vram_reserved_mb=(torch.cuda.memory_reserved() / (1024**2)) if device.type == "cuda" else None,
            )
        )

        print(
            f"Epoch {epoch:03d} | loss={train_metrics['loss']:.4f} | acc={train_metrics['accuracy']:.4f} | "
            f"val_auc={val_metrics['val_auc']:.4f} | val_acc={val_metrics['accuracy']:.4f}"
        )

        if val_metrics["val_auc"] > best_auc:
            best_auc = val_metrics["val_auc"]
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_auc": best_auc,
                    "args": vars(args),
                },
                checkpoint_path,
            )

        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()


__all__ = ["main"]

