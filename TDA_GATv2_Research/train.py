"""Training entrypoint for the asymmetric CPU/GPU TDA-GATv2 pipeline."""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import sys
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, roc_auc_score
from torch.amp.grad_scaler import GradScaler  # type: ignore
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch_geometric.loader import DataLoader

from data_pipeline import CanonicalJetTDADataset, build_canonical_shards
from model import TDAGATv2
from telemetry import TelemetryRecord, TelemetryServer, build_runtime_snapshot


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for training and preprocessing."""

    parser = argparse.ArgumentParser(description="Train the TDA-GATv2 quark/gluon classifier.")
    parser.add_argument("--project-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--data-dir", type=Path, default=Path("./data"))
    parser.add_argument("--shard-dir", type=Path, default=Path("./data/canonical_shards"))
    parser.add_argument("--cache-dir", type=Path, default=Path("./data/processed_cache"))
    parser.add_argument("--num-data", type=int, default=10_000)
    parser.add_argument("--shard-size", type=int, default=2_000)
    parser.add_argument("--generator", type=str, default="pythia")
    parser.add_argument("--batch-size", type=int, default=224)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=0.0004)
    parser.add_argument("--weight-decay", type=float, default=0.0005)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--top-k", type=int, default=64)
    parser.add_argument("--k-neighbors", type=int, default=8)
    parser.add_argument("--eps-max", type=float, default=1.2)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--maxdim", type=int, default=1)
    parser.add_argument("--telemetry-host", type=str, default="127.0.0.1")
    parser.add_argument("--telemetry-port", type=int, default=8765)
    parser.add_argument("--resume", type=Path, default=None, help="Path to a checkpoint to resume training from.")
    parser.add_argument("--no-telemetry", action="store_true", help="Disable the telemetry WebSocket server for maximum performance.")
    return parser.parse_args()


def _compute_sha256(file_path: Path) -> str:
    """Compute SHA256 hash of a file for data integrity verification."""

    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure_dataset(args: argparse.Namespace) -> CanonicalJetTDADataset:
    """Create the processed graph dataset, building shards if needed."""

    if not args.shard_dir.exists() or not any(args.shard_dir.glob("*.npz")):
        raise FileNotFoundError(
            f"REAL DATA REQUIRED: No NPZ shards found in {args.shard_dir}. "
            f"Populate this directory with official CERN Pythia canonical shards before training."
        )

    shard_paths = sorted(args.shard_dir.glob("*.npz"))
    print(f"[DATA INTEGRITY] Loading {len(shard_paths)} real CERN shards from: {args.shard_dir}")
    for shard_path in shard_paths[:3]:
        sha = _compute_sha256(shard_path)
        print(f"[DATA INTEGRITY]   {shard_path.name} | SHA256: {sha[:16]}...")

    return CanonicalJetTDADataset(
        shard_dir=args.shard_dir,
        cache_dir=args.cache_dir,
        top_k=args.top_k,
        k_neighbors=args.k_neighbors,
        maxdim=args.maxdim,
        eps_max=args.eps_max,
        allow_synthetic=False,
    )


def split_dataset(dataset: CanonicalJetTDADataset, val_fraction: float = 0.2):
    """Split the dataset into train and validation subsets."""

    n_total = len(dataset)
    n_val = max(1, int(n_total * val_fraction))
    n_train = max(1, n_total - n_val)
    generator = torch.Generator().manual_seed(42)
    return torch.utils.data.random_split(dataset, [n_train, n_val], generator=generator)


def build_loader(dataset, batch_size: int, num_workers: int, shuffle: bool) -> DataLoader:
    """Build a PyG DataLoader tuned for CPU preprocessing and GPU transfer."""

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        prefetch_factor=4,
        drop_last=False,
    )


@torch.no_grad()
def evaluate(model: TDAGATv2, loader: DataLoader, device: torch.device) -> dict[str, float]:
    """Evaluate accuracy and ROC-AUC on a validation loader."""

    model.eval()
    y_true: list[int] = []
    y_prob: list[float] = []
    y_pred: list[int] = []

    for batch in loader:
        batch = batch.to(device, non_blocking=True)
        logits, telemetry, _ = model(batch, return_attention_weights=True)
        probs = F.softmax(logits, dim=-1)[:, 1]
        preds = logits.argmax(dim=-1)
        y_true.extend(batch.y.view(-1).detach().cpu().tolist())
        y_prob.extend(probs.detach().cpu().tolist())
        y_pred.extend(preds.detach().cpu().tolist())

    acc = accuracy_score(y_true, y_pred) if y_true else 0.0
    try:
        auc = roc_auc_score(y_true, y_prob) if len(set(y_true)) > 1 else 0.5
    except ValueError:
        auc = 0.5

    return {"accuracy": float(acc), "val_auc": float(auc)}


def train_one_epoch(
    model: TDAGATv2,
    loader: DataLoader,
    optimizer: AdamW,
    scaler: GradScaler,
    device: torch.device,
    epoch: int,
    telemetry_server: TelemetryServer | None,
) -> dict[str, float]:
    """Run one training epoch with mixed precision and telemetry broadcasting."""

    model.train()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    for step, batch in enumerate(loader):
        batch = batch.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        if device.type == "cuda":
            autocast_ctx = torch.amp.autocast("cuda", dtype=torch.float16)  # type: ignore
        else:
            autocast_ctx = nullcontext()

        with autocast_ctx:
            logits, telemetry, attn = model(batch, return_attention_weights=True)
            loss = F.cross_entropy(logits, batch.y.view(-1))

        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        preds = logits.argmax(dim=-1)
        correct = int((preds == batch.y.view(-1)).sum().item())
        batch_size = int(batch.y.numel())

        total_loss += float(loss.detach().cpu()) * batch_size
        total_correct += correct
        total_examples += batch_size

        runtime = build_runtime_snapshot()
        payload = TelemetryRecord(
            epoch=epoch,
            step=step,
            phase="train",
            loss=float(loss.detach().cpu()),
            accuracy=float(correct / max(1, batch_size)),
            learning_rate=float(optimizer.param_groups[0]["lr"]),
            attention_mean=float(telemetry["attention_mean"].detach().cpu().item()),
            attention_peak=float(telemetry["attention_peak"].detach().cpu().item()),
            vram_allocated_mb=(torch.cuda.memory_allocated() / (1024**2)) if device.type == "cuda" else None,
            vram_reserved_mb=(torch.cuda.memory_reserved() / (1024**2)) if device.type == "cuda" else None,
            cpu_percent=runtime["cpu_percent"],
            ram_percent=runtime["ram_percent"],
        )
        if telemetry_server is not None:
            telemetry_server.publish(payload)

        del batch, logits, loss, telemetry, attn

    return {
        "loss": total_loss / max(1, total_examples),
        "accuracy": total_correct / max(1, total_examples),
    }


def main() -> None:
    """Entry point for the training job."""

    args = parse_args()
    args.project_dir.mkdir(parents=True, exist_ok=True)
    args.data_dir.mkdir(parents=True, exist_ok=True)
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    args.shard_dir.mkdir(parents=True, exist_ok=True)

    if not args.no_telemetry:
        telemetry_server = TelemetryServer(host=args.telemetry_host, port=args.telemetry_port)
        telemetry_server.start()
    else:
        telemetry_server = None

    dataset = ensure_dataset(args)
    train_set, val_set = split_dataset(dataset, val_fraction=args.val_fraction)
    train_loader = build_loader(train_set, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)
    val_loader = build_loader(val_set, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TDAGATv2(node_dim=5, tda_dim=22, edge_dim=3).to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))
    scaler = GradScaler("cuda", enabled=device.type == "cuda")

    models_dir = args.project_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    best_val_auc = 0.0
    checkpoint_path = models_dir / "gatv2_final.pth"
    evolution_log_path = models_dir / "evolution.log"
    start_epoch = 1
    recent_aucs: list[float] = []
    lr_reduced = False
    run_id = f"RUN_{int(__import__('time').time())}"

    if args.resume is not None:
        resume_path = Path(args.resume)
        if not resume_path.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")
        checkpoint = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        best_val_auc = float(checkpoint.get("val_auc", 0.0))
        start_epoch = 1
        print(f"Resumed from {resume_path} with best_val_auc={best_val_auc:.4f}. Starting fresh epoch count from 1.")

    log_exists = evolution_log_path.exists() and evolution_log_path.stat().st_size > 0
    with open(evolution_log_path, "a", newline="") as csv_file:
        csv_writer = csv.writer(csv_file)
        if not log_exists:
            csv_writer.writerow([
                "run_id",
                "shard_dir",
                "epoch",
                "train_loss",
                "train_accuracy",
                "val_accuracy",
                "val_auc",
                "learning_rate",
                "cpu_percent",
                "ram_percent",
            ])

    try:
        for epoch in range(start_epoch, start_epoch + args.epochs):
            train_metrics = train_one_epoch(
                model=model,
                loader=train_loader,
                optimizer=optimizer,
                scaler=scaler,
                device=device,
                epoch=epoch,
                telemetry_server=telemetry_server,
            )
            val_metrics = evaluate(model, val_loader, device)
            scheduler.step()

        epoch_snapshot = build_runtime_snapshot()
        if telemetry_server is not None:
            telemetry_server.publish(
                TelemetryRecord(
                    epoch=epoch,
                    step=len(train_loader),
                    phase="epoch_end",
                    loss=float(train_metrics["loss"]),
                    accuracy=float(train_metrics["accuracy"]),
                    val_auc=float(val_metrics["val_auc"]),
                    learning_rate=float(optimizer.param_groups[0]["lr"]),
                    vram_allocated_mb=(torch.cuda.memory_allocated() / (1024**2)) if device.type == "cuda" else None,
                    vram_reserved_mb=(torch.cuda.memory_reserved() / (1024**2)) if device.type == "cuda" else None,
                    cpu_percent=epoch_snapshot["cpu_percent"],
                    ram_percent=epoch_snapshot["ram_percent"],
                )
            )

        print(
            f"Epoch {epoch:03d} | loss={train_metrics['loss']:.4f} | acc={train_metrics['accuracy']:.4f} | "
            f"val_acc={val_metrics['accuracy']:.4f} | val_auc={val_metrics['val_auc']:.4f}"
        )

        with open(evolution_log_path, "a", newline="") as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow([
                run_id,
                str(args.shard_dir),
                epoch,
                f"{train_metrics['loss']:.6f}",
                f"{train_metrics['accuracy']:.6f}",
                f"{val_metrics['accuracy']:.6f}",
                f"{val_metrics['val_auc']:.6f}",
                f"{optimizer.param_groups[0]['lr']:.8f}",
                f"{epoch_snapshot['cpu_percent']:.2f}",
                f"{epoch_snapshot['ram_percent']:.2f}",
            ])

        if val_metrics["val_auc"] > best_val_auc:
            best_val_auc = val_metrics["val_auc"]
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_auc": best_val_auc,
                "args": vars(args),
            }, checkpoint_path)
            print(f" >>> New best model saved with AUC: {best_val_auc:.4f}")
            
            best_model_ever_path = models_dir / "best_model_ever.pt"
            torch.save(model.state_dict(), best_model_ever_path)
            print(f" >>> BEST MODEL EVER saved to {best_model_ever_path} | AUC: {best_val_auc:.4f}")

        recent_aucs.append(val_metrics["val_auc"])
        if len(recent_aucs) > 20:
            recent_aucs.pop(0)
        
        if len(recent_aucs) >= 20 and not lr_reduced:
            auc_std = float(__import__('statistics').stdev(recent_aucs))
            auc_range = max(recent_aucs) - min(recent_aucs)
            if auc_std < 0.001 and auc_range < 0.005:
                for param_group in optimizer.param_groups:
                    param_group["lr"] *= 0.5
                lr_reduced = True
                print(f" >>> LR REDUCED to {optimizer.param_groups[0]['lr']:.8f} (stagnation detected)")

        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    except KeyboardInterrupt:
        print("\n[GRACEFUL EXIT] KeyboardInterrupt received. Saving final checkpoint...")
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_auc": best_val_auc,
            "args": vars(args),
        }, checkpoint_path)
        print(f"[GRACEFUL EXIT] Final checkpoint saved to {checkpoint_path}")
        sys.exit(0)


if __name__ == "__main__":
    main()
