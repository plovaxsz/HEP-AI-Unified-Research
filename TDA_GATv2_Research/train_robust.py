"""Robust IPR training launcher with buffered I/O and auto-checkpoint every 5 cycles."""

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd
import torch

# --- CONFIG ---
TOTAL_CYCLES = 50
CYCLES_PER_WRITE = 5
EPOCHS_PER_CYCLE = 100
BATCH_SIZE = 128
NUM_WORKERS = 4
NUM_DATA = 50000
MODEL_SAVE_DIR = Path(r"D:\PISS\TDA_GATv2_Research\models")
LOG_PATH = MODEL_SAVE_DIR / "evolution.log"
CHECKPOINT_PATH = MODEL_SAVE_DIR / "gatv2_final.pth"
BEST_MODEL_EVER_PATH = MODEL_SAVE_DIR / "best_model_ever.pt"
SHARD_DIR = r"D:\PISS\TDA_GATv2_Research\data\canonical_shards_real"
CACHE_DIR = r"D:\PISS\TDA_GATv2_Research\data\processed_cache_real"
PYTHON = r"D:\PISS\.venv\Scripts\python.exe"
TRAIN_SCRIPT = r"D:\PISS\TDA_GATv2_Research\train.py"

data_buffer: list[dict] = []
global_best_val_auc = 0.0


def save_buffer_to_log(buffer: list[dict]) -> None:
    """Flush buffered metrics to CSV with safe header handling."""

    df = pd.DataFrame(buffer)
    header = not LOG_PATH.exists() or LOG_PATH.stat().st_size == 0
    df.to_csv(LOG_PATH, mode="a", index=False, header=header)
    print(f"\n[INFO] Flushed {len(buffer)} cycles to {LOG_PATH}\n")


def find_latest_checkpoint() -> Path | None:
    """Return the most recently modified checkpoint file in models/."""

    checkpoints = sorted(MODEL_SAVE_DIR.glob("checkpoint_cycle_*.pt"), key=os.path.getmtime, reverse=True)
    return checkpoints[0] if checkpoints else None


def graceful_exit(signum=None, frame=None) -> None:
    """Save current state on SIGINT/SIGTERM."""

    print("\n[GRACEFUL EXIT] Signal received. Saving current state...")
    if data_buffer:
        save_buffer_to_log(data_buffer)
    sys.exit(0)


def run_single_cycle(cycle: int) -> dict:
    """Execute one 100-epoch training subprocess and return last-epoch metrics."""

    global global_best_val_auc

    latest_ckpt = find_latest_checkpoint()
    resume_target = latest_ckpt if latest_ckpt and latest_ckpt.exists() else CHECKPOINT_PATH

    cmd = [
        PYTHON, TRAIN_SCRIPT,
        "--epochs", str(EPOCHS_PER_CYCLE),
        "--batch-size", str(BATCH_SIZE),
        "--num-workers", str(NUM_WORKERS),
        "--num-data", str(NUM_DATA),
        "--shard-dir", SHARD_DIR,
        "--cache-dir", CACHE_DIR,
        "--no-telemetry",
        "--resume", str(resume_target),
    ]

    print(f"[CYCLE {cycle}/{TOTAL_CYCLES}] Launching subprocess...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"[STDERR] {result.stderr[-2000:]}")
        raise RuntimeError(f"Training subprocess failed for cycle {cycle} (exit code {result.returncode})")
    if result.stdout:
        print(result.stdout.strip())

    # Read last line of evolution.log to get latest metrics for this cycle
    last_metrics = {}
    if LOG_PATH.exists():
        with open(LOG_PATH, "r") as f:
            lines = f.readlines()
        if len(lines) > 1:
            cols = lines[0].strip().split(",")
            vals = lines[-1].strip().split(",")
            last_metrics = dict(zip(cols, vals))
            try:
                cycle_auc = float(last_metrics.get("val_auc", 0))
                if cycle_auc > global_best_val_auc:
                    global_best_val_auc = cycle_auc
            except (ValueError, TypeError):
                pass

    return {
        "cycle": cycle,
        "val_auc": last_metrics.get("val_auc", ""),
        "train_loss": last_metrics.get("train_loss", ""),
        "train_accuracy": last_metrics.get("train_accuracy", ""),
        "val_accuracy": last_metrics.get("val_accuracy", ""),
        "learning_rate": last_metrics.get("learning_rate", ""),
        "epochs_completed": EPOCHS_PER_CYCLE,
        "status": "success",
    }


def main() -> None:
    signal.signal(signal.SIGINT, graceful_exit)
    signal.signal(signal.SIGTERM, graceful_exit)

    MODEL_SAVE_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[INIT] Robust training session: {TOTAL_CYCLES} cycles | {EPOCHS_PER_CYCLE} epochs/cycle")
    print(f"[INIT] Log: {LOG_PATH}")
    print(f"[INIT] Checkpoints: {MODEL_SAVE_DIR}")

    for cycle in range(1, TOTAL_CYCLES + 1):
        try:
            metrics = run_single_cycle(cycle)
            data_buffer.append(metrics)

            if cycle % CYCLES_PER_WRITE == 0:
                save_buffer_to_log(data_buffer)
                data_buffer.clear()

                checkpoint_name = MODEL_SAVE_DIR / f"checkpoint_cycle_{cycle}.pt"
                if CHECKPOINT_PATH.exists():
                    torch.save(torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False), checkpoint_name)
                    print(f"[CHECKPOINT] Saved {checkpoint_name}")

        except Exception as e:
            print(f"[ERROR] Cycle {cycle} failed: {e}")
            if data_buffer:
                save_buffer_to_log(data_buffer)
                data_buffer.clear()
            raise

    if data_buffer:
        save_buffer_to_log(data_buffer)

    print(f"\n[COMPLETE] All {TOTAL_CYCLES} cycles finished.")
    print(f"[COMPLETE] Best val_auc observed: {global_best_val_auc:.4f}")


if __name__ == "__main__":
    main()
