"""IPR Protocol launcher: 200 cycles x 100 epochs = 20,000 total epochs."""

import os
import subprocess
import time
from pathlib import Path


CHECKPOINT = r"D:\PISS\TDA_GATv2_Research\models\gatv2_final.pth"
EVOLUTION_LOG = r"D:\PISS\TDA_GATv2_Research\models\evolution.log"
PYTHON = r"D:\PISS\.venv\Scripts\python.exe"
TRAIN_SCRIPT = r"D:\PISS\TDA_GATv2_Research\train.py"
TRAIN_MODULE = "TDA_GATv2_Research.train"
SHARD_DIR = r"D:\PISS\TDA_GATv2_Research\data\canonical_shards_real"
CACHE_DIR = r"D:\PISS\TDA_GATv2_Research\data\processed_cache_real"
PROJECT_DIR = r"D:\PISS\TDA_GATv2_Research"


def run_ipr_protocol(runs=200, epochs=100, batch=128, num_workers=4):
    start_time = time.time()
    for i in range(runs):
        cycle_start = time.time()
        print(f"\n[IPR PROTOCOL] CYCLE {i+1}/{runs} | Epochs: {epochs} | Batch: {batch}")
        
        cmd = [
            PYTHON, TRAIN_SCRIPT,
            "--epochs", str(epochs),
            "--batch-size", str(batch),
            "--num-workers", str(num_workers),
            "--num-data", "50000",
            "--shard-dir", SHARD_DIR,
            "--cache-dir", CACHE_DIR,
            "--no-telemetry",
            "--lr", "0.0004",
            "--weight-decay", "0.0005",
        ]
        
        if os.path.exists(CHECKPOINT):
            cmd.extend(["--resume", CHECKPOINT])
        
        result = subprocess.run(cmd, check=True)
        
        cycle_time = time.time() - cycle_start
        print(f"[SUCCESS] Cycle {i+1} completed in {cycle_time:.1f}s | Checkpoint: {CHECKPOINT}")
        
        if os.path.exists(EVOLUTION_LOG):
            with open(EVOLUTION_LOG, "r") as f:
                lines = f.readlines()
            if len(lines) > 1:
                last_line = lines[-1].strip()
                print(f"[TELEMETRY] Last entry: {last_line}")
    
    total_time = time.time() - start_time
    print(f"\n[IPR COMPLETE] Total time: {total_time:.1f}s | Final checkpoint: {CHECKPOINT}")


if __name__ == "__main__":
    run_ipr_protocol(runs=49, epochs=100)
