"""Download and prepare the official CERN Pythia quark/gluon jet dataset.

This script uses EnergyFlow to fetch the public q/g benchmark and then
builds canonical NPZ shards for training. Run this after cloning the repo.

Usage:
    python TDA_GATv2_Research/download_dataset.py --num-data 50000
"""

from __future__ import annotations

import argparse
from pathlib import Path

from data_pipeline import build_canonical_shards


def main() -> None:
    parser = argparse.ArgumentParser(description="Download CERN Pythia q/g jet dataset.")
    parser.add_argument("--num-data", type=int, default=50_000, help="Total jets to download.")
    parser.add_argument("--shard-size", type=int, default=2_000, help="Samples per NPZ shard.")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "data" / "canonical_shards_real")
    parser.add_argument("--cache-dir", type=Path, default=Path(__file__).resolve().parent / "data")
    parser.add_argument("--generator", type=str, default="pythia")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.cache_dir.mkdir(parents=True, exist_ok=True)

    print(f"[DOWNLOAD] Fetching {args.num_data:,} {args.generator} q/g jets from EnergyFlow...")
    shard_paths = build_canonical_shards(
        output_dir=args.output_dir,
        num_data=args.num_data,
        shard_size=args.shard_size,
        generator=args.generator,
        allow_synthetic=False,
        cache_dir=args.cache_dir,
    )
    print(f"[DOWNLOAD] Created {len(shard_paths)} shards in {args.output_dir}")


if __name__ == "__main__":
    main()
