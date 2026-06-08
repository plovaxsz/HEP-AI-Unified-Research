"""Data acquisition & canonical conversion for Quark/Gluon jet tagging.

Implements the CPU-only preprocessing step for the asymmetric Edge-HPC design:

1) Download the public EnergyFlow quark/gluon benchmark.
2) Convert raw jet constituent features [pt, y, phi, pid] to canonical Cartesian
   4-vectors via ``ef.p4s_from_ptyphipids``.
3) Extract pseudorapidity ``eta`` and energy ``E``.
4) Stack per-particle canonical features as [pt, eta, phi, E, pid].
5) Save the resulting arrays into compressed NPZ shards.

This file is intended to run once (or when the shard cache is missing) and is
never used during training epochs.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class CanonicalDataConfig:
    """Configuration for preprocessing and sharding."""

    num_data: int = 100_000
    shard_size: int = 10_000
    generator: str = "pythia"
    ncol: int = 4
    with_bc: bool = False
    allow_synthetic: bool = False
    seed: int = 7


def _import_energyflow():
    """Import EnergyFlow lazily with a clearer error message."""

    try:
        import energyflow as ef
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "EnergyFlow import failed. Install energyflow to download the public q/g benchmark."
        ) from exc
    return ef


def _synthetic_fallback(num_data: int, max_particles: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic synthetic fallback for smoke tests.

    This returns *canonical* per-particle features expected by the training pipeline:
      [pt, eta, phi, E, pid]
    """
    import math

    rng = np.random.default_rng(seed)
    x = np.zeros((num_data, max_particles, 5), dtype=np.float32)  # [pt, eta, phi, E, pid]
    y = rng.integers(0, 2, size=num_data, dtype=np.int64)

    for idx in range(num_data):
        n_particles = int(rng.integers(max_particles // 2, max_particles + 1))
        label_bias = 0.35 if y[idx] == 1 else -0.15

        pt = rng.gamma(shape=2.0 + label_bias, scale=1.0, size=n_particles).astype(np.float32)
        eta = rng.normal(loc=0.0, scale=1.0 + 0.15 * (1 - y[idx]), size=n_particles).astype(np.float32)
        phi = rng.uniform(-math.pi, math.pi, size=n_particles).astype(np.float32)

        # Massless approximation: E ~ pt * cosh(eta)
        E = (pt * np.cosh(eta)).astype(np.float32)

        pid = rng.integers(-5, 6, size=n_particles).astype(np.float32)

        x[idx, :n_particles, 0] = pt
        x[idx, :n_particles, 1] = eta
        x[idx, :n_particles, 2] = phi
        x[idx, :n_particles, 3] = E
        x[idx, :n_particles, 4] = pid

    return x, y


def canonicalize_raw_features(raw_x: np.ndarray) -> np.ndarray:
    """Convert raw [pt, y, phi, pid] to canonical [pt, eta, phi, E, pid]."""

    ef = _import_energyflow()
    raw_x = np.asarray(raw_x, dtype=np.float32)

    p4 = ef.p4s_from_ptyphipids(raw_x)  # cartesian 4-vectors
    pt = raw_x[..., 0].astype(np.float32, copy=False)

    eta = ef.etas_from_p4s(p4).astype(np.float32)
    phi = ef.phis_from_p4s(p4, phi_ref="hardest").astype(np.float32)
    E = p4[..., 0].astype(np.float32)
    pid = raw_x[..., 3].astype(np.float32, copy=False)

    return np.stack([pt, eta, phi, E, pid], axis=-1).astype(np.float32)


def load_qg_jets(
    num_data: int,
    generator: str,
    ncol: int,
    with_bc: bool,
    cache_dir: Path | None,
    allow_synthetic: bool,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Load q/g benchmark arrays or return synthetic fallback."""

    if allow_synthetic:
        # EnergyFlow uses variable jet multiplicity with padding; for fallback we keep a fixed max.
        return _synthetic_fallback(num_data=num_data, max_particles=128, seed=seed)

    ef = _import_energyflow()
    x, y = ef.qg_jets.load(
        num_data=num_data,
        pad=True,
        ncol=ncol,
        generator=generator,
        with_bc=with_bc,
        cache_dir=str(cache_dir) if cache_dir is not None else None,
    )
    return np.asarray(x, dtype=np.float32), np.asarray(y, dtype=np.int64)


def shard_iter(total: int, shard_size: int) -> Iterable[tuple[int, int]]:
    """Yield inclusive-exclusive shard ranges."""

    for start in range(0, total, shard_size):
        stop = min(start + shard_size, total)
        yield start, stop


def build_canonical_shards(
    output_dir: str | Path,
    config: CanonicalDataConfig,
    cache_dir: str | Path | None = None,
) -> list[Path]:
    """Download/load q/g, canonicalize, and save NPZ shards.

    Output shard format:
      - ``x``: float32, shape (shard_len, max_particles, 5)
      - ``y``: int64, shape (shard_len,)
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir_path = Path(cache_dir) if cache_dir is not None else None

    raw_x, labels = load_qg_jets(
        num_data=config.num_data,
        generator=config.generator,
        ncol=config.ncol,
        with_bc=config.with_bc,
        cache_dir=cache_dir_path,
        allow_synthetic=config.allow_synthetic,
        seed=config.seed,
    )

    # Important: when allow_synthetic=True, load_qg_jets() returns already-canonical
    # features in the shape expected by the PyG preprocessing pipeline:
    #   [pt, eta, phi, E, pid] => last dim = 5
    if config.allow_synthetic:
        canonical_x = np.asarray(raw_x, dtype=np.float32)
    else:
        canonical_x = canonicalize_raw_features(raw_x)

    shard_paths: list[Path] = []
    for start, stop in shard_iter(total=len(canonical_x), shard_size=config.shard_size):
        shard_path = output_dir / f"qg_{start:07d}_{stop:07d}.npz"
        np.savez_compressed(
            shard_path,
            x=canonical_x[start:stop],
            y=labels[start:stop],
        )
        shard_paths.append(shard_path)

    return shard_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build canonical q/g jet shards for TDA-GATv2")
    parser.add_argument("--output-dir", type=Path, default=Path("./data/canonical_shards"))
    parser.add_argument("--cache-dir", type=Path, default=Path("./data"))
    parser.add_argument("--num-data", type=int, default=100_000)
    parser.add_argument("--shard-size", type=int, default=2_000)
    parser.add_argument("--generator", type=str, default="pythia")
    parser.add_argument("--ncol", type=int, default=4)
    parser.add_argument("--with-bc", action="store_true")
    parser.add_argument("--allow-synthetic", action="store_true")
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    cfg = CanonicalDataConfig(
        num_data=args.num_data,
        shard_size=args.shard_size,
        generator=args.generator,
        ncol=args.ncol,
        with_bc=args.with_bc,
        allow_synthetic=args.allow_synthetic,
        seed=args.seed,
    )

    build_canonical_shards(output_dir=args.output_dir, config=cfg, cache_dir=args.cache_dir)


if __name__ == "__main__":
    main()


__all__ = [
    "CanonicalDataConfig",
    "canonicalize_raw_features",
    "load_qg_jets",
    "build_canonical_shards",
]

