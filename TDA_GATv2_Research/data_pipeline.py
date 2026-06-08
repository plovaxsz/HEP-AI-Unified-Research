"""Data acquisition and CPU-bound topology preprocessing for q/g tagging.

This module keeps the expensive, combinatorial part of the pipeline on the CPU:

* load the public EnergyFlow quark/gluon benchmark when available;
* convert the raw constituents into a canonical per-particle representation;
* build sparse k-NN graphs from the relativistic Delta R metric;
* compute persistent homology with Ripser on the CPU;
* cache PyG ``Data`` objects so topology is never recomputed during training.

The design is intentionally asymmetric because the target machine is a laptop-class
system with large RAM but limited VRAM. That means topology is a preprocessing step,
not an in-epoch GPU kernel.
"""

from __future__ import annotations

import gc
import math
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from ripser import ripser
from torch_geometric.data import Data
from torch.utils.data import Dataset


DEFAULT_EPS_MAX = 1.2
DEFAULT_TOP_K = 64
DEFAULT_K_NEIGHBORS = 8
DEFAULT_TOPO_BINS = 8


def _import_energyflow():
    """Import EnergyFlow lazily with a clear error message on failure."""

    try:
        import energyflow as ef
    except Exception as exc:  # pragma: no cover - depends on local environment
        raise RuntimeError(
            "EnergyFlow could not be imported. The benchmark loader depends on the "
            "optional Wasserstein package, which is unavailable in this environment."
        ) from exc
    return ef


def wrap_delta_phi(phi_i: np.ndarray | float, phi_j: np.ndarray | float) -> np.ndarray:
    """Return periodic azimuthal differences in the interval (-pi, pi]."""

    return np.arctan2(np.sin(np.asarray(phi_i) - np.asarray(phi_j)), np.cos(np.asarray(phi_i) - np.asarray(phi_j)))


def build_delta_r_matrix(eta: np.ndarray, phi: np.ndarray) -> np.ndarray:
    """Build the pairwise Delta R matrix for a jet constituent cloud."""

    eta = np.asarray(eta, dtype=np.float32)
    phi = np.asarray(phi, dtype=np.float32)
    deta = eta[:, None] - eta[None, :]
    dphi = wrap_delta_phi(phi[:, None], phi[None, :])
    return np.sqrt(deta * deta + dphi * dphi).astype(np.float32)


def _finite_lifetimes(diagram: np.ndarray, eps_max: float) -> np.ndarray:
    """Convert a persistence diagram into finite lifetimes.

    Infinite deaths are clipped to ``eps_max`` so that the summary vector remains
    numerically stable and fixed-length.
    """

    if diagram.size == 0:
        return np.zeros(0, dtype=np.float32)
    birth = diagram[:, 0].astype(np.float32, copy=False)
    death = diagram[:, 1].astype(np.float32, copy=False)
    death = np.where(np.isfinite(death), death, np.float32(eps_max))
    life = np.clip(death - birth, 0.0, np.float32(eps_max))
    return life.astype(np.float32, copy=False)


def _betti_curve(diagram: np.ndarray, grid: np.ndarray, eps_max: float) -> np.ndarray:
    """Compute a discrete Betti curve from a persistence diagram."""

    if diagram.size == 0:
        return np.zeros(len(grid), dtype=np.float32)

    birth = diagram[:, 0].astype(np.float32, copy=False)
    death = diagram[:, 1].astype(np.float32, copy=False)
    death = np.where(np.isfinite(death), death, np.float32(eps_max))

    curve = np.zeros(len(grid), dtype=np.float32)
    for i, t in enumerate(grid):
        curve[i] = float(np.sum((birth <= t) & (t < death)))
    return curve


def topological_summary(
    diagrams: Sequence[np.ndarray],
    eps_max: float = DEFAULT_EPS_MAX,
    n_bins: int = DEFAULT_TOPO_BINS,
) -> np.ndarray:
    """Compress persistent homology diagrams into a fixed-size vector.

    The summary is ordered as:

    * Betti curve for H0 (``n_bins`` values)
    * Betti curve for H1 (``n_bins`` values)
    * total persistence, entropy, and point count for H0
    * total persistence, entropy, and point count for H1

    This yields a 22-dimensional vector when ``n_bins == 8``.
    """

    grid = np.linspace(0.0, eps_max, num=n_bins, endpoint=False, dtype=np.float32)
    features: list[np.ndarray] = []

    for dim in (0, 1):
        diagram = diagrams[dim] if dim < len(diagrams) else np.empty((0, 2), dtype=np.float32)
        features.append(_betti_curve(diagram, grid, eps_max))

    for dim in (0, 1):
        diagram = diagrams[dim] if dim < len(diagrams) else np.empty((0, 2), dtype=np.float32)
        lifetimes = _finite_lifetimes(diagram, eps_max)
        total_persistence = float(lifetimes.sum())
        n_points = float(lifetimes.size)
        if total_persistence > 0.0:
            probs = lifetimes / total_persistence
            entropy = float(-(probs * np.log(probs + 1e-12)).sum())
        else:
            entropy = 0.0
        features.append(np.asarray([total_persistence, entropy, n_points], dtype=np.float32))

    return np.concatenate(features, axis=0).astype(np.float32, copy=False)


def _synthetic_qg_like_benchmark(
    num_data: int,
    max_particles: int = 128,
    seed: int = 7,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a deterministic smoke-test dataset when EnergyFlow is unavailable."""

    rng = np.random.default_rng(seed)
    x = np.zeros((num_data, max_particles, 4), dtype=np.float32)
    y = rng.integers(0, 2, size=num_data, dtype=np.int64)

    for idx in range(num_data):
        n_particles = int(rng.integers(max_particles // 2, max_particles + 1))
        label_bias = 0.35 if y[idx] == 1 else -0.15
        pt = rng.gamma(shape=2.0 + label_bias, scale=1.0, size=n_particles).astype(np.float32)
        rapidity = rng.normal(loc=0.0, scale=1.0 + 0.15 * (1 - y[idx]), size=n_particles).astype(np.float32)
        phi = rng.uniform(-math.pi, math.pi, size=n_particles).astype(np.float32)
        pid = rng.integers(-5, 6, size=n_particles, dtype=np.int64).astype(np.float32)
        x[idx, :n_particles, 0] = pt
        x[idx, :n_particles, 1] = rapidity
        x[idx, :n_particles, 2] = phi
        x[idx, :n_particles, 3] = pid

    return x, y


def load_qg_jets(
    num_data: int = 100_000,
    generator: str = "pythia",
    cache_dir: str | Path | None = None,
    allow_synthetic: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Load the public quark/gluon benchmark or a synthetic smoke-test fallback."""

    if allow_synthetic:
        return _synthetic_qg_like_benchmark(num_data=num_data)

    if cache_dir is None:
        cache_dir = Path("./data")

    try:
        ef = _import_energyflow()
        x, y = ef.qg_jets.load(
            num_data=num_data,
            pad=True,
            ncol=4,
            generator=generator,
            with_bc=False,
            cache_dir=str(cache_dir) if cache_dir is not None else None,
        )
        return np.asarray(x, dtype=np.float32), np.asarray(y, dtype=np.int64)
    except Exception:
        if not allow_synthetic:
            raise
        return _synthetic_qg_like_benchmark(num_data=num_data)


def canonicalize_qg_features(raw_x: np.ndarray) -> np.ndarray:
    """Convert raw EnergyFlow features into canonical ``[pt, eta, phi, E, pid]`` arrays."""

    ef = _import_energyflow()
    raw_x = np.asarray(raw_x, dtype=np.float32)
    p4 = ef.p4s_from_ptyphipids(raw_x)
    pt = raw_x[..., 0].astype(np.float32, copy=False)
    eta = ef.etas_from_p4s(p4).astype(np.float32)
    phi = ef.phis_from_p4s(p4, phi_ref="hardest").astype(np.float32)
    energy = p4[..., 0].astype(np.float32)
    pid = raw_x[..., 3].astype(np.float32, copy=False)
    return np.stack([pt, eta, phi, energy, pid], axis=-1).astype(np.float32)


def build_canonical_shards(
    output_dir: str | Path,
    num_data: int = 100_000,
    shard_size: int = 10_000,
    generator: str = "pythia",
    allow_synthetic: bool = False,
    cache_dir: str | Path | None = None,
) -> list[Path]:
    """Download the benchmark and store canonical shards as compressed NPZ files."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_x, labels = load_qg_jets(
        num_data=num_data,
        generator=generator,
        cache_dir=cache_dir,
        allow_synthetic=allow_synthetic,
    )
    canonical_x = canonicalize_qg_features(raw_x)

    shard_paths: list[Path] = []
    for start in range(0, len(canonical_x), shard_size):
        stop = min(start + shard_size, len(canonical_x))
        shard_path = output_dir / f"qg_{start:07d}_{stop:07d}.npz"
        np.savez_compressed(shard_path, x=canonical_x[start:stop], y=labels[start:stop])
        shard_paths.append(shard_path)

    return shard_paths


def _process_one_jet(
    raw_jet: np.ndarray,
    label: int,
    top_k: int = DEFAULT_TOP_K,
    k_neighbors: int = DEFAULT_K_NEIGHBORS,
    maxdim: int = 1,
    eps_max: float = DEFAULT_EPS_MAX,
) -> Data:
    """Convert one jet into a PyG ``Data`` object with cached topology features."""

    jet = np.asarray(raw_jet, dtype=np.float32)
    valid = jet[:, 0] > 0.0
    jet = jet[valid]

    if jet.size == 0:
        jet = np.zeros((1, raw_jet.shape[-1]), dtype=np.float32)

    order = np.argsort(-jet[:, 0])
    jet = jet[order][:top_k]

    pt = jet[:, 0]
    eta = jet[:, 1]
    phi = jet[:, 2]
    energy = jet[:, 3]
    pid = jet[:, 4]

    x = np.stack([pt, eta, phi, energy, pid], axis=-1).astype(np.float32)

    delta_r = build_delta_r_matrix(eta, phi)
    diag = ripser(delta_r, distance_matrix=True, maxdim=maxdim, thresh=eps_max)["dgms"]
    topo = topological_summary(diag, eps_max=eps_max, n_bins=DEFAULT_TOPO_BINS)

    n_nodes = len(x)
    if n_nodes == 1:
        edge_index = np.asarray([[0], [0]], dtype=np.int64)
        edge_attr = np.asarray([[0.0, 0.0, 0.0]], dtype=np.float32)
    else:
        masked = delta_r.copy()
        np.fill_diagonal(masked, np.inf)
        nn_index = np.argsort(masked, axis=1)[:, : min(k_neighbors, n_nodes - 1)]

        sources: list[int] = []
        targets: list[int] = []
        edge_attr_rows: list[list[float]] = []
        for src in range(n_nodes):
            for dst in nn_index[src]:
                dphi = float(wrap_delta_phi(phi[src], phi[dst]))
                deta = float(eta[src] - eta[dst])
                sources.append(src)
                targets.append(int(dst))
                edge_attr_rows.append([float(math.sqrt(deta * deta + dphi * dphi)), deta, dphi])

        edge_index = np.asarray([sources, targets], dtype=np.int64)
        edge_attr = np.asarray(edge_attr_rows, dtype=np.float32)

    data = Data(
        x=torch.from_numpy(x),
        edge_index=torch.from_numpy(edge_index),
        edge_attr=torch.from_numpy(edge_attr),
        y=torch.tensor([int(label)], dtype=torch.long),
        u=torch.from_numpy(topo).unsqueeze(0),
        tda=torch.from_numpy(topo).unsqueeze(0),
    )
    data.num_nodes = int(n_nodes)
    return data


class CanonicalJetTDADataset(Dataset):
    """Dataset that loads canonical jet shards and caches processed PyG graphs."""

    def __init__(
        self,
        shard_dir: str | Path,
        cache_dir: str | Path | None = None,
        top_k: int = DEFAULT_TOP_K,
        k_neighbors: int = DEFAULT_K_NEIGHBORS,
        maxdim: int = 1,
        eps_max: float = DEFAULT_EPS_MAX,
        allow_synthetic: bool = False,
    ) -> None:
        self.shard_dir = Path(shard_dir)
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        self.top_k = top_k
        self.k_neighbors = k_neighbors
        self.maxdim = maxdim
        self.eps_max = eps_max
        self.allow_synthetic = allow_synthetic

        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.shards = sorted(self.shard_dir.glob("*.npz"))
        if not self.shards:
            if not self.allow_synthetic:
                raise FileNotFoundError(
                    f"No NPZ shards were found in {self.shard_dir}. Build them first with build_canonical_shards()."
                )
            self.shards = []
            self._synthetic_x, self._synthetic_y = _synthetic_qg_like_benchmark(512)

        self._index_map: list[tuple[int, int]] = []
        if self.shards:
            for shard_idx, shard_path in enumerate(self.shards):
                with np.load(shard_path, allow_pickle=False) as shard:
                    shard_len = int(shard["y"].shape[0])
                for local_idx in range(shard_len):
                    self._index_map.append((shard_idx, local_idx))
        else:
            self._index_map = [(0, idx) for idx in range(len(self._synthetic_y))]

    def __len__(self) -> int:
        return len(self._index_map)

    def _cache_path(self, global_index: int) -> Path | None:
        if self.cache_dir is None:
            return None
        return self.cache_dir / f"graph_{global_index:08d}.npz"

    def _load_raw_sample(self, global_index: int) -> tuple[np.ndarray, int]:
        shard_idx, local_idx = self._index_map[global_index]
        if self.shards:
            with np.load(self.shards[shard_idx], allow_pickle=False) as shard:
                return shard["x"][local_idx], int(shard["y"][local_idx])
        return self._synthetic_x[local_idx], int(self._synthetic_y[local_idx])

    def __getitem__(self, global_index: int) -> Data:
        cache_path = self._cache_path(global_index)
        if cache_path is not None and cache_path.exists():
            cached = np.load(cache_path, allow_pickle=False)
            data = Data(
                x=torch.from_numpy(cached["x"]),
                edge_index=torch.from_numpy(cached["edge_index"]),
                edge_attr=torch.from_numpy(cached["edge_attr"]),
                y=torch.tensor([int(cached["y"])], dtype=torch.long),
                u=torch.from_numpy(cached["u"]),
                tda=torch.from_numpy(cached["u"]),
            )
            data.num_nodes = int(cached["x"].shape[0])
            return data

        raw_x, label = self._load_raw_sample(global_index)
        data = _process_one_jet(
            raw_x,
            label,
            top_k=self.top_k,
            k_neighbors=self.k_neighbors,
            maxdim=self.maxdim,
            eps_max=self.eps_max,
        )

        if cache_path is not None:
            np.savez_compressed(
                cache_path,
                x=data.x.cpu().numpy(),
                edge_index=data.edge_index.cpu().numpy(),
                edge_attr=data.edge_attr.cpu().numpy(),
                y=int(data.y.item()),
                u=data.u.cpu().numpy(),
            )

        gc.collect()
        return data


def build_dataset_from_shards(
    shard_dir: str | Path,
    cache_dir: str | Path | None = None,
    **dataset_kwargs,
) -> CanonicalJetTDADataset:
    """Convenience helper for downstream training scripts."""

    return CanonicalJetTDADataset(shard_dir=shard_dir, cache_dir=cache_dir, **dataset_kwargs)


__all__ = [
    "CanonicalJetTDADataset",
    "DEFAULT_EPS_MAX",
    "DEFAULT_K_NEIGHBORS",
    "DEFAULT_TOP_K",
    "build_canonical_shards",
    "build_dataset_from_shards",
    "build_delta_r_matrix",
    "canonicalize_qg_features",
    "load_qg_jets",
    "topological_summary",
    "wrap_delta_phi",
]
