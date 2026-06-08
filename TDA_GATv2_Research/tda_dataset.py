"""CPU-bound TDA preprocessing and PyG dataset for topology-conditioned tagging.

This module implements the expensive topological data analysis (TDA) step on CPU:

For each jet:
  1) Remove zero padding and keep top-N particles by descending pT (N=64).
  2) Compute the relativistic distance matrix using
       d_ij = sqrt((eta_i - eta_j)^2 + (wrap(dphi_ij))^2)
  3) Compute persistent homology with ripser:
       ripser(distance_matrix, distance_matrix=True, maxdim=1)
  4) Convert diagrams into a fixed-size 22D summary vector ``t_J``:
       - Betti curve for H0: 8 bins
       - Betti curve for H1: 8 bins
       - Total persistence, persistence entropy, number of points for H0
       - Total persistence, persistence entropy, number of points for H1
  5) Build sparse k-NN graph (k=8) using the same distance matrix.
  6) Cache the processed ``torch_geometric.data.Data`` to disk.

Training MUST never recompute TDA; caching guarantees a one-time CPU cost.
"""

from __future__ import annotations

import gc
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch
from ripser import ripser
from torch_geometric.data import Data
from torch.utils.data import Dataset


@dataclass(frozen=True)
class TDAConfig:
    """TDA and graph construction hyperparameters."""

    top_n: int = 64
    k_neighbors: int = 8
    eps_max: float = 1.2
    maxdim: int = 1
    topo_bins: int = 8


DEFAULT_TDA_CONFIG = TDAConfig()


def wrap_delta_phi(phi_i: np.ndarray | float, phi_j: np.ndarray | np.ndarray) -> np.ndarray:
    """Return wrapped azimuthal differences in (-pi, pi]."""

    return np.arctan2(np.sin(np.asarray(phi_i) - np.asarray(phi_j)), np.cos(np.asarray(phi_i) - np.asarray(phi_j)))


def build_distance_matrix(eta: np.ndarray, phi: np.ndarray) -> np.ndarray:
    """Build the relativistic distance matrix used for TDA and k-NN edges."""

    eta = np.asarray(eta, dtype=np.float32)
    phi = np.asarray(phi, dtype=np.float32)
    deta = eta[:, None] - eta[None, :]
    dphi = wrap_delta_phi(phi[:, None], phi[None, :])
    return np.sqrt(deta * deta + dphi * dphi).astype(np.float32)


def _finite_lifetimes(diagram: np.ndarray, eps_max: float) -> np.ndarray:
    if diagram.size == 0:
        return np.zeros(0, dtype=np.float32)
    birth = diagram[:, 0].astype(np.float32, copy=False)
    death = diagram[:, 1].astype(np.float32, copy=False)
    death = np.where(np.isfinite(death), death, np.float32(eps_max))
    life = np.clip(death - birth, 0.0, np.float32(eps_max))
    return life


def _betti_curve(diagram: np.ndarray, grid: np.ndarray, eps_max: float) -> np.ndarray:
    if diagram.size == 0:
        return np.zeros(len(grid), dtype=np.float32)

    birth = diagram[:, 0].astype(np.float32, copy=False)
    death = diagram[:, 1].astype(np.float32, copy=False)
    death = np.where(np.isfinite(death), death, np.float32(eps_max))

    curve = np.zeros(len(grid), dtype=np.float32)
    for i, t in enumerate(grid):
        curve[i] = float(np.sum((birth <= t) & (t < death)))
    return curve


def tda_summary_from_diagrams(
    diagrams: Sequence[np.ndarray],
    *,
    eps_max: float,
    topo_bins: int,
) -> np.ndarray:
    """Return the 22D topological summary vector t_J."""

    grid = np.linspace(0.0, eps_max, num=topo_bins, endpoint=False, dtype=np.float32)
    features: list[np.ndarray] = []

    # Betti curves for H0 and H1
    for dim in (0, 1):
        diagram = diagrams[dim] if dim < len(diagrams) else np.empty((0, 2), dtype=np.float32)
        features.append(_betti_curve(diagram, grid, eps_max))

    # Total persistence, entropy, and number of points for H0 and H1
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

    out = np.concatenate(features, axis=0).astype(np.float32, copy=False)
    if out.shape[0] != 22:
        raise ValueError(f"Expected 22D summary vector, got shape {out.shape}")
    return out


def build_knn_edges(
    distance_matrix: np.ndarray,
    eta: np.ndarray,
    phi: np.ndarray,
    *,
    k_neighbors: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Build sparse k-NN graph edges and edge attributes.

    Edge attributes are [d_ij, deta, dphi].
    """

    n_nodes = distance_matrix.shape[0]
    if n_nodes == 1:
        edge_index = np.asarray([[0], [0]], dtype=np.int64)
        edge_attr = np.asarray([[0.0, 0.0, 0.0]], dtype=np.float32)
        return edge_index, edge_attr

    masked = distance_matrix.copy()
    np.fill_diagonal(masked, np.inf)
    k = min(k_neighbors, n_nodes - 1)
    nn_index = np.argsort(masked, axis=1)[:, :k]

    sources: list[int] = []
    targets: list[int] = []
    edge_attr_rows: list[list[float]] = []

    for src in range(n_nodes):
        for dst in nn_index[src]:
            dphi = float(wrap_delta_phi(phi[src], phi[dst]))
            deta = float(eta[src] - eta[dst])
            d = float(math.sqrt(deta * deta + dphi * dphi))
            sources.append(src)
            targets.append(int(dst))
            edge_attr_rows.append([d, deta, dphi])

    edge_index = np.asarray([sources, targets], dtype=np.int64)
    edge_attr = np.asarray(edge_attr_rows, dtype=np.float32)
    return edge_index, edge_attr


def _process_one_jet(
    canonical_jet: np.ndarray,
    label: int,
    *,
    tda_cfg: TDAConfig,
) -> Data:
    """CPU preprocessing for one jet."""

    jet = np.asarray(canonical_jet, dtype=np.float32)

    # canonical features: [pt, eta, phi, E, pid]
    pt = jet[..., 0]
    valid = pt > 0.0
    jet = jet[valid]

    if jet.shape[0] == 0:
        # Keep a single dummy node.
        jet = np.zeros((1, canonical_jet.shape[-1]), dtype=np.float32)

    # Top-N by descending pt
    order = np.argsort(-jet[:, 0])
    jet = jet[order][: tda_cfg.top_n]

    pt = jet[:, 0]
    eta = jet[:, 1]
    phi = jet[:, 2]
    energy = jet[:, 3]
    pid = jet[:, 4]

    node_x = np.stack([pt, eta, phi, energy, pid], axis=-1).astype(np.float32)

    distance_matrix = build_distance_matrix(eta, phi)
    ripser_out = ripser(distance_matrix, distance_matrix=True, maxdim=tda_cfg.maxdim, thresh=tda_cfg.eps_max)
    diagrams = ripser_out["dgms"]

    t_J = tda_summary_from_diagrams(diagrams, eps_max=tda_cfg.eps_max, topo_bins=tda_cfg.topo_bins)

    edge_index, edge_attr = build_knn_edges(distance_matrix, eta, phi, k_neighbors=tda_cfg.k_neighbors)

    data = Data(
        x=torch.from_numpy(node_x),
        edge_index=torch.from_numpy(edge_index),
        edge_attr=torch.from_numpy(edge_attr),
        y=torch.tensor([int(label)], dtype=torch.long),
        u=torch.from_numpy(t_J).unsqueeze(0),
    )
    # Compatibility aliases
    data.tda = data.u

    data.num_nodes = int(node_x.shape[0])
    return data


class CanonicalJetTDADataset(Dataset):
    """PyG Dataset reading canonical shards and caching TDA-processed graphs."""

    def __init__(
        self,
        shard_dir: str | Path,
        *,
        cache_dir: str | Path | None,
        tda_cfg: TDAConfig = DEFAULT_TDA_CONFIG,
        allow_synthetic: bool = False,
    ) -> None:
        self.shard_dir = Path(shard_dir)
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        self.tda_cfg = tda_cfg
        self.allow_synthetic = allow_synthetic

        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.shards = sorted(self.shard_dir.glob("*.npz"))
        if not self.shards and not self.allow_synthetic:
            raise FileNotFoundError(
                f"No canonical shards found in {self.shard_dir}. Create them with build_canonical_data.py first."
            )

        self._index_map: list[tuple[int, int]] = []
        self._synthetic_x: np.ndarray | None = None
        self._synthetic_y: np.ndarray | None = None

        if self.shards:
            for shard_idx, shard_path in enumerate(self.shards):
                with np.load(shard_path, allow_pickle=False) as shard:
                    shard_len = int(shard["y"].shape[0])
                for local_idx in range(shard_len):
                    self._index_map.append((shard_idx, local_idx))
        else:
            if not self.allow_synthetic:
                raise RuntimeError("Internal: allow_synthetic must be True when shards are empty")
            # Synthetic fallback canonical jets: [pt, y, phi, pid] would be needed; for smoke tests we store
            # already-canonical layout.
            self._synthetic_x, self._synthetic_y = self._build_synthetic_canonical(num_data=512, seed=7)
            self._index_map = [(0, i) for i in range(len(self._synthetic_y))]

    @staticmethod
    def _build_synthetic_canonical(num_data: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
        import math

        rng = np.random.default_rng(seed)
        max_particles = 128
        x = np.zeros((num_data, max_particles, 5), dtype=np.float32)  # [pt, eta, phi, E, pid]
        y = rng.integers(0, 2, size=num_data, dtype=np.int64)

        for idx in range(num_data):
            n_particles = int(rng.integers(max_particles // 2, max_particles + 1))
            label_bias = 0.35 if y[idx] == 1 else -0.15
            pt = rng.gamma(shape=2.0 + label_bias, scale=1.0, size=n_particles).astype(np.float32)
            eta = rng.normal(loc=0.0, scale=1.0 + 0.15 * (1 - y[idx]), size=n_particles).astype(np.float32)
            phi = rng.uniform(-math.pi, math.pi, size=n_particles).astype(np.float32)
            # Massless approx: E ~ pT * cosh(eta)
            E = (pt * np.cosh(eta)).astype(np.float32)
            pid = rng.integers(-5, 6, size=n_particles).astype(np.float32)

            x[idx, :n_particles, 0] = pt
            x[idx, :n_particles, 1] = eta
            x[idx, :n_particles, 2] = phi
            x[idx, :n_particles, 3] = E
            x[idx, :n_particles, 4] = pid

        return x, y

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
                x = shard["x"][local_idx]
                y = int(shard["y"][local_idx])
            return x, y

        assert self._synthetic_x is not None and self._synthetic_y is not None
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
            )
            data.tda = data.u
            data.num_nodes = int(data.x.shape[0])
            return data

        raw_x, label = self._load_raw_sample(global_index)
        data = _process_one_jet(raw_x, label, tda_cfg=self.tda_cfg)

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


__all__ = [
    "TDAConfig",
    "CanonicalJetTDADataset",
    "build_distance_matrix",
    "tda_summary_from_diagrams",
]

