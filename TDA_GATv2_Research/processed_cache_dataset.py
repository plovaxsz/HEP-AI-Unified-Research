"""Dataset loader for precomputed (cached) PyG graphs.

This bypasses:
- EnergyFlow real-data downloading
- ripser-based TDA recomputation

Expected cache format (created by CanonicalJetTDADataset):
  graph_{global_index:08d}.npz containing keys:
    - x: float array [num_nodes, node_dim]
    - edge_index: int64 array [2, num_edges]
    - edge_attr: float array [num_edges, edge_dim]
    - y: int label
    - u: float array [1, tda_dim] (graph-level topology summary)

The Data object returned includes:
  - x, edge_index, edge_attr, y, u
  - aliases: data.tda = data.u
"""

from __future__ import annotations

import glob
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch_geometric.data import Data
from torch.utils.data import Dataset


@dataclass(frozen=True)
class ProcessedCacheConfig:
    graph_pattern: str = "graph_*.npz"


DEFAULT_PROCESSED_CACHE_CONFIG = ProcessedCacheConfig()


class ProcessedGraphDataset(Dataset):
    """Loads precomputed graph NPZ files from a processed cache directory."""

    def __init__(
        self,
        *,
        cache_dir: str | Path,
        config: ProcessedCacheConfig = DEFAULT_PROCESSED_CACHE_CONFIG,
        max_items: Optional[int] = None,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.config = config

        if not self.cache_dir.exists():
            raise FileNotFoundError(f"Processed cache dir not found: {self.cache_dir}")

        paths = sorted(self.cache_dir.glob(self.config.graph_pattern))
        if not paths:
            raise FileNotFoundError(
                f"No cached graphs found in {self.cache_dir} with pattern {self.config.graph_pattern}"
            )

        if max_items is not None:
            paths = paths[: int(max_items)]

        self.paths = paths

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> Data:
        p = self.paths[idx]
        cached = np.load(p, allow_pickle=False)

        x = torch.from_numpy(cached["x"]).contiguous()
        edge_index = torch.from_numpy(cached["edge_index"]).contiguous()
        edge_attr = torch.from_numpy(cached["edge_attr"]).contiguous()

        y = int(cached["y"]) if "y" in cached else int(cached["label"])  # defensive

        u = torch.from_numpy(cached["u"]).contiguous()
        # stored as [1, tda_dim]
        if u.dim() == 1:
            u = u.unsqueeze(0)

        data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=torch.tensor([y], dtype=torch.long),
            u=u,
        )
        data.tda = data.u
        data.num_nodes = int(x.shape[0])
        return data


__all__ = ["ProcessedGraphDataset", "ProcessedCacheConfig"]

