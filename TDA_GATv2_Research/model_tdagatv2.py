"""Topology-conditioned GATv2 model with Physics XAI telemetry.

Implements ``TDAGATv2``:

- Node features: concatenate standard node features with graph-level topological
  summary ``u`` broadcasted to nodes.
- Two stacked ``torch_geometric.nn.GATv2Conv`` layers with configurable heads.
- XAI: uses ``return_attention_weights=True`` to extract raw attention weights.
  We compute:
    - ``attention_mean``: mean of alpha weights across edges and heads
    - ``attention_peak``: maximum alpha weight
- Fusion readout: concatenate
    - global mean pool
    - global max pool
    - topology embedding (MLP output from u)
  then run a final MLP classifier.

The module returns (logits, telemetry, attention_weights_or_none).
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn import GATv2Conv, global_max_pool, global_mean_pool


def _attention_stats(attention_tuple: tuple[Tensor, Tensor]) -> dict[str, Tensor]:
    """Reduce raw attention weights into scalar telemetry tensors."""

    _, alpha = attention_tuple  # alpha: [E, heads] or [E] depending on PyG version
    alpha = alpha.detach()
    alpha_flat = alpha.reshape(-1)
    alpha_clamped = alpha_flat.clamp_min(1e-12)

    return {
        "attention_mean": alpha_flat.mean(),
        "attention_peak": alpha_flat.max(),
        "attention_entropy": -(alpha_clamped * alpha_clamped.log()).mean(),
    }


class TDAGATv2(nn.Module):
    """Topology-conditioned GATv2 for quark/gluon classification."""

    def __init__(
        self,
        *,
        node_dim: int,
        tda_dim: int,
        edge_dim: int,
        num_classes: int = 2,
        hidden_channels: int = 32,
        heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.node_dim = node_dim
        self.tda_dim = tda_dim
        self.edge_dim = edge_dim
        self.heads = heads
        self.hidden_channels = hidden_channels

        self.node_encoder = nn.Sequential(
            nn.LayerNorm(node_dim + tda_dim),
            nn.Linear(node_dim + tda_dim, hidden_channels),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.gat1 = GATv2Conv(
            in_channels=hidden_channels,
            out_channels=hidden_channels,
            heads=heads,
            concat=True,
            edge_dim=edge_dim,
            dropout=dropout,
            residual=True,
        )
        self.gat2 = GATv2Conv(
            in_channels=hidden_channels * heads,
            out_channels=hidden_channels,
            heads=heads,
            concat=True,
            edge_dim=edge_dim,
            dropout=dropout,
            residual=True,
        )

        self.topo_mlp = nn.Sequential(
            nn.Linear(tda_dim, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 64),
            nn.GELU(),
        )

        fusion_dim = hidden_channels * heads * 2 + 64
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes),
        )

    def forward(
        self,
        data: Any,
        *,
        return_attention_weights: bool = True,
    ) -> tuple[Tensor, dict[str, Tensor], dict[str, dict[str, Tensor]] | None]:
        """Forward pass.

        Returns:
            logits: Tensor [batch_size, num_classes]
            telemetry: dict with at least attention_mean, attention_peak
            attention_weights: raw edge_index/alpha per layer if requested.
        """

        x: Tensor = data.x
        edge_index: Tensor = data.edge_index
        edge_attr: Tensor = data.edge_attr
        batch: Tensor = data.batch

        # Graph-level topology summary
        u: Tensor = getattr(data, "u", None)
        if u is None:
            raise AttributeError("Expected Data object to have attribute 'u' with shape [num_graphs, tda_dim].")

        if u.dim() == 1:
            u = u.unsqueeze(0)

        u = u.float()
        node_u = u[batch]  # broadcast to nodes: [num_nodes, tda_dim]

        x = torch.cat([x.float(), node_u], dim=-1)
        x = self.node_encoder(x)

        x, att1 = self.gat1(
            x,
            edge_index,
            edge_attr=edge_attr,
            return_attention_weights=True,
        )
        x = F.gelu(x)

        x, att2 = self.gat2(
            x,
            edge_index,
            edge_attr=edge_attr,
            return_attention_weights=True,
        )
        x = F.gelu(x)

        pooled_mean = global_mean_pool(x, batch)
        pooled_max = global_max_pool(x, batch)
        topo_embed = self.topo_mlp(u)

        logits = self.classifier(torch.cat([pooled_mean, pooled_max, topo_embed], dim=-1))

        layer1_stats = _attention_stats(att1)
        layer2_stats = _attention_stats(att2)

        attention_mean = torch.stack([layer1_stats["attention_mean"], layer2_stats["attention_mean"]]).mean()
        attention_peak = torch.stack([layer1_stats["attention_peak"], layer2_stats["attention_peak"]]).max()

        telemetry: dict[str, Tensor] = {
            "attention_mean": attention_mean,
            "attention_peak": attention_peak,
            "attention_entropy": (layer1_stats["attention_entropy"] + layer2_stats["attention_entropy"]) / 2,
        }

        if not return_attention_weights:
            return logits, telemetry, None

        attention_weights = {
            "layer1": {"edge_index": att1[0].detach().cpu(), "alpha": att1[1].detach().cpu()},
            "layer2": {"edge_index": att2[0].detach().cpu(), "alpha": att2[1].detach().cpu()},
        }
        return logits, telemetry, attention_weights


__all__ = ["TDAGATv2"]

