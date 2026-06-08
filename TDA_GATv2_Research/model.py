"""Graph attention model for topology-conditioned quark/gluon tagging.

The model keeps the message-passing stack intentionally small enough for a 4 GB VRAM
device while still exposing attention weights for telemetry and post-hoc analysis.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn import GATv2Conv, global_max_pool, global_mean_pool


def _attention_stats(attention_tuple: tuple[Tensor, Tensor]) -> dict[str, Tensor]:
    """Reduce raw attention weights into small scalar telemetry tensors."""

    _, alpha = attention_tuple
    alpha = alpha.detach()
    alpha_flat = alpha.reshape(-1)
    alpha_clamped = alpha_flat.clamp_min(1e-12)
    return {
        "attention_mean": alpha_flat.mean(),
        "attention_peak": alpha_flat.max(),
        "attention_entropy": -(alpha_clamped * alpha_clamped.log()).mean(),
    }


class TDAGATv2(nn.Module):
    """Topology-conditioned GATv2 for quark/gluon jet classification."""

    def __init__(
        self,
        node_dim: int = 5,
        tda_dim: int = 22,
        edge_dim: int = 3,
        num_classes: int = 2,
        hidden_dim: int = 64,
        gat_hidden: int = 32,
        heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.node_dim = node_dim
        self.tda_dim = tda_dim
        self.edge_dim = edge_dim
        self.dropout = dropout

        self.node_encoder = nn.Sequential(
            nn.LazyLinear(hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        self.gat1 = GATv2Conv(
            in_channels=hidden_dim,
            out_channels=gat_hidden,
            heads=heads,
            concat=True,
            edge_dim=edge_dim,
            dropout=dropout,
            residual=True,
        )
        self.gat2 = GATv2Conv(
            in_channels=gat_hidden * heads,
            out_channels=gat_hidden,
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

        fusion_dim = gat_hidden * heads * 2 + 64
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes),
        )

    def forward(
        self,
        data: Any,
        return_attention_weights: bool = True,
    ) -> tuple[Tensor, dict[str, Tensor], dict[str, dict[str, Tensor]] | None]:
        """Run a forward pass and return logits, telemetry, and raw attention weights."""

        x: Tensor = data.x.float()
        edge_index: Tensor = data.edge_index.long()
        edge_attr: Tensor = data.edge_attr.float()
        batch: Tensor = data.batch

        topo = getattr(data, "u", None)
        if topo is None:
            topo = getattr(data, "tda", None)
        if topo is None:
            topo = torch.zeros((int(batch.max().item()) + 1, self.tda_dim), device=x.device, dtype=x.dtype)
        topo = topo.float()
        if topo.dim() == 1:
            topo = topo.unsqueeze(0)

        node_topo = topo[batch]
        x = torch.cat([x, node_topo], dim=-1)
        x = self.node_encoder(x)

        x, att1 = self.gat1(x, edge_index, edge_attr=edge_attr, return_attention_weights=True)
        x = F.gelu(x)
        x, att2 = self.gat2(x, edge_index, edge_attr=edge_attr, return_attention_weights=True)
        x = F.gelu(x)

        pooled_mean = global_mean_pool(x, batch)
        pooled_max = global_max_pool(x, batch)
        topo_embed = self.topo_mlp(topo)
        logits = self.classifier(torch.cat([pooled_mean, pooled_max, topo_embed], dim=-1))

        layer1_stats = _attention_stats(att1)
        layer2_stats = _attention_stats(att2)
        telemetry = {
            "layer1": layer1_stats,
            "layer2": layer2_stats,
            "attention_mean": torch.stack([
                layer1_stats["attention_mean"],
                layer2_stats["attention_mean"],
            ]).mean(),
            "attention_peak": torch.stack([
                layer1_stats["attention_peak"],
                layer2_stats["attention_peak"],
            ]).max(),
        }

        if not return_attention_weights:
            return logits, telemetry, None

        attention_weights = {
            "layer1": {
                "edge_index": att1[0].detach().cpu(),
                "alpha": att1[1].detach().cpu(),
            },
            "layer2": {
                "edge_index": att2[0].detach().cpu(),
                "alpha": att2[1].detach().cpu(),
            },
        }
        return logits, telemetry, attention_weights


__all__ = ["TDAGATv2"]
