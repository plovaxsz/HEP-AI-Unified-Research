"""Phase 2: Graph Generative Adversarial Network (GraphGAN) for Jet Synthesis.

Architecture:
    - Generator: 128-dim latent vector -> variable-size jet graph (N=15-50 particles)
    - Discriminator: Reuses TDAGATv2 (Phase 1 validated architecture)
    - Physics-Informed Loss: Penalizes jets with unphysical invariant mass / total pT

Training: BCE-GAN (stable baseline for smoke test)

Outputs:
    - gen_graph_gan_training.log       : Adversarial training metrics
    - figure_gan_convergence.png/pdf   : Generator/Discriminator loss curves
    - figure_jet_comparison.png/pdf    : Real vs Synthetic jet profiles
    - figure_particle_multiplicity.png/pdf : Multiplicity distribution comparison
    - graph_gan_final.pt               : Final generator + discriminator weights
"""

from __future__ import annotations

import argparse
import logging
import math
import random
import sys
import time
import warnings
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import gaussian_kde, wasserstein_distance
from torch import Tensor
from torch_geometric.nn import GATv2Conv, global_max_pool, global_mean_pool

warnings.filterwarnings(
    "ignore",
    message="The usage of `scatter\\(reduce='max'\\)` can be accelerated via the 'torch-scatter' package",
    category=UserWarning,
)

# --- CONFIG ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
REAL_DATA_DIR = Path(r"D:\PISS\TDA_GATv2_Research\data\canonical_shards_real")
CACHE_DIR = Path(r"D:\PISS\TDA_GATv2_Research\data\processed_cache_real")
OUTPUT_DIR = Path(r"D:\PISS\TDA_GATv2_Research\models")
LOG_PATH = OUTPUT_DIR / "gen_graph_gan_training.log"
EVOLUTION_LOG_PATH = OUTPUT_DIR / "phase2_gan_evolution.log"

LATENT_DIM = 128
HIDDEN_DIM = 128
GAT_HIDDEN = 64
HEADS = 4
DROPOUT = 0.1
LEARNING_RATE_G = 2e-4
LEARNING_RATE_D = 2e-4
BATCH_SIZE = 16
NUM_EPOCHS = 500
NODE_MIN = 15
NODE_MAX = 50
PHYSICS_LOSS_WEIGHT = 0.3
MOMENTUM_PENALTY_WEIGHT = 2.0
CHECKPOINT_INTERVAL = 100

plt.style.use("seaborn-v0_8-darkgrid")
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
})

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_PATH),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


# ============================================================================
# PHYSICS-INFORMED LOSS
# ============================================================================

class PhysicsInformedLoss(nn.Module):
    """Enforce physical constraints on generated jet graphs in log-space."""

    def __init__(
        self,
        real_mean_mass: float = 45.0,
        real_std_mass: float = 15.0,
        real_mean_pt: float = 520.0,
        real_std_pt: float = 50.0,
        weight: float = 0.3,
        momentum_weight: float = 2.0,
    ) -> None:
        super().__init__()
        self.real_mean_mass = real_mean_mass
        self.real_std_mass = real_std_mass
        self.real_mean_pt = real_mean_pt
        self.real_std_pt = real_std_pt
        self.weight = weight
        self.momentum_weight = momentum_weight

    @classmethod
    def from_dataset(cls, real_jets: list, weight: float = 0.3, momentum_weight: float = 2.0) -> "PhysicsInformedLoss":
        masses = []
        total_pts = []
        for data in real_jets:
            x = data.x
            pt = x[:, 0].clamp(min=0.0)
            eta = x[:, 1]
            phi = x[:, 2]
            energy = x[:, 3].clamp(min=0.0)
            px = pt * torch.cos(phi)
            py = pt * torch.sin(phi)
            pz = pt * torch.sinh(eta)
            mass_sq = energy.sum() ** 2 - (px.sum() ** 2 + py.sum() ** 2 + pz.sum() ** 2)
            masses.append(torch.sqrt(torch.clamp(mass_sq, min=0.0)).item())
            total_pts.append(pt.sum().item())
        masses = torch.tensor(masses)
        total_pts = torch.tensor(total_pts)
        return cls(
            real_mean_mass=float(masses.mean()),
            real_std_mass=float(masses.std().clamp(min=1.0)),
            real_mean_pt=float(total_pts.mean()),
            real_std_pt=float(total_pts.std().clamp(min=1.0)),
            weight=weight,
            momentum_weight=momentum_weight,
        )

    def inverse_log_transform(self, x: Tensor) -> Tensor:
        x_phys = x.clone()
        x_phys[:, 0] = torch.exp(x_phys[:, 0]) - 1.0
        x_phys[:, 3] = torch.exp(x_phys[:, 3]) - 1.0
        return x_phys

    def compute_jet_mass(self, x: Tensor) -> Tensor:
        x_phys = self.inverse_log_transform(x)
        pt = x_phys[:, 0].clamp(min=0.0)
        eta = x_phys[:, 1]
        phi = x_phys[:, 2]
        energy = x_phys[:, 3].clamp(min=0.0)
        px = pt * torch.cos(phi)
        py = pt * torch.sin(phi)
        pz = pt * torch.sinh(eta)
        total_px = px.sum()
        total_py = py.sum()
        total_pz = pz.sum()
        total_e = energy.sum()
        mass_sq = total_e ** 2 - (total_px ** 2 + total_py ** 2 + total_pz ** 2)
        mass = torch.sqrt(torch.clamp(mass_sq, min=0.0))
        return mass

    def compute_total_pt(self, x: Tensor) -> Tensor:
        x_phys = self.inverse_log_transform(x)
        return x_phys[:, 0].sum()

    def compute_leading_pt(self, x: Tensor) -> Tensor:
        x_phys = self.inverse_log_transform(x)
        return x_phys[:, 0].max()

    def forward(self, x: Tensor, n_nodes: int) -> Tensor:
        if x.shape[0] < 2:
            return torch.tensor(0.0, device=x.device)
        mass = self.compute_jet_mass(x)
        total_pt = self.compute_total_pt(x)
        leading_pt = self.compute_leading_pt(x)
        mass_deviation = (mass - self.real_mean_mass) / self.real_std_mass
        pt_deviation = (total_pt - self.real_mean_pt) / self.real_std_pt
        leading_pt_target = self.real_mean_pt * 0.35
        leading_pt_deviation = (leading_pt - leading_pt_target) / (self.real_std_pt * 0.5)
        n_penalty = 0.0
        if n_nodes < NODE_MIN or n_nodes > NODE_MAX:
            n_penalty = 1.0
        penalty = (
            torch.abs(mass_deviation) +
            torch.abs(pt_deviation) * 0.1 +
            torch.abs(leading_pt_deviation) * self.momentum_weight +
            n_penalty * 0.5
        )
        return self.weight * penalty


# ============================================================================
# GENERATOR
# ============================================================================

class GraphGenerator(nn.Module):
    """Generate variable-size jet graphs from latent vectors."""

    def __init__(
        self,
        latent_dim: int = 128,
        hidden_dim: int = 128,
        node_min: int = 15,
        node_max: int = 50,
        node_features: int = 5,
        edge_features: int = 3,
        k_neighbors: int = 8,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.node_min = node_min
        self.node_max = node_max
        self.node_features = node_features
        self.edge_features = edge_features
        self.k_neighbors = k_neighbors
        self.node_range = node_max - node_min + 1

        self.mlp = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        self.count_head = nn.Linear(hidden_dim, self.node_range)
        self.node_decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, node_features * self.node_range),
        )
        self.edge_mlp = nn.Sequential(
            nn.Linear(node_features * 2, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, edge_features),
        )

    def forward(self, z: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        batch_size = z.shape[0]
        features = self.mlp(z)
        count_logits = self.count_head(features)
        count_probs = F.softmax(count_logits, dim=-1)
        node_counts = torch.argmax(count_probs, dim=-1) + self.node_min

        node_feature_logits = self.node_decoder(features)
        node_feature_logits = node_feature_logits.view(batch_size, self.node_range, self.node_features)

        x_list = []
        edge_index_list = []
        edge_attr_list = []
        batch_list = []

        node_offset = 0

        for i in range(batch_size):
            n = int(node_counts[i].item())
            idx = n - self.node_min
            base_features = node_feature_logits[i, idx, :]
            noise = torch.randn(n, self.node_features, device=z.device) * 0.1
            node_features_i = base_features.unsqueeze(0) + noise

            pt = F.relu(node_features_i[:, 0])
            eta = torch.tanh(node_features_i[:, 1]) * 3.0
            phi = torch.tanh(node_features_i[:, 2]) * math.pi
            energy = F.relu(node_features_i[:, 3])
            pid = torch.tanh(node_features_i[:, 4]) * 5.0
            node_features_i = torch.stack([pt, eta, phi, energy, pid], dim=-1)

            eta = node_features_i[:, 1]
            phi = node_features_i[:, 2]
            deta = eta.unsqueeze(0) - eta.unsqueeze(1)
            dphi = torch.atan2(
                torch.sin(phi.unsqueeze(0) - phi.unsqueeze(1)),
                torch.cos(phi.unsqueeze(0) - phi.unsqueeze(1)),
            )
            dr = torch.sqrt(deta ** 2 + dphi ** 2)
            inf_diag = torch.full_like(dr, float("inf"))
            inf_diag = torch.triu(inf_diag, diagonal=1) + torch.tril(inf_diag, diagonal=-1)
            dr = dr + inf_diag
            nn_idx = torch.argsort(dr, dim=-1)[:, :self.k_neighbors]

            sources = []
            targets = []
            edge_attrs = []
            for src in range(n):
                for dst in nn_idx[src]:
                    dphi_val = float(torch.atan2(
                        torch.sin(phi[src] - phi[dst]),
                        torch.cos(phi[src] - phi[dst]),
                    ).item())
                    deta_val = float((eta[src] - eta[dst]).item())
                    sources.append(src)
                    targets.append(int(dst.item()))
                    edge_attrs.append([
                        float(torch.sqrt(torch.tensor(deta_val**2 + dphi_val**2)).item()),
                        deta_val,
                        dphi_val,
                    ])

            edge_index = torch.tensor([sources, targets], dtype=torch.long, device=z.device)
            edge_index = edge_index + node_offset
            edge_attr = torch.tensor(edge_attrs, dtype=torch.float, device=z.device)

            x_list.append(node_features_i)
            edge_index_list.append(edge_index)
            edge_attr_list.append(edge_attr)
            batch_list.append(torch.full((n,), i, dtype=torch.long, device=z.device))
            node_offset += n

        x = torch.cat(x_list, dim=0)
        edge_index = torch.cat(edge_index_list, dim=-1)
        edge_attr = torch.cat(edge_attr_list, dim=0)
        batch = torch.cat(batch_list, dim=0)

        return x, edge_index, edge_attr, batch, node_counts


# ============================================================================
# DISCRIMINATOR
# ============================================================================

class GraphDiscriminator(nn.Module):
    """Discriminator based on TDAGATv2 (Phase 1 validated architecture)."""

    def __init__(
        self,
        node_dim: int = 5,
        tda_dim: int = 22,
        edge_dim: int = 3,
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
            nn.Linear(128, 1),
        )

    def forward(self, data: Any) -> Tensor:
        x = data.x.float()
        edge_index = data.edge_index.long()
        edge_attr = data.edge_attr.float()
        batch = data.batch

        topo = getattr(data, "u", None)
        if topo is None:
            topo = getattr(data, "tda", None)
        if topo is None:
            topo = torch.zeros(
                (int(batch.max().item()) + 1, self.tda_dim),
                device=x.device,
                dtype=x.dtype,
            )
        topo = topo.float()
        if topo.dim() == 1:
            topo = topo.unsqueeze(0)

        node_topo = topo[batch]
        x = torch.cat([x, node_topo], dim=-1)
        x = self.node_encoder(x)

        x, _ = self.gat1(x, edge_index, edge_attr=edge_attr, return_attention_weights=True)
        x = F.gelu(x)
        x, _ = self.gat2(x, edge_index, edge_attr=edge_attr, return_attention_weights=True)
        x = F.gelu(x)

        pooled_mean = global_mean_pool(x, batch)
        pooled_max = global_max_pool(x, batch)
        topo_embed = self.topo_mlp(topo)
        logits = self.classifier(torch.cat([pooled_mean, pooled_max, topo_embed], dim=-1))
        return logits


# ============================================================================
# DATA LOADING
# ============================================================================

def load_real_jets(shard_dir: Path, cache_dir: Path, max_samples: int = 5000) -> list:
    from data_pipeline import CanonicalJetTDADataset
    dataset = CanonicalJetTDADataset(
        shard_dir=shard_dir,
        cache_dir=cache_dir,
        allow_synthetic=False,
    )
    samples = []
    for i in range(min(max_samples, len(dataset))):
        data = dataset[i]
        data.batch = torch.zeros(data.x.shape[0], dtype=torch.long)
        samples.append(data)
    return samples


def inverse_transform_generator_output(x: Tensor) -> Tensor:
    x_phys = x.clone()
    x_phys[:, 0] = torch.exp(x_phys[:, 0].clamp(min=0.0, max=7.0)) - 1.0
    x_phys[:, 3] = torch.exp(x_phys[:, 3].clamp(min=0.0, max=7.0)) - 1.0
    return x_phys


def transform_real_jets_to_log_space(jets: list) -> None:
    for data in jets:
        data.x = data.x.clone()
        data.x[:, 0] = torch.log(data.x[:, 0].clamp(min=0.0) + 1.0)
        data.x[:, 3] = torch.log(data.x[:, 3].clamp(min=0.0) + 1.0)


# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_gan_convergence(g_losses: list, d_losses: list, physics_losses: list) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.suptitle(
        "Figure 6: GraphGAN Training Convergence",
        fontsize=13,
        fontweight="bold",
        y=1.02,
    )
    epochs = range(1, len(g_losses) + 1)
    ax.plot(epochs, g_losses, color="crimson", linewidth=2, label="Generator Loss")
    ax.plot(epochs, d_losses, color="navy", linewidth=2, label="Discriminator Loss")
    ax.plot(epochs, physics_losses, color="orange", linewidth=2, linestyle="--", label="Physics Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Adversarial Training Dynamics")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    save_figure(fig, "figure_gan_convergence")
    plt.close(fig)


def plot_jet_comparison(real_jets: list, fake_jets: list, n_samples: int = 5) -> None:
    fig, axes = plt.subplots(1, n_samples, figsize=(n_samples * 4, 4))
    fig.suptitle(
        "Figure 7: Real vs Synthetic Jet Energy Profiles",
        fontsize=13,
        fontweight="bold",
        y=1.02,
    )
    for i in range(n_samples):
        ax = axes[i] if n_samples > 1 else axes
        real_pt = real_jets[i].x[:, 0].cpu().numpy()
        fake_pt = fake_jets[i].x[:, 0].detach().cpu().numpy()
        real_sorted = np.sort(real_pt)[::-1]
        fake_sorted = np.sort(fake_pt)[::-1]
        ax.bar(range(len(real_sorted)), real_sorted, alpha=0.7, color="steelblue", label="Real")
        ax.bar(range(len(fake_sorted)), fake_sorted, alpha=0.7, color="crimson", label="Synthetic")
        ax.set_xlabel("Particle Rank (by pT)")
        ax.set_ylabel("pT (GeV)")
        ax.set_title(f"Jet {i+1}")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    save_figure(fig, "figure_jet_comparison")
    plt.close(fig)


def plot_particle_multiplicity(real_jets: list, fake_jets: list) -> None:
    real_counts = [data.x.shape[0] for data in real_jets]
    fake_counts = [data.x.shape[0] for data in fake_jets]
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.suptitle(
        "Figure 8: Particle Multiplicity Distribution",
        fontsize=13,
        fontweight="bold",
        y=1.02,
    )
    bins = np.arange(NODE_MIN - 1, NODE_MAX + 2)
    ax.hist(real_counts, bins=bins, alpha=0.7, color="steelblue", label="Real Jets", edgecolor="black")
    ax.hist(fake_counts, bins=bins, alpha=0.7, color="crimson", label="Synthetic Jets", edgecolor="black")
    ax.axvline(x=np.mean(real_counts), color="blue", linestyle="--", linewidth=2, label=f"Real Mean = {np.mean(real_counts):.1f}")
    ax.axvline(x=np.mean(fake_counts), color="red", linestyle="--", linewidth=2, label=f"Synthetic Mean = {np.mean(fake_counts):.1f}")
    ax.set_xlabel("Particle Count (N)")
    ax.set_ylabel("Frequency")
    ax.set_title("Multiplicity Distribution Comparison")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    save_figure(fig, "figure_particle_multiplicity")
    plt.close(fig)


def save_figure(fig: plt.Figure, name: str) -> None:
    png_path = OUTPUT_DIR / f"{name}.png"
    pdf_path = OUTPUT_DIR / f"{name}.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    print(f"  Saved: {png_path}")
    print(f"  Saved: {pdf_path}")


# ============================================================================
# TRAINING LOOP
# ============================================================================

def train_gan(
    generator: GraphGenerator,
    discriminator: GraphDiscriminator,
    real_jets: list,
    num_epochs: int = 50,
    batch_size: int = 16,
    lr_g: float = 2e-4,
    lr_d: float = 2e-4,
    physics_weight: float = 0.3,
) -> None:
    optimizer_g = torch.optim.Adam(generator.parameters(), lr=lr_g, betas=(0.5, 0.999))
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=lr_d, betas=(0.5, 0.999))
    physics_loss_fn = PhysicsInformedLoss.from_dataset(
        real_jets, weight=physics_weight, momentum_weight=MOMENTUM_PENALTY_WEIGHT
    )

    g_losses = []
    d_losses = []
    physics_losses = []

    evolution_file = open(EVOLUTION_LOG_PATH, "a", newline="", encoding="utf-8")
    if evolution_file.tell() == 0:
        evolution_file.write("epoch,g_loss,d_loss,phys_penalty,avg_node_var\n")
        evolution_file.flush()

    logger.info(f"Starting GraphGAN training: {num_epochs} epochs, {len(real_jets)} real jets")

    for epoch in range(1, num_epochs + 1):
        generator.train()
        discriminator.train()

        batch_real = random.sample(real_jets, min(batch_size, len(real_jets)))
        batch_real = [data.to(DEVICE) for data in batch_real]

        z = torch.randn(len(batch_real), LATENT_DIM, device=DEVICE)
        with torch.no_grad():
            x_fake, edge_index_fake, edge_attr_fake, batch_fake, _ = generator(z)

        fake_data_list = []
        for i in range(len(batch_real)):
            mask = batch_fake == i
            n = int(mask.sum().item())
            if n == 0:
                continue
            x_i = x_fake[mask]
            edge_mask = (batch_fake[edge_index_fake[0]] == i) & (batch_fake[edge_index_fake[1]] == i)
            ei_i = edge_index_fake[:, edge_mask]
            local_idx = torch.zeros(batch_fake.shape[0], dtype=torch.long, device=DEVICE)
            local_idx[mask] = torch.arange(n, device=DEVICE)
            ei_i = local_idx[ei_i]
            ea_i = edge_attr_fake[edge_mask]
            from torch_geometric.data import Data
            fake_data = Data(
                x=x_i,
                edge_index=ei_i,
                edge_attr=ea_i,
                y=torch.tensor([0], device=DEVICE),
            )
            fake_data.batch = torch.zeros(n, dtype=torch.long, device=DEVICE)
            fake_data_list.append(fake_data)

        if len(fake_data_list) == 0:
            continue

        optimizer_d.zero_grad()
        real_logits = torch.cat([discriminator(data) for data in batch_real])
        fake_logits = torch.cat([discriminator(data) for data in fake_data_list])
        d_loss = F.binary_cross_entropy_with_logits(fake_logits, torch.zeros_like(fake_logits)) + \
                 F.binary_cross_entropy_with_logits(real_logits, torch.ones_like(real_logits))
        d_loss.backward()
        optimizer_d.step()

        optimizer_g.zero_grad()
        fake_logits = torch.cat([discriminator(data) for data in fake_data_list])
        g_loss_adv = F.binary_cross_entropy_with_logits(fake_logits, torch.ones_like(fake_logits))

        physics_penalty = torch.tensor(0.0, device=DEVICE)
        for fake_data in fake_data_list:
            fake_data.x = inverse_transform_generator_output(fake_data.x)
            physics_penalty += physics_loss_fn(fake_data.x, fake_data.x.shape[0])
        physics_penalty = physics_penalty / len(fake_data_list)

        g_loss = g_loss_adv + physics_penalty
        g_loss.backward()
        optimizer_g.step()

        g_losses.append(float(g_loss.item()))
        d_losses.append(float(d_loss.item()))
        physics_losses.append(float(physics_penalty.item()))
        avg_node_var = float(x_fake.var(dim=0).mean().item())

        evolution_file.write(
            f"{epoch},{g_losses[-1]:.6f},{d_losses[-1]:.6f},{physics_losses[-1]:.6f},{avg_node_var:.6f}\n"
        )
        evolution_file.flush()

        logger.info(
            f"Epoch {epoch:04d} | G_loss: {g_losses[-1]:.4f} | "
            f"D_loss: {d_losses[-1]:.4f} | Phys_Penalty: {physics_losses[-1]:.4f} | "
            f"Avg_Node_Var: {avg_node_var:.4f}"
        )

    logger.info("GraphGAN training completed.")

    torch.save({
        "generator_state_dict": generator.state_dict(),
        "discriminator_state_dict": discriminator.state_dict(),
        "g_losses": g_losses,
        "d_losses": d_losses,
        "physics_losses": physics_losses,
    }, OUTPUT_DIR / "graph_gan_final.pt")
    logger.info(f"Model saved to {OUTPUT_DIR / 'graph_gan_final.pt'}")

    logger.info("Generating visualizations...")
    with torch.no_grad():
        z_vis = torch.randn(10, LATENT_DIM, device=DEVICE)
        x_v, ei_v, ea_v, batch_v, _ = generator(z_vis)
        fake_jets_vis = []
        for i in range(10):
            mask = batch_v == i
            n = int(mask.sum().item())
            if n == 0:
                continue
            x_i = x_v[mask]
            edge_mask = (batch_v[ei_v[0]] == i) & (batch_v[ei_v[1]] == i)
            ei_i = ei_v[:, edge_mask]
            local_idx = torch.zeros(batch_v.shape[0], dtype=torch.long, device=DEVICE)
            local_idx[mask] = torch.arange(n, device=DEVICE)
            ei_i = local_idx[ei_i]
            ea_i = ea_v[edge_mask]
            from torch_geometric.data import Data
            fake_jets_vis.append(Data(
                x=x_i.detach().cpu(),
                edge_index=ei_i.detach().cpu(),
                edge_attr=ea_i.detach().cpu(),
                y=torch.tensor([0]),
            ))
        for i in range(len(fake_jets_vis)):
            print(f"  Fake jet {i}: n={fake_jets_vis[i].x.shape[0]}, edges={fake_jets_vis[i].edge_index.shape[1]}")

    real_sample = random.sample(real_jets, min(10, len(real_jets)))
    for data in real_sample:
        data.x = inverse_transform_generator_output(data.x.clone())
    for data in fake_jets_vis:
        data.x = inverse_transform_generator_output(data.x.clone())
    plot_gan_convergence(g_losses, d_losses, physics_losses)
    plot_jet_comparison(real_sample, fake_jets_vis, n_samples=5)
    plot_particle_multiplicity(real_sample, fake_jets_vis)

    logger.info("Phase 2 (Generative Physics) artifacts generated.")

    evolution_file.close()


# ============================================================================
# PHASE 3 BRIDGE
# ============================================================================

def print_phase3_bridge() -> None:
    print("=" * 70)
    print("PHASE 2 -> PHASE 3 BRIDGE: Anomaly Detection via Reconstruction Error")
    print("=" * 70)
    print("""
The trained GraphGAN generator encodes the 'standard model' of jet physics.
In Phase 3 (Anomaly Detection), we will:

1. RECONSTRUCTION ERROR: For any new jet, compute the error between the
   original graph and the generator's reconstruction. High error = anomaly.

2. LATENT SPACE INTERPOLATION: Jets that fall in low-density regions of the
   latent space are candidates for new physics signatures.

3. WASSERSTEIN DISTANCE: Use the TDA-Wasserstein distance (from Phase 1) to
   quantify topological dissimilarity between real and reconstructed jets.

4. THRESHOLD SETTING: Use the validation distribution of reconstruction errors
   to set a 3-sigma threshold. Jets exceeding this threshold trigger an
   'Anomaly Alert' for human review.
""")
    print("=" * 70)


# ============================================================================
# MAIN
# ============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 2: GraphGAN for Jet Generation")
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr-g", type=float, default=LEARNING_RATE_G)
    parser.add_argument("--lr-d", type=float, default=LEARNING_RATE_D)
    parser.add_argument("--latent-dim", type=int, default=LATENT_DIM)
    parser.add_argument("--max-samples", type=int, default=5000)
    parser.add_argument("--physics-weight", type=float, default=PHYSICS_LOSS_WEIGHT)
    parser.add_argument("--cpu", action="store_true", help="Force CPU training")
    args = parser.parse_args()

    global DEVICE
    if args.cpu:
        DEVICE = torch.device("cpu")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 70)
    logger.info("PHASE 2: GENERATIVE PHYSICS - GraphGAN for Jet Synthesis")
    logger.info("=" * 70)
    logger.info(f"Device: {DEVICE}")
    logger.info(f"Latent dim: {args.latent_dim}")
    logger.info(f"Physics loss weight: {args.physics_weight}")

    logger.info(f"Loading real jets from {REAL_DATA_DIR}...")
    real_jets = load_real_jets(REAL_DATA_DIR, CACHE_DIR, max_samples=args.max_samples)
    logger.info(f"Loaded {len(real_jets)} real jets")

    generator = GraphGenerator(
        latent_dim=args.latent_dim,
        hidden_dim=HIDDEN_DIM,
        node_min=NODE_MIN,
        node_max=NODE_MAX,
    ).to(DEVICE)

    discriminator = GraphDiscriminator().to(DEVICE)

    logger.info(f"Generator parameters: {sum(p.numel() for p in generator.parameters()):,}")

    with torch.no_grad():
        dummy_data = real_jets[0].clone().to(DEVICE)
        _ = discriminator(dummy_data)
    logger.info(f"Discriminator parameters: {sum(p.numel() for p in discriminator.parameters()):,}")

    train_gan(
        generator=generator,
        discriminator=discriminator,
        real_jets=real_jets,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        lr_g=args.lr_g,
        lr_d=args.lr_d,
        physics_weight=args.physics_weight,
    )

    print_phase3_bridge()


if __name__ == "__main__":
    main()
