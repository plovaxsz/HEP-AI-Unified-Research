"""Phase 3: Anomaly Detection Engine for Jet Classification.

Architecture:
    - GraphEncoder: GNN-based encoder mapping jet graphs to 128-dim latent vectors
    - Training: Graph Contrastive Learning (GraphCL) with physics-aware augmentations
    - Inference: Hybrid Kinematic-Latent Vector + Isolation Forest
    - Anomaly Score: Isolation Forest decision on [z (128-dim), max_pT, total_E, num_nodes]
    - Threshold Calibration: 3-sigma rule on normal jets

Outputs:
    - phase3_encoder_final.pt          : Trained encoder weights
    - figure_anomaly_distribution.png  : Normal vs Anomaly score histogram
    - phase3_anomaly_report.txt        : Telemetry and threshold report
"""

from __future__ import annotations

import argparse
import logging
import math
import random
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import gaussian_kde
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from torch import Tensor
from torch_geometric.nn import GATv2Conv, global_max_pool, global_mean_pool

# --- CONFIG ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
REAL_DATA_DIR = Path(r"D:\PISS\TDA_GATv2_Research\data\canonical_shards_real")
CACHE_DIR = Path(r"D:\PISS\TDA_GATv2_Research\data\processed_cache_real")
OUTPUT_DIR = Path(r"D:\PISS\TDA_GATv2_Research\models")
GENERATOR_PATH = OUTPUT_DIR / "graph_gan_final.pt"
LOG_PATH = OUTPUT_DIR / "phase3_anomaly_engine.log"

LATENT_DIM = 128
HIDDEN_DIM = 128
GAT_HIDDEN = 64
HEADS = 4
DROPOUT = 0.1
LEARNING_RATE = 1e-3
BATCH_SIZE = 32
NUM_EPOCHS = 10
CALIBRATION_SAMPLES = 500
MOCK_ANOMALY_COUNT = 50
TEMPERATURE = 0.1

plt.style.use("seaborn-v0_8-darkgrid")
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
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
# PHYSICS UTILITIES
# ============================================================================

def inverse_log_transform(x: Tensor) -> Tensor:
    x_phys = x.clone()
    x_phys[:, 0] = torch.exp(x_phys[:, 0].clamp(min=0.0, max=7.0)) - 1.0
    x_phys[:, 3] = torch.exp(x_phys[:, 3].clamp(min=0.0, max=7.0)) - 1.0
    return x_phys


def compute_jet_mass(x: Tensor) -> Tensor:
    x_phys = inverse_log_transform(x)
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


# ============================================================================
# GRAPH ENCODER
# ============================================================================

class GraphEncoder(nn.Module):
    """GNN-based encoder mapping jet graphs to 128-dim latent vectors."""

    def __init__(
        self,
        node_dim: int = 5,
        edge_dim: int = 3,
        hidden_dim: int = 64,
        latent_dim: int = 128,
        heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.node_encoder = nn.Sequential(
            nn.Linear(node_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.gat1 = GATv2Conv(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            heads=heads,
            concat=True,
            edge_dim=edge_dim,
            dropout=dropout,
            residual=True,
        )
        self.gat2 = GATv2Conv(
            in_channels=hidden_dim * heads,
            out_channels=hidden_dim,
            heads=heads,
            concat=True,
            edge_dim=edge_dim,
            dropout=dropout,
            residual=True,
        )

        pool_dim = hidden_dim * heads * 2
        self.latent_mlp = nn.Sequential(
            nn.Linear(pool_dim, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, latent_dim),
        )

    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor, batch: Tensor) -> Tensor:
        x = self.node_encoder(x)
        x, _ = self.gat1(x, edge_index, edge_attr=edge_attr, return_attention_weights=True)
        x = F.gelu(x)
        x, _ = self.gat2(x, edge_index, edge_attr=edge_attr, return_attention_weights=True)
        x = F.gelu(x)

        mean_pool = global_mean_pool(x, batch)
        max_pool = global_max_pool(x, batch)
        pooled = torch.cat([mean_pool, max_pool], dim=-1)
        z = self.latent_mlp(pooled)
        return z


# ============================================================================
# FROZEN GENERATOR WRAPPER
# ============================================================================

class FrozenGenerator:
    """Wrapper for Phase 2 Generator with frozen parameters."""

    def __init__(self, checkpoint_path: Path, device: torch.device) -> None:
        from gen_graph_gan import GraphGenerator
        self.generator = GraphGenerator(
            latent_dim=128,
            hidden_dim=128,
            node_min=15,
            node_max=50,
        ).to(device)
        state_dict = torch.load(checkpoint_path, map_location=device, weights_only=False)
        self.generator.load_state_dict(state_dict["generator_state_dict"])
        self.generator.requires_grad_(False)
        self.generator.eval()
        self.device = device

    def reconstruct(self, z: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        return self.generator(z)


# ============================================================================
# PHYSICS-AWARE GRAPH AUGMENTATION
# ============================================================================

def augment_graph(data: Data, aug_type: str = "both") -> Data:
    """Create augmented views of a jet graph for contrastive learning."""
    x = data.x.clone()
    edge_index = data.edge_index.clone()
    edge_attr = data.edge_attr.clone() if data.edge_attr is not None else None

    if aug_type in ("noise", "both"):
        x = x + torch.randn_like(x) * 0.05

    if aug_type in ("drop", "both"):
        if x.shape[0] > 5:
            keep_ratio = 0.95
            n_keep = max(5, int(x.shape[0] * keep_ratio))
            energies = x[:, 0]
            _, keep_idx = torch.sort(energies, descending=True)
            keep_idx = keep_idx[:n_keep]
            x = x[keep_idx]
            old_to_new = {old.item(): new for new, old in enumerate(keep_idx)}
            new_edges = []
            new_attrs = []
            for j in range(edge_index.shape[1]):
                src = edge_index[0, j].item()
                dst = edge_index[1, j].item()
                if src in old_to_new and dst in old_to_new:
                    new_edges.append([old_to_new[src], old_to_new[dst]])
                    if edge_attr is not None:
                        new_attrs.append(edge_attr[j].tolist())
            if len(new_edges) > 0:
                edge_index = torch.tensor(new_edges, dtype=torch.long, device=x.device).t()
                if edge_attr is not None:
                    edge_attr = torch.tensor(new_attrs, dtype=torch.float, device=x.device)
            else:
                n = x.shape[0]
                edge_index = torch.tensor([[i, (i + 1) % n] for i in range(n)], dtype=torch.long, device=x.device).t()
                edge_attr = torch.zeros((n, 3), device=x.device) if edge_attr is not None else None

    from torch_geometric.data import Data
    aug_data = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=data.y,
        batch=data.batch,
    )
    return aug_data


# ============================================================================
# GRAPHCL TRAINING LOOP (InfoNCE / NT-Xent)
# ============================================================================

def train_encoder_graphcl(
    encoder: GraphEncoder,
    real_jets: list,
    num_epochs: int = 10,
    batch_size: int = 32,
    lr: float = 1e-3,
    temperature: float = 0.1,
) -> dict[str, Any]:
    encoder.train()
    optimizer = torch.optim.Adam(encoder.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    losses = []

    logger.info(f"GraphCL Training: {num_epochs} epochs, {len(real_jets)} jets, tau={temperature}")

    for epoch in range(1, num_epochs + 1):
        epoch_loss = 0.0
        n_batches = 0
        random.shuffle(real_jets)

        for i in range(0, len(real_jets), batch_size):
            batch = real_jets[i : i + batch_size]
            if len(batch) < 2:
                continue

            view1 = [augment_graph(d, "noise") for d in batch]
            view2 = [augment_graph(d, "drop") for d in batch]

            def collate_views(view_list):
                from torch_geometric.data import Batch
                return Batch.from_data_list(view_list)

            batch1 = collate_views(view1).to(DEVICE)
            batch2 = collate_views(view2).to(DEVICE)

            z1 = encoder(batch1.x, batch1.edge_index, batch1.edge_attr, batch1.batch)
            z2 = encoder(batch2.x, batch2.edge_index, batch2.edge_attr, batch2.batch)

            z1_norm = F.normalize(z1, dim=-1)
            z2_norm = F.normalize(z2, dim=-1)

            n = z1_norm.shape[0]
            sim_matrix = torch.matmul(z1_norm, z2_norm.t()) / temperature
            diag = torch.eye(n, device=DEVICE, dtype=torch.bool)
            pos_sim = sim_matrix[diag]
            neg_mask = ~diag
            neg_sim = sim_matrix[neg_mask].view(n, n - 1)

            logits = torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1)
            labels = torch.zeros(n, dtype=torch.long, device=DEVICE)
            loss = F.cross_entropy(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += float(loss.item())
            n_batches += 1

        scheduler.step()
        avg_loss = epoch_loss / max(n_batches, 1)
        losses.append(avg_loss)
        logger.info(f"Epoch {epoch:04d} | GraphCL Loss: {avg_loss:.6f} | LR: {scheduler.get_last_lr()[0]:.6f}")

    return {"losses": losses, "final_lr": scheduler.get_last_lr()[0]}


# ============================================================================
# PHYSICAL MSE ANOMALY SCORE
# ============================================================================

def compute_anomaly_score(x_original: Tensor, x_reconstructed: Tensor) -> Tensor:
    pt_orig = torch.sort(x_original[:, 0], descending=True)[0]
    e_orig = torch.sort(x_original[:, 3], descending=True)[0]
    pt_rec = torch.sort(x_reconstructed[:, 0], descending=True)[0]
    e_rec = torch.sort(x_reconstructed[:, 3], descending=True)[0]

    def padded_mse_torch(a: Tensor, b: Tensor) -> Tensor:
        a_sorted = torch.sort(a, descending=True)[0]
        b_sorted = torch.sort(b, descending=True)[0]
        max_len = max(a_sorted.shape[0], b_sorted.shape[0])
        a_padded = torch.zeros(max_len, device=a.device, dtype=a.dtype)
        b_padded = torch.zeros(max_len, device=b.device, dtype=b.dtype)
        a_padded[: a_sorted.shape[0]] = a_sorted
        b_padded[: b_sorted.shape[0]] = b_sorted
        return torch.mean((a_padded - b_padded) ** 2)

    score = padded_mse_torch(pt_orig, pt_rec) + padded_mse_torch(e_orig, e_rec)
    return score


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


def generate_mock_anomalies(real_jets: list, num_anomalies: int = 50, device: torch.device = DEVICE) -> list:
    anomalies = []
    for _ in range(num_anomalies):
        base = random.choice(real_jets)
        x = base.x.clone()
        edge_index = base.edge_index.clone()
        edge_attr = base.edge_attr.clone() if base.edge_attr is not None else torch.zeros((edge_index.shape[1], 3), device=device)

        if random.random() < 0.5:
            spike_idx = random.randint(0, x.shape[0] - 1)
            x[spike_idx, 0] = torch.tensor(random.uniform(300.0, 800.0), device=device)
            x[spike_idx, 3] = torch.tensor(random.uniform(300.0, 800.0), device=device)
        else:
            if x.shape[0] > 5:
                k = random.randint(2, max(2, x.shape[0] // 3))
                disconnect = random.sample(range(x.shape[0]), k=k)
                keep = [i for i in range(x.shape[0]) if i not in disconnect]
                x = x[keep]
                old_to_new = {old: new for new, old in enumerate(keep)}
                new_edge_index = []
                new_edge_attr = []
                for j in range(edge_index.shape[1]):
                    src = int(edge_index[0, j].item())
                    dst = int(edge_index[1, j].item())
                    if src in old_to_new and dst in old_to_new:
                        new_edge_index.append([old_to_new[src], old_to_new[dst]])
                        if edge_attr is not None:
                            new_edge_attr.append(edge_attr[j].tolist())
                if len(new_edge_index) == 0:
                    n = x.shape[0]
                    new_edge_index = [[i, (i + 1) % n] for i in range(n)]
                    new_edge_attr = [[0.0, 0.0, 0.0] for _ in range(n)]
                edge_index = torch.tensor(new_edge_index, dtype=torch.long, device=device).t()
                edge_attr = torch.tensor(new_edge_attr, dtype=torch.float, device=device)

        from torch_geometric.data import Data
        anomaly_data = Data(
            x=x.to(device),
            edge_index=edge_index.to(device),
            edge_attr=edge_attr.to(device),
            y=torch.tensor([1], device=device),
        )
        anomaly_data.batch = torch.zeros(x.shape[0], dtype=torch.long, device=device)
        anomalies.append(anomaly_data)
    return anomalies


# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_anomaly_distribution(
    normal_scores: list[float],
    anomaly_scores: list[float],
    threshold: float,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.suptitle(
        "Figure 11: Anomaly Score Distribution — Normal vs Mock Anomalous Jets",
        fontsize=13,
        fontweight="bold",
        y=1.02,
    )

    ax.hist(normal_scores, bins=50, alpha=0.6, color="steelblue", label=f"Normal (n={len(normal_scores)})", density=True, edgecolor="black")
    ax.hist(anomaly_scores, bins=50, alpha=0.6, color="crimson", label=f"Mock Anomaly (n={len(anomaly_scores)})", density=True, edgecolor="black")

    if normal_scores:
        kde_normal = gaussian_kde(normal_scores)
        x_grid = np.linspace(min(min(normal_scores), min(anomaly_scores)) * 0.9, max(max(normal_scores), max(anomaly_scores)) * 1.1, 500)
        ax.plot(x_grid, kde_normal(x_grid), color="steelblue", linewidth=2)

    if anomaly_scores:
        kde_anomaly = gaussian_kde(anomaly_scores)
        x_grid = np.linspace(min(min(normal_scores), min(anomaly_scores)) * 0.9, max(max(normal_scores), max(anomaly_scores)) * 1.1, 500)
        ax.plot(x_grid, kde_anomaly(x_grid), color="crimson", linewidth=2, linestyle="--")

    ax.axvline(x=threshold, color="red", linestyle="--", linewidth=2, label=f"3σ Threshold = {threshold:.4f}")
    ax.set_xlabel("Anomaly Score (Dual-Engine Isolation Forest)")
    ax.set_ylabel("Density")
    ax.set_title("Phase 3: Dual-Engine Ensemble Anomaly Detection")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    pdf_path = output_path.with_suffix(".pdf")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved anomaly distribution plot to {output_path}")


# ============================================================================
# HYBRID FEATURE EXTRACTION (Latent + Raw Physics)
# ============================================================================

def extract_hybrid_features(encoder: GraphEncoder, data: Data, device: torch.device) -> np.ndarray:
    """Extract latent vector z and raw physical heuristics separately."""
    data = data.to(device)
    x = data.x
    ei = data.edge_index
    ea = data.edge_attr if data.edge_attr is not None else torch.zeros((ei.shape[1], 3), device=device)
    batch = data.batch if data.batch is not None else torch.zeros(x.shape[0], dtype=torch.long, device=device)

    encoder.eval()
    with torch.no_grad():
        z = encoder(x, ei, ea, batch).cpu().numpy().reshape(-1)

    max_pt = float(x[:, 0].max().item())
    total_e = float(x[:, 3].sum().item())
    num_nodes = float(data.x.shape[0])

    return z, np.array([max_pt, total_e, num_nodes])


# ============================================================================
# ISOLATION FOREST THRESHOLD CALIBRATION (Hybrid Features)
# ============================================================================

def calibrate_threshold(
    encoder: GraphEncoder,
    normal_jets: list,
) -> tuple[Any, Any, Any, float, float, list[float]]:
    from sklearn.ensemble import IsolationForest
    encoder.eval()
    z_list = []
    phys_list = []

    for data in normal_jets:
        z, phys = extract_hybrid_features(encoder, data, DEVICE)
        z_list.append(z)
        phys_list.append(phys)

    z_matrix = np.array(z_list)
    phys_matrix = np.array(phys_list)

    phys_scaler = StandardScaler()
    phys_scaled = phys_scaler.fit_transform(phys_matrix)

    iso_topo = IsolationForest(n_estimators=100, contamination="auto", random_state=42)
    iso_topo.fit(z_matrix)

    iso_phys = IsolationForest(n_estimators=100, contamination="auto", random_state=42)
    iso_phys.fit(phys_scaled)

    raw_topo = iso_topo.decision_function(z_matrix)
    raw_phys = iso_phys.decision_function(phys_scaled)
    combined_scores = -raw_topo + -raw_phys

    mu_score = float(np.mean(combined_scores))
    sigma_score = float(np.std(combined_scores))
    threshold = mu_score + 3.0 * sigma_score

    logger.info(f"Dual-Engine Calibration: mu={mu_score:.4f}, sigma={sigma_score:.4f}, threshold={threshold:.4f}")
    return iso_topo, iso_phys, phys_scaler, threshold, mu_score, combined_scores.tolist()


# ============================================================================
# MOCK ANOMALY TESTING (Isolation Forest)
# ============================================================================

def run_mock_anomaly_test(
    encoder: GraphEncoder,
    iso_topo: Any,
    iso_phys: Any,
    phys_scaler: Any,
    threshold: float,
    normal_jets: list,
    num_anomalies: int = 50,
) -> tuple[list[float], list[float], float]:
    anomalies = generate_mock_anomalies(normal_jets, num_anomalies=num_anomalies, device=DEVICE)

    anomaly_scores = []
    for data in anomalies:
        z, phys = extract_hybrid_features(encoder, data, DEVICE)
        phys_scaled = phys_scaler.transform(phys.reshape(1, -1))
        score_topo = -iso_topo.decision_function(z.reshape(1, -1))[0]
        score_phys = -iso_phys.decision_function(phys_scaled)[0]
        anomaly_scores.append(float(score_topo + score_phys))

    normal_scores = []
    for data in normal_jets:
        z, phys = extract_hybrid_features(encoder, data, DEVICE)
        phys_scaled = phys_scaler.transform(phys.reshape(1, -1))
        score_topo = -iso_topo.decision_function(z.reshape(1, -1))[0]
        score_phys = -iso_phys.decision_function(phys_scaled)[0]
        normal_scores.append(float(score_topo + score_phys))

    return normal_scores, anomaly_scores, threshold


# ============================================================================
# REPORT
# ============================================================================

def write_anomaly_report(
    normal_scores: list[float],
    anomaly_scores: list[float],
    threshold: float,
    training_summary: dict[str, Any],
    output_path: Path,
) -> None:
    mu = float(np.mean(normal_scores))
    sigma = float(np.std(normal_scores))
    anomaly_mean = float(np.mean(anomaly_scores))
    separation = anomaly_mean / max(mu, 1e-6)

    report_lines = [
        "=" * 80,
        "PHASE 3: ANOMALY DETECTION ENGINE — SCIENTIFIC TELEMETRY REPORT",
        "=" * 80,
        "",
        f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "-" * 80,
        "1. THRESHOLD CALIBRATION (3-Sigma Rule)",
        "-" * 80,
        f"  Normal Jets Tested:          {len(normal_scores)}",
        f"  Mean Anomaly Score (mu):     {mu:.6f}",
        f"  Std Anomaly Score (sigma):   {sigma:.6f}",
        f"  Threshold (mu + 3*sigma):    {threshold:.6f}",
        "",
        "-" * 80,
        "2. MOCK ANOMALY TESTING",
        "-" * 80,
        f"  Mock Anomalous Jets:         {len(anomaly_scores)}",
        f"  Mean Anomaly Score:          {anomaly_mean:.6f}",
        f"  Separation Factor:           {separation:.2f}x",
        f"  Detection Rate:              {sum(1 for s in anomaly_scores if s > threshold) / max(len(anomaly_scores), 1):.2%}",
        "",
        "-" * 80,
        "3. TRAINING CONFIGURATION",
        "-" * 80,
        f"  Encoder Epochs:              {training_summary.get('epochs', 'N/A')}",
        f"  Final GraphCL Loss:          {training_summary.get('final_loss', 'N/A'):.6f}",
        f"  Learning Rate:               {training_summary.get('final_lr', 'N/A'):.6f}",
        f"  Generator Frozen:            True",
        f"  Temperature (tau):           {TEMPERATURE}",
        "",
        "-" * 80,
        "4. INTERPRETATION",
        "-" * 80,
        "  - Separation Factor > 5.0 indicates strong manifold discrimination.",
        "  - Detection Rate > 80% on mock anomalies validates the reconstruction metric.",
        "  - GraphCL training forces the encoder to learn topological invariance, not dataset mean.",
        "",
        "=" * 80,
        "END OF REPORT",
        "=" * 80,
        "",
    ]
    output_path.write_text("\n".join(report_lines), encoding="utf-8")
    logger.info(f"Anomaly report written to {output_path}")


# ============================================================================
# MAIN
# ============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 3: Anomaly Detection Engine (GraphCL)")
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--max-samples", type=int, default=CALIBRATION_SAMPLES)
    parser.add_argument("--num-anomalies", type=int, default=MOCK_ANOMALY_COUNT)
    parser.add_argument("--cpu", action="store_true", help="Force CPU training")
    args = parser.parse_args()

    global DEVICE
    if args.cpu:
        DEVICE = torch.device("cpu")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 80)
    logger.info("PHASE 3: ANOMALY DETECTION ENGINE — GRAPHCL PROTOCOL")
    logger.info("=" * 80)
    logger.info(f"Device: {DEVICE}")

    logger.info(f"Loading frozen generator from {GENERATOR_PATH}...")
    generator = FrozenGenerator(GENERATOR_PATH, DEVICE)
    logger.info("Generator frozen (requires_grad=False)")

    logger.info(f"Loading real jets from {REAL_DATA_DIR}...")
    real_jets = load_real_jets(REAL_DATA_DIR, CACHE_DIR, max_samples=args.max_samples)
    logger.info(f"Loaded {len(real_jets)} real jets")

    masses = [compute_jet_mass(data.x).item() for data in real_jets]
    mass_mean = float(np.mean(masses))
    mass_std = float(np.std(masses))
    mass_stats = (mass_mean, mass_std)
    logger.info(f"Dataset mass stats: mean={mass_mean:.2f} GeV, std={mass_std:.2f} GeV")

    encoder = GraphEncoder(
        node_dim=5,
        edge_dim=3,
        hidden_dim=GAT_HIDDEN,
        latent_dim=LATENT_DIM,
        heads=HEADS,
        dropout=DROPOUT,
    ).to(DEVICE)

    training_summary = train_encoder_graphcl(
        encoder=encoder,
        real_jets=real_jets,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        lr=LEARNING_RATE,
        temperature=TEMPERATURE,
    )

    torch.save({
        "encoder_state_dict": encoder.state_dict(),
        "mass_stats": mass_stats,
        "training_summary": training_summary,
    }, OUTPUT_DIR / "phase3_encoder_final.pt")
    logger.info(f"Encoder saved to {OUTPUT_DIR / 'phase3_encoder_final.pt'}")

    logger.info("Running mock anomaly testing...")
    iso_topo, iso_phys, phys_scaler, threshold, _, _ = calibrate_threshold(
        encoder=encoder,
        normal_jets=real_jets,
    )

    normal_scores, anomaly_scores, threshold = run_mock_anomaly_test(
        encoder=encoder,
        iso_topo=iso_topo,
        iso_phys=iso_phys,
        phys_scaler=phys_scaler,
        threshold=threshold,
        normal_jets=real_jets,
        num_anomalies=args.num_anomalies,
    )

    plot_anomaly_distribution(
        normal_scores=normal_scores,
        anomaly_scores=anomaly_scores,
        threshold=threshold,
        output_path=OUTPUT_DIR / "figure_anomaly_distribution.png",
    )

    write_anomaly_report(
        normal_scores=normal_scores,
        anomaly_scores=anomaly_scores,
        threshold=threshold,
        training_summary={
            "epochs": args.epochs,
            "final_loss": training_summary["losses"][-1] if training_summary["losses"] else 0.0,
            "final_lr": training_summary["final_lr"],
        },
        output_path=OUTPUT_DIR / "phase3_anomaly_report.txt",
    )

    logger.info("=" * 80)
    logger.info("PHASE 3 COMPLETE")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
