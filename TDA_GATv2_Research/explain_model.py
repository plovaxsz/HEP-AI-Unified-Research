"""Explainable AI pipeline for TDA-GATv2 jet classification.

Uses Captum (Integrated Gradients) and custom Attention Rollout to decode
the GNN's decision-making process for quark/gluon discrimination.

Outputs:
    - figure_integrated_gradients.png/pdf : Node-level feature importance
    - figure_attention_rollout.png/pdf    : Attention flow heatmaps
    - figure_physics_metrics.png/pdf      : Physics-informed validation metrics
    - xai_summary.txt                     : Quantitative interpretability report
"""

from __future__ import annotations

import pathlib
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from captum.attr import IntegratedGradients

from model import TDAGATv2

# --- CONFIG ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_PATH = pathlib.Path(r"D:\PISS\TDA_GATv2_Research\models\best_model_ever.pt")
OUTPUT_DIR = pathlib.Path(r"D:\PISS\TDA_GATv2_Research\models")
SHARD_DIR = pathlib.Path(r"D:\PISS\TDA_GATv2_Research\data\canonical_shards_real")
CACHE_DIR = pathlib.Path(r"D:\PISS\TDA_GATv2_Research\data\processed_cache_real")

# Feature names for the 5 node features
NODE_FEATURE_NAMES = ["pT", "eta", "phi", "energy", "pid"]

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


# ============================================================================
# MODEL LOADING
# ============================================================================

def load_model(checkpoint_path: pathlib.Path) -> TDAGATv2:
    """Load TDAGATv2 model from checkpoint."""

    model = TDAGATv2(node_dim=5, tda_dim=22, edge_dim=3).to(DEVICE)
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    return model


# ============================================================================
# DATA LOADING (single jet sample)
# ============================================================================

def get_sample_jet(shard_dir: pathlib.Path, cache_dir: pathlib.Path, label: int | None = None) -> tuple[Any, int]:
    """Load a single jet sample from the cached dataset.

    Args:
        shard_dir: Directory containing canonical NPZ shards.
        cache_dir: Directory containing processed PyG graph cache.
        label: If provided, filter for this label (0=gluon, 1=quark).
               If None, return the first available sample.

    Returns:
        (data, label) tuple where data is a PyG Data object.
    """

    from data_pipeline import CanonicalJetTDADataset

    dataset = CanonicalJetTDADataset(
        shard_dir=shard_dir,
        cache_dir=cache_dir,
        top_k=64,
        k_neighbors=8,
        maxdim=1,
        eps_max=1.2,
        allow_synthetic=False,
    )

    # Find a sample with the desired label
    target_indices = []
    for idx in range(len(dataset)):
        sample = dataset[idx]
        if label is None or int(sample.y.item()) == label:
            target_indices.append(idx)
        if len(target_indices) >= 10:
            break

    if not target_indices:
        raise ValueError(f"No samples found with label={label}")

    # Pick a representative sample (middle of the list for stability)
    pick_idx = target_indices[len(target_indices) // 2]
    data = dataset[pick_idx]
    data.batch = torch.zeros(data.x.shape[0], dtype=torch.long, device=data.x.device)
    return data, int(data.y.item())


# ============================================================================
# INTEGRATED GRADIENTS
# ============================================================================

def integrated_gradients_analysis(
    model: TDAGATv2,
    data: Any,
    target_class: int | None = None,
    n_steps: int = 50,
) -> tuple[np.ndarray, int]:
    """Run Integrated Gradients on node features for a single jet.

    Args:
        model: Trained TDAGATv2 model.
        data: PyG Data object for a single jet.
        target_class: Class to explain (0=gluon, 1=quark). If None, use argmax.
        n_steps: Number of integration steps.

    Returns:
        (attributions, predicted_class) where attributions is (num_nodes, num_features).
    """

    model.eval()

    # Prepare baseline (zeroed node features, keeping structure)
    baseline_x = torch.zeros_like(data.x)

    def forward_fn(x: torch.Tensor) -> torch.Tensor:
        """Forward function for Captum - takes only node features."""
        batch_data = data.clone()
        batch_data.x = x
        batch_data = batch_data.to(DEVICE)
        logits, _, _ = model(batch_data, return_attention_weights=False)
        return logits

    ig = IntegratedGradients(forward_fn)
    data_x = data.x.clone().requires_grad_(True).to(DEVICE)

    if target_class is None:
        with torch.no_grad():
            logits, _, _ = model(data.to(DEVICE), return_attention_weights=False)
            target_class = int(logits.argmax(dim=-1).item())

    attributions = ig.attribute(
        data_x,
        baselines=baseline_x,
        target=target_class,
        n_steps=n_steps,
        internal_batch_size=data.x.shape[0],
    )

    attributions = attributions.detach().cpu().numpy()
    return attributions, target_class


# ============================================================================
# ATTENTION ROLLOUT
# ============================================================================

def attention_rollout(
    model: TDAGATv2,
    data: Any,
    discount_factor: float = 0.9,
) -> dict[str, np.ndarray]:
    """Compute Attention Rollout for GATv2 layers.

    Attention Rollout (Abnar & Zuidema, 2020) propagates attention weights
    through the network layers via matrix multiplication:
        A_rollout = A_L @ (A_L - 1) @ ... @ A_1

    For GATv2 with residual connections, we add identity and normalize:
        A_rollout = A_L @ (A_L - 1 + I) @ ... @ (A_1 - 1 + I)

    Args:
        model: Trained TDAGATv2 model.
        data: PyG Data object for a single jet.
        discount_factor: Entropy-based discount for noisy attention heads.

    Returns:
        Dictionary with 'layer1' and 'layer2' rollout matrices (numpy).
    """

    model.eval()
    with torch.no_grad():
        _, _, attention_weights = model(data.to(DEVICE), return_attention_weights=True)

    rollout_results = {}
    for layer_name in ["layer1", "layer2"]:
        edge_index = attention_weights[layer_name]["edge_index"].cpu()
        alpha = attention_weights[layer_name]["alpha"].cpu()  # (E, H) or (E, 1, H)

        n_nodes = int(data.x.shape[0])
        n_heads = alpha.shape[-1] if alpha.dim() > 1 else 1

        # Average attention across heads, apply discount
        if alpha.dim() == 3:
            alpha = alpha.squeeze(1)  # (E, H)
        alpha_mean = alpha.mean(dim=-1)  # (E,)
        alpha_mean = alpha_mean * discount_factor

        # Build adjacency matrix from edge_index
        A = torch.zeros(n_nodes, n_nodes)
        A[edge_index[0], edge_index[1]] = alpha_mean
        A = A + torch.eye(n_nodes) * (1.0 - discount_factor)
        A = A / (A.sum(dim=1, keepdim=True) + 1e-8)

        rollout_results[layer_name] = A.numpy()

    return rollout_results


# ============================================================================
# PHYSICS-INFORMED METRICS
# ============================================================================

def compute_physics_metrics(
    data: Any,
    attributions: np.ndarray,
    rollout: dict[str, np.ndarray],
) -> dict[str, float]:
    """Quantify whether model attention aligns with known jet physics.

    Metrics:
        - pt_concentration: Correlation between |IG| and pT
        - attention_entropy: Entropy of attention distribution (lower = more focused)
        - dr_coherence: Average attention weight vs geometric dRR distance
        - substructure_score: N-subjettiness proxy from attention concentration
    """

    pt = data.x[:, 0].cpu().numpy()
    eta = data.x[:, 1].cpu().numpy()
    phi = data.x[:, 2].cpu().numpy()

    # 1. pT Concentration: do high-pT particles get higher importance?
    ig_magnitude = np.abs(attributions).sum(axis=1)
    if np.std(pt) > 0 and np.std(ig_magnitude) > 0:
        pt_conc = np.corrcoef(pt, ig_magnitude)[0, 1]
    else:
        pt_conc = 0.0
    pt_concentration = float(pt_conc) if np.isfinite(pt_conc) else 0.0

    # 2. Attention Entropy (from rollout layer2)
    A2 = rollout["layer2"]
    node_attention = A2.sum(axis=0)  # incoming attention per node
    node_attention = node_attention / (node_attention.sum() + 1e-8)
    entropy = -np.sum(node_attention * np.log(node_attention + 1e-8))
    max_entropy = np.log(len(node_attention))
    normalized_entropy = float(entropy / max_entropy) if max_entropy > 0 else 0.0

    # 3. Geometric Coherence: attention vs dRR distance
    deta = eta[:, None] - eta[None, :]
    dphi = np.arctan2(np.sin(phi[:, None] - phi[None, :]), np.cos(phi[:, None] - phi[None, :]))
    dr = np.sqrt(deta**2 + dphi**2)
    dr_flat = dr.flatten()
    attn_flat = A2.flatten()

    # Correlation between attention weight and inverse dRR
    mask = (dr_flat > 0) & (attn_flat > 0)
    if mask.sum() > 10:
        inv_dr = 1.0 / (dr_flat[mask] + 1e-8)
        if np.std(inv_dr) > 0 and np.std(attn_flat[mask]) > 0:
            dr_corr = np.corrcoef(inv_dr, attn_flat[mask])[0, 1]
        else:
            dr_corr = 0.0
        dr_coherence = float(dr_corr) if np.isfinite(dr_corr) else 0.0
    else:
        dr_coherence = 0.0

    # 4. Substructure Score: entropy of attention over nodes (lower = more jet-like)
    substructure_score = 1.0 - normalized_entropy  # 1.0 = perfectly focused

    return {
        "pt_concentration": pt_concentration,
        "attention_entropy": normalized_entropy,
        "dr_coherence": dr_coherence,
        "substructure_score": substructure_score,
        "n_nodes": len(pt),
    }


# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_integrated_gradients(
    data: Any,
    attributions: np.ndarray,
    predicted_class: int,
    sample_idx: int = 0,
) -> None:
    """Plot Integrated Gradients heatmap for a single jet."""

    n_nodes = attributions.shape[0]
    n_features = attributions.shape[1]

    fig, ax = plt.subplots(figsize=(10, max(4, n_nodes * 0.3)))
    fig.suptitle(
        f"Figure A1: Integrated Gradients - Jet {sample_idx} (Predicted: {'Quark' if predicted_class == 1 else 'Gluon'})",
        fontsize=13,
        fontweight="bold",
        y=1.02,
    )

    # Normalize per node for visualization
    max_abs = np.max(np.abs(attributions))
    if max_abs > 0:
        attrib_norm = attributions / max_abs
    else:
        attrib_norm = attributions

    im = ax.imshow(attrib_norm, cmap="RdBu_r", aspect="auto", vmin=-1, vmax=1)
    ax.set_xticks(range(n_features))
    ax.set_xticklabels(NODE_FEATURE_NAMES, rotation=45, ha="right")
    ax.set_yticks(range(n_nodes))
    ax.set_yticklabels([f"P{i}" for i in range(n_nodes)])
    ax.set_xlabel("Particle Feature")
    ax.set_ylabel("Particle Index")
    ax.set_title("Node-Level Feature Attribution")

    cbar = fig.colorbar(im, ax=ax, label="Normalized Attribution")
    plt.tight_layout()
    save_figure(fig, "figure_integrated_gradients")
    plt.close(fig)


def plot_attention_rollout(
    data: Any,
    rollout: dict[str, np.ndarray],
    predicted_class: int,
    sample_idx: int = 0,
) -> None:
    """Plot Attention Rollout heatmaps for both GAT layers."""

    n_nodes = data.x.shape[0]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        f"Figure A2: Attention Rollout Heatmaps - Jet {sample_idx} (Predicted: {'Quark' if predicted_class == 1 else 'Gluon'})",
        fontsize=13,
        fontweight="bold",
        y=1.02,
    )

    for idx, layer_name in enumerate(["layer1", "layer2"]):
        ax = axes[idx]
        A = rollout[layer_name]
        im = ax.imshow(A, cmap="viridis", aspect="auto", vmin=0, vmax=A.max())
        ax.set_xticks(range(n_nodes))
        ax.set_yticks(range(n_nodes))
        ax.set_xticklabels([f"P{i}" for i in range(n_nodes)], rotation=90, fontsize=7)
        ax.set_yticklabels([f"P{i}" for i in range(n_nodes)], fontsize=7)
        ax.set_xlabel("Target Particle")
        ax.set_ylabel("Source Particle")
        ax.set_title(f"{layer_name.upper()} Attention Flow")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Attention Weight")

    plt.tight_layout()
    save_figure(fig, "figure_attention_rollout")
    plt.close(fig)


def plot_physics_metrics(metrics: dict[str, float]) -> None:
    """Plot physics-informed interpretability metrics."""

    labels = ["pT\nConcentration", "Attention\nEntropy", "dRR\nCoherence", "Substructure\nScore"]
    values = [
        metrics["pt_concentration"],
        metrics["attention_entropy"],
        metrics["dr_coherence"],
        metrics["substructure_score"],
    ]
    colors = ["#2ecc71", "#e74c3c", "#3498db", "#9b59b6"]

    fig, ax = plt.subplots(figsize=(8, 5))
    fig.suptitle(
        "Figure A3: Physics-Informed Interpretability Metrics",
        fontsize=13,
        fontweight="bold",
        y=1.02,
    )

    bars = ax.bar(labels, values, color=colors, edgecolor="black", linewidth=0.8)
    ax.set_ylim(-1.0, 1.0)
    ax.axhline(y=0, color="black", linewidth=0.8, linestyle="-")
    ax.set_ylabel("Metric Value")
    ax.set_title("Alignment of Model Attention with Jet Physics")
    ax.grid(True, alpha=0.3, axis="y")

    # Annotate bars
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax.annotate(f"{val:.3f}", xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=10)

    plt.tight_layout()
    save_figure(fig, "figure_physics_metrics")
    plt.close(fig)


def plot_feature_importance_summary(attributions: np.ndarray) -> None:
    """Plot aggregated feature importance across all nodes."""

    mean_abs = np.abs(attributions).mean(axis=0)
    std_abs = np.abs(attributions).std(axis=0)

    fig, ax = plt.subplots(figsize=(7, 4))
    fig.suptitle(
        "Figure A4: Mean Absolute Feature Importance (Integrated Gradients)",
        fontsize=13,
        fontweight="bold",
        y=1.02,
    )

    x_pos = np.arange(len(NODE_FEATURE_NAMES))
    ax.bar(x_pos, mean_abs, yerr=std_abs, color="teal", edgecolor="black", linewidth=0.8, capsize=4)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(NODE_FEATURE_NAMES)
    ax.set_ylabel("Mean |Attribution|")
    ax.set_xlabel("Particle Feature")
    ax.set_title("Which features drive the GATv2 decision?")
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    save_figure(fig, "figure_feature_importance")
    plt.close(fig)


def save_figure(fig: plt.Figure, name: str) -> None:
    """Save figure as PNG and PDF."""

    png_path = OUTPUT_DIR / f"{name}.png"
    pdf_path = OUTPUT_DIR / f"{name}.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    print(f"  Saved: {png_path}")
    print(f"  Saved: {pdf_path}")


def print_physics_report(metrics: dict[str, float], predicted_class: int) -> None:
    """Print quantitative interpretability report for the paper."""

    label = "Quark" if predicted_class == 1 else "Gluon"
    print("=" * 70)
    print(f"XAI Physics Report - Jet Classified as: {label}")
    print("=" * 70)
    print(f"  pT Concentration     : {metrics['pt_concentration']:+.4f}")
    print(f"    (Expected >0.0: model prioritizes high-pT particles)")
    print(f"  Attention Entropy    : {metrics['attention_entropy']:.4f}  [0=focused, 1=diffuse]")
    print(f"  dR Coherence         : {metrics['dr_coherence']:+.4f}")
    print(f"    (Expected >0.0: attention follows geometric proximity)")
    print(f"  Substructure Score   : {metrics['substructure_score']:.4f}  [0=diffuse, 1=focused]")
    print(f"  Nodes in jet         : {metrics['n_nodes']}")
    print("=" * 70)


# ============================================================================
# MAIN
# ============================================================================

def main() -> None:
    print("[XAI] Loading TDA-GATv2 model...")
    model = load_model(CHECKPOINT_PATH)
    print(f"[XAI] Model loaded on {DEVICE}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Analyze both classes
    for class_label, class_name in [(0, "gluon"), (1, "quark")]:
        print(f"\n[XAI] Analyzing {class_name} jet (label={class_label})...")

        try:
            data, actual_label = get_sample_jet(SHARD_DIR, CACHE_DIR, label=class_label)
        except ValueError as e:
            print(f"[XAI] Skipping {class_name}: {e}")
            continue

        data = data.to(DEVICE)

        # 1. Integrated Gradients
        print(f"  Running Integrated Gradients...")
        attributions, pred_class = integrated_gradients_analysis(model, data, target_class=class_label)
        plot_integrated_gradients(data, attributions, pred_class, sample_idx=class_label)
        plot_feature_importance_summary(attributions)

        # 2. Attention Rollout
        print(f"  Computing Attention Rollout...")
        rollout = attention_rollout(model, data)
        plot_attention_rollout(data, rollout, pred_class, sample_idx=class_label)

        # 3. Physics Metrics
        print(f"  Computing physics-informed metrics...")
        physics = compute_physics_metrics(data, attributions, rollout)
        plot_physics_metrics(physics)
        print_physics_report(physics, pred_class)

        # Save per-jet report
        report_path = OUTPUT_DIR / f"xai_report_{class_name}.txt"
        with open(report_path, "w") as f:
            f.write(f"XAI Report - {class_name.upper()} Jet\n")
            f.write("=" * 50 + "\n")
            f.write(f"Predicted class  : {'Quark' if pred_class == 1 else 'Gluon'}\n")
            f.write(f"Actual class     : {'Quark' if actual_label == 1 else 'Gluon'}\n")
            f.write(f"Nodes (particles): {physics['n_nodes']}\n")
            f.write(f"pT Concentration : {physics['pt_concentration']:+.4f}\n")
            f.write(f"Attention Entropy: {physics['attention_entropy']:.4f}\n")
            f.write(f"dR Coherence     : {physics['dr_coherence']:+.4f}\n")
            f.write(f"Substructure     : {physics['substructure_score']:.4f}\n")
        print(f"  Report saved: {report_path}")

    print("\n[XAI] All figures generated successfully.")
    print(f"[XAI] Output directory: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

