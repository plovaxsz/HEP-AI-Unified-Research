"""
Phase 6: Quantum-Conditioned Generative Validation (QCGV)

Architecture:
    - Conditional DDPM that reconstructs 4-vector kinematics [p_T, eta, phi, m]
    - Conditioned on Quantum Attention Matrix rows from Phase 5
    - Conditioning Encoder: Attention row (139) -> Quantum Context Vector (16)
    - Noise Predictor MLP: [x_t (4), t_emb (32), c (16)] -> 4
    - Hardware: VRAM-friendly for RTX 3060ti (8GB)

Framework: PyTorch 2.x + sklearn QuantileTransformer + Log-Manifold + PINN
"""

from __future__ import annotations

import argparse
import logging
import math
import time
import sys
from pathlib import Path
from typing import Dict, Tuple, Optional, List

import matplotlib.pyplot as plt
import numpy as np
import pennylane as qml
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import QuantileTransformer
from torch import Tensor
from tqdm import tqdm

# ----------------------------------------------------------------------
# Hyper-parameters & constants
# ----------------------------------------------------------------------
EPSILON = 1e-6                     # for log-transform & positivity guarantee

BASE_DIR = Path(__file__).resolve().parent
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUTPUT_DIR = BASE_DIR / "models"
LOG_DIR = BASE_DIR / "logs"
LOG_PATH = LOG_DIR / "phase6_execution.log"
REPORT_PATH = OUTPUT_DIR / "phase6_qcgv_report.txt"
FIGURE_PATH = OUTPUT_DIR / "figure_phase6_money_plot.png"
PHASE4_EMBEDDINGS = BASE_DIR / "data" / "phase4_quantum_embeddings.pt"
PHASE5_ATTENTION = BASE_DIR / "data" / "phase5_attention_matrices.pt"
OUTPUT_GENERATED = BASE_DIR / "data" / "phase6_generated_kinematics.pt"
MODEL_WEIGHTS_PATH = OUTPUT_DIR / "best_phase6_ddpm.pth"

DEFAULT_NUM_JETS = 5000
DEFAULT_BATCH_SIZE = 512
DEFAULT_EPOCHS = 100
DIFFUSION_T = 1000
COND_DIM = 16
TIME_EMB_DIM = 32
N_QUANTILES = 10000

# ----------------------------------------------------------------------
# Logging setup
# ----------------------------------------------------------------------
LOG_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

logger = logging.getLogger(__name__)
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)
    _log = logging.FileHandler(LOG_PATH, encoding="utf-8")
    _log.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(_log)


# ----------------------------------------------------------------------
# Hardware diagnostics
# ----------------------------------------------------------------------
def diagnose_hardware() -> None:
    """Print hardware status for debugging and reproducibility."""
    if DEVICE.type == "cuda":
        props = torch.cuda.get_device_properties(DEVICE)
        vram = props.total_memory / (1024 ** 3)
        logger.info(f"CUDA Device: {props.name}")
        logger.info(f"VRAM Available: {vram:.2f} GB")
    else:
        logger.info("Device: CPU (CUDA not available)")


# ----------------------------------------------------------------------
# Time embedding
# ----------------------------------------------------------------------
class SinusoidalTimeEmbedding(nn.Module):
    """
    Sinusoidal timestep embedding for diffusion process.
    
    Implements the standard sinusoidal embedding from "Attention Is All You Need"
    but applied to diffusion timesteps.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: Tensor) -> Tensor:
        device = t.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * (-emb))
        emb = t[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb


# ----------------------------------------------------------------------
# Conditioning encoder
# ----------------------------------------------------------------------
class ConditioningEncoder(nn.Module):
    """
    Compresses attention row (139) into Quantum Context Vector (16).
    
    Uses SiLU activation and LayerNorm for stable gradient flow.
    """

    def __init__(self, input_dim: int = 139, hidden_dim: int = 64, output_dim: int = COND_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, output_dim),
            nn.SiLU(),
            nn.LayerNorm(output_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


# ----------------------------------------------------------------------
# Noise predictor MLP (upgraded architecture)
# ----------------------------------------------------------------------
class NoisePredictor(nn.Module):
    """
    Epsilon_theta: predicts noise from x_t, t, and conditioning context.
    
    Architecture: 52 -> 512 -> 512 -> 256 -> 4 with GELU activations.
    The input dimension of 52 comes from 4 (x_t) + 32 (t_emb) + 16 (cond).
    """

    def __init__(self, cond_dim: int = COND_DIM, time_emb_dim: int = TIME_EMB_DIM):
        super().__init__()
        self.time_emb = SinusoidalTimeEmbedding(time_emb_dim)
        self.cond_encoder = ConditioningEncoder()

        input_dim = 4 + time_emb_dim + cond_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.GELU(),
            nn.Linear(512, 512),
            nn.GELU(),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Linear(256, 4),
        )

    def forward(self, x_t: Tensor, t: Tensor, cond: Tensor) -> Tensor:
        t_emb = self.time_emb(t)
        c = self.cond_encoder(cond)
        h = torch.cat([x_t, t_emb, c], dim=-1)
        return self.net(h)


# ----------------------------------------------------------------------
# Diffusion process helpers
# ----------------------------------------------------------------------
def get_alphas(beta_start: float = 1e-4, beta_end: float = 0.02, t: int = DIFFUSION_T) -> Tuple[Tensor, Tensor]:
    """
    Compute alpha values for DDPM.
    
    Returns:
        alpha: noise schedule
        alpha_bar: cumulative product of alpha
    """
    beta = torch.linspace(beta_start, beta_end, t, device=DEVICE)
    alpha = 1.0 - beta
    alpha_bar = torch.cumprod(alpha, dim=0)
    return alpha.to(DEVICE), alpha_bar.to(DEVICE)


def add_noise(x_0: Tensor, t: Tensor, alpha_bar: Tensor) -> Tuple[Tensor, Tensor]:
    """
    Add noise to x_0 following the DDPM forward process.
    
    Args:
        x_0: clean data tensor [B, 4]
        t: timestep indices [B]
        alpha_bar: cumulative alpha values
        
    Returns:
        x_t: noisy tensor
        noise: the added noise (for loss computation)
    """
    noise = torch.randn_like(x_0)
    sqrt_alpha_bar = torch.sqrt(alpha_bar[t]).unsqueeze(1)
    sqrt_one_minus = torch.sqrt(1.0 - alpha_bar[t]).unsqueeze(1)
    x_t = sqrt_alpha_bar * x_0 + sqrt_one_minus * noise
    return x_t, noise


def sample_reverse(
    model: nn.Module,
    cond: Tensor,
    alpha: Tensor,
    alpha_bar: Tensor,
    scaler: QuantileTransformer,
    t_total: int = DIFFUSION_T,
) -> Tuple[Tensor, Tensor]:
    """
    Generate samples by reverse diffusion process.
    
    Returns:
        x_original: kinematics in original scale [p_T, eta, phi, m]
        x_normalized: intermediate normalized tensor
    """
    model.eval()
    with torch.no_grad():
        x_t = torch.randn(cond.shape[0], 4, device=DEVICE, dtype=torch.float32)
        for t in tqdm(range(t_total - 1, -1, -1), desc="Sampling", leave=False):
            t_batch = torch.full((cond.shape[0],), t, device=DEVICE, dtype=torch.long)
            eps = model(x_t, t_batch, cond)
            
            # Robust handling of NaN and Inf in model output
            if torch.isnan(eps).any() or torch.isinf(eps).any():
                logger.warning(f"NaN/Inf detected at t={t}, replacing with zeros")
                eps = torch.zeros_like(eps)
            
            # Clamp extreme values to prevent explosion
            eps = torch.clamp(eps, min=-10.0, max=10.0)
            
            alpha_t = alpha[t]
            alpha_bar_t = alpha_bar[t]
            
            # Prevent division by zero or near-zero values
            alpha_t_safe = torch.clamp(alpha_t, min=1e-8)
            one_minus_alpha_bar_t = torch.clamp(1.0 - alpha_bar_t, min=1e-8)
            
            factor = (1.0 - alpha_t) / torch.sqrt(one_minus_alpha_bar_t + 1e-8)
            x_t = (1.0 / torch.sqrt(alpha_t_safe + 1e-8)) * (x_t - factor * eps)
            
            if t > 0:
                sigma = torch.sqrt(torch.clamp((1.0 - alpha_t) / (one_minus_alpha_bar_t + 1e-8), min=1e-8))
                # Add noise with clamping to prevent extreme values
                noise = torch.randn_like(x_t) * sigma
                noise = torch.clamp(noise, min=-10.0, max=10.0)
                x_t = x_t + noise
            
            # Additional stability check for x_t
            x_t = torch.clamp(x_t, min=-50.0, max=50.0)

        # Move to CPU for sklearn operations
        x_np = x_t.detach().cpu().numpy()
        
        # Final check for NaN/Inf before sklearn transform
        if np.isnan(x_np).any() or np.isinf(x_np).any():
            logger.warning("NaN/Inf detected in samples before inverse transform, clamping")
            x_np = np.nan_to_num(x_np, nan=0.0, posinf=1e4, neginf=-1e4)
            x_np = np.clip(x_np, -1e4, 1e4)
        
        try:
            x_jittered_log = scaler.inverse_transform(x_np)
        except ValueError as e:
            logger.warning(f"QuantileTransformer inverse failed: {e}. Using identity transform.")
            x_jittered_log = x_np  # Fallback to identity if transform fails
        
        x_log = x_jittered_log.copy()
        # Apply inverse log-transform with safety checks
        x_log[:, 0] = np.exp(np.clip(x_log[:, 0], -20, 20)) - EPSILON
        x_log[:, 3] = np.exp(np.clip(x_log[:, 3], -20, 20)) - EPSILON
        x_log[:, 0] = np.maximum(x_log[:, 0], 0.0)  # Ensure non-negative
        x_log[:, 3] = np.maximum(x_log[:, 3], 0.0)  # Ensure non-negative
        
        x_original = torch.tensor(x_log, dtype=torch.float32, device=DEVICE)
        return x_original, x_t


# ----------------------------------------------------------------------
# Data loading & preprocessing (with log-manifold + jitter + QuantileTransformer)
# ----------------------------------------------------------------------
def load_and_preprocess_data(num_jets: int) -> Tuple[np.ndarray, Tensor, Tensor, QuantileTransformer, Tensor, Tensor, int]:
    """
    Load Phase 4 & 5 data, filter valid particles, apply log-transform.
    
    Returns:
        x_normalized: normalized tensor [N, 4]
        cond: conditioning tensor [N, 139]
        x_original: original kinematics [N, 4]
        scaler: fitted QuantileTransformer
        threshold_norm: threshold in normalized space for physics loss
        total_particles: count of valid particles
    """
    if not PHASE4_EMBEDDINGS.exists():
        logger.error(f"Phase 4 embeddings not found at {PHASE4_EMBEDDINGS}. Run Phase 4 first.")
        sys.exit(1)
    if not PHASE5_ATTENTION.exists():
        logger.error(f"Phase 5 attention matrices not found at {PHASE5_ATTENTION}. Run Phase 5 first.")
        sys.exit(1)

    try:
        phase4 = torch.load(PHASE4_EMBEDDINGS, map_location="cpu", weights_only=False)
        phase5 = torch.load(PHASE5_ATTENTION, map_location="cpu", weights_only=False)
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        sys.exit(1)

    kinematics = phase4["kinematics"][:num_jets]
    attention = phase5["attention"][:num_jets]

    valid_mask = (kinematics[..., 0] > 1e-5)

    valid_particles = []
    valid_conds = []

    for b in range(kinematics.shape[0]):
        particle_mask = valid_mask[b]
        n_valid = particle_mask.sum().item()
        if n_valid == 0:
            continue
        valid_particles.append(kinematics[b][particle_mask])
        valid_rows = attention[b][particle_mask]
        valid_conds.append(valid_rows)

    x_all = torch.cat(valid_particles, dim=0).float()
    cond_all = torch.cat(valid_conds, dim=0).float()

    logger.info(f"Loaded {num_jets} jets -> {x_all.shape[0]} valid particles")

    x_log = x_all.clone()
    x_log[:, 0] = torch.log(x_all[:, 0] + EPSILON)
    x_log[:, 3] = torch.log(x_all[:, 3] + EPSILON)

    jitter = torch.randn_like(x_log) * 1e-5
    x_jittered = x_log + jitter

    scaler = QuantileTransformer(output_distribution='normal', n_quantiles=N_QUANTILES, random_state=42)
    x_normalized_np = scaler.fit_transform(x_jittered.numpy())
    x_normalized = torch.tensor(x_normalized_np, dtype=torch.float32)

    eta_mean = float(x_all[:, 1].mean())
    phi_mean = float(x_all[:, 2].mean())
    log_eps = math.log(EPSILON)
    ref_log = np.array([log_eps, eta_mean, phi_mean, log_eps], dtype=np.float32)
    threshold_norm = torch.tensor(scaler.transform(ref_log.reshape(1, -1))[0], dtype=torch.float32, device=DEVICE)

    return x_normalized.numpy(), cond_all, x_all, scaler, x_normalized, threshold_norm, x_all.shape[0]


# ----------------------------------------------------------------------
# TRAINING (with physics-informed asymmetric loss & Checkpointing)
# ----------------------------------------------------------------------
def train_model(
    model: nn.Module,
    x: Tensor,
    cond: Tensor,
    threshold_norm: Tensor,
    epochs: int = DEFAULT_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    t_total: int = DIFFUSION_T,
) -> Tuple[nn.Module, float]:
    """
    Train the conditional DDPM using physics-informed asymmetric MSE loss.
    
    Args:
        model: Noise predictor MLP
        x: normalized kinematic tensor
        cond: conditioning tensor
        threshold_norm: physics violation threshold
        epochs: training epochs
        batch_size: batch size
        t_total: diffusion timesteps
        
    Returns:
        Trained model with best weights loaded, best loss
    """
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    alpha, alpha_bar = get_alphas()

    dataset_size = x.shape[0]
    steps_per_epoch = (dataset_size + batch_size - 1) // batch_size
    best_loss = float('inf')

    for epoch in range(epochs):
        epoch_loss = 0.0
        perm = torch.randperm(dataset_size)
        x_shuffled = x[perm]
        cond_shuffled = cond[perm]

        pbar = tqdm(range(0, dataset_size, batch_size), desc=f"Epoch {epoch+1}/{epochs}")
        for start in pbar:
            end = min(start + batch_size, dataset_size)
            x_batch = x_shuffled[start:end].to(DEVICE)
            cond_batch = cond_shuffled[start:end].to(DEVICE)

            t = torch.randint(0, t_total, (x_batch.shape[0],), device=DEVICE)
            x_t, noise = add_noise(x_batch, t, alpha_bar)

            pred_noise = model(x_t, t, cond_batch)

            base_loss = F.mse_loss(pred_noise, noise, reduction='none')

            sqrt_alpha_bar_t = torch.sqrt(alpha_bar[t]).unsqueeze(1)
            sqrt_one_minus_t = torch.sqrt(1.0 - alpha_bar[t]).unsqueeze(1)
            x0_pred = (x_t - sqrt_one_minus_t * pred_noise) / sqrt_alpha_bar_t

            violation = (x0_pred[:, 0] < threshold_norm[0]) | (x0_pred[:, 3] < threshold_norm[3])
            penalty_weights = torch.ones((x_batch.shape[0], 1), device=DEVICE, dtype=torch.float32)
            penalty_weights[violation] = 10.0

            loss = (base_loss * penalty_weights).mean()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.6f}"})

        avg_loss = epoch_loss / steps_per_epoch
        logger.info(f"Epoch {epoch+1}/{epochs} - Avg Loss: {avg_loss:.6f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            try:
                torch.save(model.state_dict(), MODEL_WEIGHTS_PATH)
                logger.info(f"New best loss ({best_loss:.6f}). Weights saved to {MODEL_WEIGHTS_PATH}")
            except Exception as e:
                logger.error(f"Failed to save model weights: {e}")

        scheduler.step()

    try:
        model.load_state_dict(torch.load(MODEL_WEIGHTS_PATH, weights_only=True))
        logger.info("Loaded best model weights for sampling.")
    except Exception as e:
        logger.warning(f"Could not load best weights: {e}")

    return model, best_loss


# ----------------------------------------------------------------------
# Visualization
# ----------------------------------------------------------------------
def plot_money_plot(original: Tensor, generated: Tensor, output_path: Path) -> None:
    """Create 1x2 histogram overlay for p_T and mass distributions."""
    sns.set_style("whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Phase 6: Quantum-Conditioned Generative Validation (QCGV)", fontsize=14, fontweight="bold")

    pT_orig = original[:, 0].cpu().numpy()
    pT_gen = generated[:, 0].cpu().numpy()
    m_orig = original[:, 3].cpu().numpy()
    m_gen = generated[:, 3].cpu().numpy()

    axes[0].hist(pT_orig, bins=80, alpha=0.6, label="Original", color="steelblue", density=True)
    axes[0].hist(pT_gen, bins=80, alpha=0.6, label="Reconstructed", color="coral", density=True)
    axes[0].set_xlabel(r"$p_T$")
    axes[0].set_ylabel("Density")
    axes[0].set_title(r"Transverse Momentum ($p_T$) Distribution")
    axes[0].legend()

    axes[1].hist(m_orig, bins=80, alpha=0.6, label="Original", color="steelblue", density=True)
    axes[1].hist(m_gen, bins=80, alpha=0.6, label="Reconstructed", color="coral", density=True)
    axes[1].set_xlabel(r"Mass ($m$)")
    axes[1].set_ylabel("Density")
    axes[1].set_title(r"Mass ($m$) Distribution")
    axes[1].legend()

    plt.tight_layout()
    try:
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Saved money plot to {output_path}")
    except Exception as e:
        logger.error(f"Failed to save plot: {e}")
        plt.close(fig)


# ----------------------------------------------------------------------
# Reporting
# ----------------------------------------------------------------------
def write_report(
    output_path: Path,
    training_time: float,
    final_loss: float,
    original: Tensor,
    generated: Tensor,
    hparams: Dict[str, int],
    total_particles: int,
) -> None:
    """Write publication-quality report."""
    orig_mean = original.mean(dim=0).tolist() if original.numel() > 0 else []
    orig_std = original.std(dim=0).tolist() if original.numel() > 0 else []
    gen_mean = generated.mean(dim=0).tolist() if generated.numel() > 0 else []
    gen_std = generated.std(dim=0).tolist() if generated.numel() > 0 else []

    report_lines = [
        "=" * 80,
        "PHASE 6: QUANTUM-CONDITIONED GENERATIVE VALIDATION (QCGV) — REPORT",
        "=" * 80,
        "",
        f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"Total Execution Time: {training_time:.2f} seconds",
        "",
        "-" * 80,
        "1. HYPERPARAMETERS",
        "-" * 80,
        f"  Epochs:            {hparams['epochs']}",
        f"  Batch Size:        {hparams['batch_size']}",
        f"  Learning Rate:     {hparams['lr']}",
        f"  Diffusion Steps:   {hparams['t_total']}",
        f"  Generation Samples:  {original.shape[0] if original.numel() > 0 else 0}",
        "",
        "-" * 80,
        "2. TRAINING METRICS",
        "-" * 80,
        f"  Final/Best Loss:   {final_loss:.6f}",
        "",
        "-" * 80,
        "3. KINEMATIC STATISTICS (Original Scale)",
        "-" * 80,
        f"  Comparison Subset - Mean: {orig_mean}",
        f"  Comparison Subset - Std:  {orig_std}",
        f"  Generated - Mean:         {gen_mean}",
        f"  Generated - Std:          {gen_std}",
        "",
        "-" * 80,
        "4. MODEL DETAILS",
        "-" * 80,
        f"  Time Embedding:    {TIME_EMB_DIM}",
        f"  Conditioning Dim:  {COND_DIM}",
        f"  Noise Predictor:   input: 4 + 32 + 16 = 52, hidden: 512x2 + 256 + 4, activation: GELU",
        f"  Positivity:        Log-Manifold diffeomorphism enforced (p_T, m > 0)",
        "",
        "-" * 80,
        "5. DATA STATISTICS",
        "-" * 80,
        f"  Total Valid Particles Loaded: {total_particles}",
        f"  Features: p_T, eta, phi, m",
        "",
        "=" * 80,
        "END OF REPORT",
        "=" * 80,
    ]
    
    try:
        output_path.write_text("\n".join(report_lines), encoding="utf-8")
        logger.info(f"Report written to {output_path}")
    except Exception as e:
        logger.error(f"Failed to write report: {e}")


# ----------------------------------------------------------------------
# Argument parsing
# ----------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 6: Quantum-Conditioned Generative Validation (QCGV)")
    parser.add_argument("--num-jets", type=int, default=DEFAULT_NUM_JETS, help="Number of jets to process")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="Training batch size")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS, help="Number of training epochs")
    parser.add_argument("--output", type=Path, default=OUTPUT_GENERATED, help="Output .pt path")
    parser.add_argument("--generate-samples", type=int, default=0, help="Samples to generate (0=all)")
    return parser.parse_args()


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main() -> None:
    start_time = time.time()

    logger.info("=" * 70)
    logger.info("PHASE 6: QUANTUM-CONDITIONED GENERATIVE VALIDATION (QCGV)")
    diagnose_hardware()
    logger.info("=" * 70)

    args = parse_args()

    x_norm, cond, x_original, scaler, x_norm_tensor, threshold_norm, total_particles = load_and_preprocess_data(args.num_jets)
    x_norm_tensor = x_norm_tensor.to(DEVICE)
    cond = cond.to(DEVICE)

    logger.info("Initializing DDPM model...")
    model = NoisePredictor().to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {n_params:,}")

    logger.info("Training...")
    model, best_loss = train_model(
        model=model,
        x=x_norm_tensor,
        cond=cond,
        threshold_norm=threshold_norm,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )

    logger.info("Generating samples...")
    alpha, alpha_bar = get_alphas()
    n_gen = cond.shape[0] if args.generate_samples == 0 else min(args.generate_samples, cond.shape[0])
    n_vis = min(n_gen, 10000)
    cond_sample = cond[:n_vis]
    x_original_subset = x_original[:n_vis]

    generated, generated_norm = sample_reverse(
        model, cond_sample, alpha, alpha_bar, scaler
    )
    logger.info(f"Generated {generated.shape[0]} particle 4-vectors")

    x_vis_norm = torch.tensor(x_norm[:n_vis], dtype=torch.float32, device=DEVICE)
    orig_mean_norm = x_vis_norm.mean(dim=0)
    orig_std_norm = x_vis_norm.std(dim=0)
    gen_mean_norm = generated_norm.mean(dim=0)
    gen_std_norm = generated_norm.std(dim=0)

    mean_mse = F.mse_loss(gen_mean_norm, orig_mean_norm).item()
    std_mse = F.mse_loss(gen_std_norm, orig_std_norm).item()
    avg_mse = mean_mse + std_mse

    logger.info(f"Statistical MSE - Mean: {mean_mse:.6f}, Std: {std_mse:.6f}, Total: {avg_mse:.6f}")

    hparams = {"epochs": args.epochs, "batch_size": args.batch_size, "lr": 2e-3, "t_total": DIFFUSION_T}

    plot_money_plot(original=x_original_subset, generated=generated.cpu(), output_path=FIGURE_PATH)

    elapsed = time.time() - start_time
    write_report(
        output_path=REPORT_PATH,
        training_time=elapsed,
        final_loss=avg_mse,
        original=x_original_subset.cpu(),
        generated=generated.cpu(),
        hparams=hparams,
        total_particles=total_particles,
    )

    OUTPUT_GENERATED.parent.mkdir(parents=True, exist_ok=True)
    try:
        torch.save({"generated": generated.cpu(), "original_subset": x_original_subset.cpu()}, OUTPUT_GENERATED)
        logger.info(f"Saved tensors to {OUTPUT_GENERATED}")
    except Exception as e:
        logger.error(f"Failed to save tensors: {e}")

    logger.info("=" * 70)
    logger.info("PHASE 6 QCGV COMPLETE")
    logger.info(f"Output: {args.output}")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()