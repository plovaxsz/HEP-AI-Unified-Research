"""Phase 5: Quantum Pairformer Attention (QPA) Layer.

Architecture:
    - Input: kinematics [p_T, eta, phi, m] and Phase 4 LEQE embeddings
    - Classical preprocessing: pairwise ΔR and invariant mass m_ij for all particle pairs in a jet
    - Quantum circuit: 2-qubit entanglement workspace (lightning.qubit) with tanh-normalized inputs
    - Measurement: PauliZ(0) @ PauliZ(1) expectation -> Entanglement Score in [-1, 1]
    - Output: Quantum Attention Matrix (batch_size, num_particles, num_particles)

Hardware: RTX 3060ti (8GB VRAM) safe via chunked pair batching.
Framework: PennyLane 0.34.0 + PyTorch 2.x. No Qiskit.
"""

from __future__ import annotations

import argparse
import logging
import math
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pennylane as qml
import seaborn as sns
import torch
import torch.nn as nn
from torch import Tensor

# --- CONFIG ---
BASE_DIR = Path(__file__).resolve().parent
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUTPUT_DIR = BASE_DIR / "models"
LOG_DIR = BASE_DIR / "logs"
LOG_PATH = LOG_DIR / "phase5_execution.log"
REPORT_PATH = OUTPUT_DIR / "phase5_qpa_report.txt"
FIGURE_PATH = OUTPUT_DIR / "figure_phase5_qpa_analysis.png"
PHASE4_EMBEDDINGS = BASE_DIR / "data" / "phase4_quantum_embeddings.pt"
OUTPUT_ATTENTION = BASE_DIR / "data" / "phase5_attention_matrices.pt"

DEFAULT_NUM_JETS = 5000
DEFAULT_BATCH_SIZE = 64
PAIR_BATCH_SIZE = 8192

N_QUBITS = 2
N_LAYERS = 2

# --- LOGGING SETUP ---
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


# ============================================================================
# QUANTUM PAIRFORMER ATTENTION CIRCUIT (FIXED: tanh normalization + PauliZZ)
# ============================================================================

def build_qpa_circuit(n_qubits: int = N_QUBITS):
    wires = list(range(n_qubits))

    def circuit(inputs: Tensor, weights: Tensor) -> Tensor:
        # inputs: [deltaR, log_mass] for the particle pair, shape (batch, 2)
        # weights: shape (n_layers, n_qubits, 3)
        deltaR = inputs[:, 0]
        log_m = inputs[:, 1]

        # --- FIX A: Kinematic Normalization (tanh squashing to [0, pi]) ---
        norm_delta_R = torch.tanh(deltaR / 5.0) * math.pi
        norm_mass = torch.tanh(log_m / 50.0) * math.pi

        # --- FIX B: Data Encoding Layer with normalized features ---
        qml.RY(norm_delta_R, wires=wires[0])
        qml.RZ(norm_delta_R, wires=wires[1])
        qml.CRY(norm_mass, wires=[wires[0], wires[1]])

        # --- Trainable Entanglement Layer ---
        for layer in range(weights.shape[0]):
            for q in range(n_qubits):
                qml.Rot(weights[layer, q, 0], weights[layer, q, 1], weights[layer, q, 2], wires=wires[q])
            if layer < weights.shape[0] - 1:
                qml.CNOT(wires=[wires[0], wires[1]])

        # --- FIX B: Measurement: PauliZ(0) @ PauliZ(1) -> range [-1, 1] ---
        return qml.expval(qml.PauliZ(wires[0]) @ qml.PauliZ(wires[1]))

    qnode = qml.QNode(circuit, qml.device("lightning.qubit", wires=n_qubits), interface="torch")
    return qnode


class QPA_Layer(nn.Module):
    """Quantum Pairformer Attention Layer.

    Maps a batch of particle-pair features into quantum attention scores.
    """

    def __init__(self, n_qubits: int = N_QUBITS, n_layers: int = N_LAYERS):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        qnode = build_qpa_circuit(n_qubits=n_qubits)
        self.qlayer = qml.qnn.TorchLayer(
            qnode,
            weight_shapes={"weights": (n_layers, n_qubits, 3)},
        )
        # Post-net now maps single scalar output to 1
        self.post_net = nn.Sequential(
            nn.Linear(1, 1),
        )
        logger.info(f"QPA_Layer initialized: n_qubits={n_qubits}, n_layers={n_layers}")

    def forward(self, pair_features: Tensor) -> Tensor:
        # pair_features: (N_pairs, 2) where 2 = [deltaR, log_mass]
        if pair_features.dim() == 1:
            pair_features = pair_features.unsqueeze(0)
        q_out = self.qlayer(pair_features)  # (N_pairs,)
        q_out = q_out.unsqueeze(-1)  # (N_pairs, 1)
        out = self.post_net(q_out)  # (N_pairs, 1)
        return out.squeeze(-1)  # (N_pairs,)


# ============================================================================
# PAIRWISE FEATURE COMPUTATION (OOM-safe chunking)
# ============================================================================

def compute_pairwise_features(kinematics: Tensor) -> Tensor:
    """Compute classical pairwise ΔR and invariant mass for all particles in a jet.

    Args:
        kinematics: (batch_size, num_particles, 4) with [p_T, eta, phi, m].

    Returns:
        pair_features: (batch_size, num_particles, num_particles, 2) with [deltaR, log_mass].
    """
    kinematics = kinematics.float()
    pt = kinematics[..., 0]
    eta = kinematics[..., 1]
    phi = kinematics[..., 2]
    mass = kinematics[..., 3]

    delta_eta = eta.unsqueeze(2) - eta.unsqueeze(1)  # (B, N, N)
    delta_phi = phi.unsqueeze(2) - phi.unsqueeze(1)  # (B, N, N)
    deltaR = torch.sqrt(delta_eta ** 2 + delta_phi ** 2)

    px = pt.unsqueeze(2) * torch.cos(phi.unsqueeze(2))
    py = pt.unsqueeze(2) * torch.sin(phi.unsqueeze(2))
    pz = pt.unsqueeze(2) * torch.sinh(eta.unsqueeze(2))
    e = torch.sqrt(pt.unsqueeze(2) ** 2 + pz ** 2 + mass.unsqueeze(2) ** 2)

    px_i = px
    py_i = py
    pz_i = pz
    e_i = e

    px_j = px.transpose(1, 2)
    py_j = py.transpose(1, 2)
    pz_j = pz.transpose(1, 2)
    e_j = e.transpose(1, 2)

    m_ij_sq = (e_i + e_j) ** 2 - (px_i + px_j) ** 2 - (py_i + py_j) ** 2 - (pz_i + pz_j) ** 2
    m_ij_sq = torch.clamp(m_ij_sq, min=0.0)
    m_ij = torch.sqrt(m_ij_sq)

    log_mass = torch.log(m_ij + 1e-6)

    pair_features = torch.stack([deltaR, log_mass], dim=-1)  # (B, N, N, 2)
    return pair_features


# ============================================================================
# QUANTUM ATTENTION MATRIX GENERATION (OOM-safe)
# ============================================================================

def compute_attention_matrix_batched(
    qpa_layer: QPA_Layer,
    kinematics: Tensor,
    pair_batch_size: int = PAIR_BATCH_SIZE,
) -> Tensor:
    """Batched quantum attention matrix computation (OOM-safe) with Zero-Padding Masking.

    Processes multiple jets per forward chunk while skipping zero-padded particles.
    Valid particles have p_T > 1e-5. Padded pairs default to -1.0.
    """
    qpa_layer.eval()
    pair_features = compute_pairwise_features(kinematics)  # (B, N, N, 2)
    batch_size, num_particles = kinematics.shape[:2]

    # --- Zero-Padding Mask: particles with p_T > 1e-5 are valid ---
    valid_mask = (kinematics[..., 0] > 1e-5)  # (B, N)

    # --- Initialize with -1.0 (background/no-interaction score) ---
    attention_matrix = torch.full((batch_size, num_particles, num_particles), -1.0, device="cpu")

    total_jets = batch_size
    done = 0
    while done < total_jets:
        batch_end = min(done + DEFAULT_BATCH_SIZE, total_jets)
        jet_batch_feats = pair_features[done:batch_end]  # (b, N, N, 2)
        jet_valid_mask = valid_mask[done:batch_end]     # (b, N)
        b = jet_batch_feats.shape[0]

        # --- Build valid pair mask (i != j, both particles valid) ---
        row_mask = jet_valid_mask.unsqueeze(2)  # (b, N, 1)
        col_mask = jet_valid_mask.unsqueeze(1)  # (b, 1, N)
        pair_mask = row_mask & col_mask  # (b, N, N)

        # --- Exclude diagonal (will be handled separately) ---
        diag_idx = torch.arange(num_particles, device="cpu")
        pair_mask[:, diag_idx, diag_idx] = False

        # --- Flatten and get valid pair indices ---
        flat_mask = pair_mask.reshape(-1)  # (b*N*N,)
        flat_feats = jet_batch_feats.reshape(-1, 2)  # (b*N*N, 2)

        valid_indices = flat_mask.nonzero(as_tuple=True)[0]
        valid_features = flat_feats[valid_indices]

        if valid_features.shape[0] > 0:
            # Compute quantum scores for valid pairs only
            pair_tensor = valid_features.to(DEVICE)
            scores = []
            for start in range(0, pair_tensor.shape[0], pair_batch_size):
                end = min(start + pair_batch_size, pair_tensor.shape[0])
                chunk = pair_tensor[start:end]
                with torch.no_grad():
                    score = qpa_layer(chunk).cpu()
                scores.append(score)
            flat_scores = torch.cat(scores, dim=0)

            # --- Reconstruct scores to matrix ---
            reconstructed = torch.zeros(b * num_particles * num_particles, device="cpu")
            reconstructed[valid_indices] = flat_scores.cpu()
            attention_matrix[done:batch_end] = reconstructed.reshape(b, num_particles, num_particles)

        # --- Diagonal self-attention (identity) for valid particles ---
        for bi in range(b):
            diag_indices = diag_idx[jet_valid_mask[bi]]
            attention_matrix[done + bi, diag_indices, diag_indices] = 1.0

        done = batch_end
        vram_alloc = torch.cuda.memory_allocated() / (1024 ** 2) if DEVICE.type == "cuda" else 0.0
        logger.info(
            "  Batch %d/%d processed | jets %d-%d / %d | valid pairs=%d | VRAM: %.1f MB",
            (done // DEFAULT_BATCH_SIZE) + 1,
            (total_jets + DEFAULT_BATCH_SIZE - 1) // DEFAULT_BATCH_SIZE,
            done - jet_batch_feats.shape[0] + 1,
            done,
            total_jets,
            valid_indices.shape[0],
            vram_alloc,
        )
    return attention_matrix


# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_qpa_analysis(attention_matrices: Tensor, output_path: Path) -> None:
    """1x2 dashboard: heatmap of first jet + score distribution from full dataset."""
    sns.set_style("whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Phase 5: Quantum Pairformer Attention (QPA)", fontsize=14, fontweight="bold")

    mat_np = attention_matrices.detach().cpu().numpy()
    jet0 = mat_np[0]
    im = axes[0].imshow(jet0, cmap="viridis", aspect="auto", vmin=-1, vmax=1)
    axes[0].set_title("Quantum Attention Matrix (Jet 0)")
    axes[0].set_xlabel("Particle j")
    axes[0].set_ylabel("Particle i")
    fig.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04)

    flat_scores = mat_np.ravel()
    score_mean = float(flat_scores.mean())
    score_std = float(flat_scores.std())
    axes[1].hist(flat_scores, bins=80, color="steelblue", edgecolor="black", alpha=0.7, density=True)
    axes[1].axvline(score_mean, color="red", linestyle="--", linewidth=2, label=f"Mean={score_mean:.4f}")
    axes[1].axvline(score_mean + score_std, color="orange", linestyle=":", linewidth=1.5, label=f"±1σ={score_std:.4f}")
    axes[1].axvline(score_mean - score_std, color="orange", linestyle=":", linewidth=1.5)
    axes[1].set_xlabel("Entanglement Score")
    axes[1].set_ylabel("Density")
    axes[1].set_title("Distribution of Entanglement Scores (Full Dataset)")
    axes[1].legend()

    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    pdf_path = output_path.with_suffix(".pdf")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved QPA analysis plot to {output_path}")


# ============================================================================
# REPORTING
# ============================================================================

def write_qpa_report(
    output_path: Path,
    execution_time: float,
    attention_shape: tuple[int, ...],
    avg_score: float,
    score_std: float,
    score_min: float,
    score_max: float,
    n_params: int,
) -> None:
    score_variance = score_std ** 2
    no_collapse = abs(avg_score) < 0.95 and score_std > 0.05
    healthy_range = -0.99 <= score_min <= 1.0 and -0.99 <= score_max <= 1.0

    report_lines = [
        "=" * 80,
        "PHASE 5: QUANTUM PAIRFORMER ATTENTION (QPA) — PRODUCTION REPORT",
        "=" * 80,
        "",
        f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"Total Execution Time: {execution_time:.2f} seconds",
        "",
        "-" * 80,
        "1. SHAPE DIMENSIONS",
        "-" * 80,
        f"  Attention Matrix Shape:      {attention_shape}",
        "",
        "-" * 80,
        "2. ENTANGLEMENT SCORE STATISTICS",
        "-" * 80,
        f"  Average Entanglement Score:  {avg_score:.6f}",
        f"  Std Dev of Scores:           {score_std:.6f}",
        f"  Variance:                    {score_variance:.6f}",
        f"  Min Score:                   {score_min:.6f}",
        f"  Max Score:                   {score_max:.6f}",
        "",
        "-" * 80,
        "3. VALIDATION FLAGS",
        "-" * 80,
        f"  No Representation Collapse: {'PASS' if no_collapse else 'FAIL'}",
        f"  Scores In Bounds [-1,1]:    {'PASS' if healthy_range else 'FAIL'}",
        "",
        "-" * 80,
        "4. MODEL DETAILS",
        "-" * 80,
        f"  Trainable Parameters:        {n_params:,}",
        f"  Qubits:                      {N_QUBITS}",
        f"  Layers:                      {N_LAYERS}",
        "",
        "-" * 80,
        "5. OOM-SAFETY NOTES",
        "-" * 80,
        "  Pairwise matrix computed in chunks (PAIR_BATCH_SIZE).",
        "  Jets processed sequentially to bound memory.",
        "",
        "-" * 80,
        "6. NORMALIZATION FIX APPLIED",
        "-" * 80,
        "  deltaR normalized with tanh(deltaR/5.0)*pi",
        "  log_mass normalized with tanh(log_m/50.0)*pi",
        "  Measurement: PauliZ(0) @ PauliZ(1) -> [-1, 1]",
        "  Diagonal enforced to 1.0 (self-attention).",
        "",
        "=" * 80,
        "END OF REPORT",
        "=" * 80,
        "",
    ]
    output_path.write_text("\n".join(report_lines), encoding="utf-8")
    logger.info(f"Report written to {output_path}")


# ============================================================================
# DATA LOADING
# ============================================================================

def load_phase4_data(num_jets: int) -> tuple[Tensor, Tensor]:
    if not PHASE4_EMBEDDINGS.exists():
        raise FileNotFoundError(
            f"Phase 4 embeddings not found at {PHASE4_EMBEDDINGS}. Run Phase 4 first."
        )
    ckpt = torch.load(PHASE4_EMBEDDINGS, map_location="cpu", weights_only=False)
    kinematics = ckpt["kinematics"][:num_jets]
    embeddings = ckpt["embeddings"][:num_jets]
    labels = ckpt["labels"][:num_jets]
    logger.info(f"Loaded Phase 4 data: kinematics={kinematics.shape}, embeddings={embeddings.shape}, labels={labels.shape}")
    return kinematics, embeddings, labels


# ============================================================================
# ARGS & MAIN
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Phase 5: Quantum Pairformer Attention (QPA)")
    parser.add_argument("--num-jets", type=int, default=DEFAULT_NUM_JETS, help="Number of jets to process")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="Jet batch size for attention computation")
    parser.add_argument("--pair-batch-size", type=int, default=PAIR_BATCH_SIZE, help="Max pairwise chunks per step")
    parser.add_argument("--output", type=Path, default=OUTPUT_ATTENTION, help="Output .pt for attention matrices")
    parser.add_argument("--save-inputs", action="store_true", help="Also save kinematics + embeddings alongside attention")
    return parser.parse_args()


def main() -> None:
    start_time = time.time()
    logger.info("=" * 70)
    logger.info("PHASE 5: QUANTUM PAIRFORMER ATTENTION (QPA)")
    logger.info("=" * 70)

    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    kinematics, embeddings, labels = load_phase4_data(args.num_jets)

    logger.info("Initializing QPA Layer...")
    qpa_layer = QPA_Layer(n_qubits=N_QUBITS, n_layers=N_LAYERS).to(DEVICE)
    n_params = sum(p.numel() for p in qpa_layer.parameters())
    logger.info(f"QPA parameters: {n_params:,}")

    logger.info("Computing Quantum Attention Matrices...")
    attention_matrices = compute_attention_matrix_batched(
        qpa_layer=qpa_layer,
        kinematics=kinematics,
        pair_batch_size=args.pair_batch_size,
    )
    logger.info(f"Attention matrix shape: {tuple(attention_matrices.shape)}")
    avg_score = float(attention_matrices.mean().item())
    score_std = float(attention_matrices.std().item())
    score_min = float(attention_matrices.min().item())
    score_max = float(attention_matrices.max().item())
    logger.info(f"Average Entanglement Score: {avg_score:.6f} ± {score_std:.6f}")
    logger.info(f"Score Range: [{score_min:.6f}, {score_max:.6f}]")

    save_dict = {"attention": attention_matrices, "shape": tuple(attention_matrices.shape)}
    if args.save_inputs:
        save_dict["kinematics"] = kinematics
        save_dict["embeddings"] = embeddings
        save_dict["labels"] = labels
    torch.save(save_dict, args.output)
    logger.info(f"Attention matrices saved to {args.output}")

    plot_qpa_analysis(attention_matrices=attention_matrices, output_path=FIGURE_PATH)

    elapsed = time.time() - start_time
    write_qpa_report(
        output_path=REPORT_PATH,
        execution_time=elapsed,
        attention_shape=tuple(attention_matrices.shape),
        avg_score=avg_score,
        score_std=score_std,
        score_min=score_min,
        score_max=score_max,
        n_params=n_params,
    )

    logger.info("=" * 70)
    logger.info("PHASE 5 QPA COMPLETE")
    logger.info(f"Output: {args.output}")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
