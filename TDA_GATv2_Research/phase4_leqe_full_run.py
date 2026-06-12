"""Phase 4 Full-Scale Runner: Quantum Embedding Generation for Real Jets.

Pipeline:
    1. Load 5,000 real jets from canonical_shards_real
    2. Extract raw kinematics (pT, eta, phi, m) per particle
    3. Process through LEQE_Layer with OOM-safe chunking (RTX 3060ti optimized)
    4. Save quantum embeddings to phase4_quantum_embeddings.pt
    5. Log progress to console + file
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import torch

from phase4_leqe import LEQE_Layer, embed_jet_dataset, LOG_DIR, OUTPUT_DIR, N_QUBITS
import phase4_leqe as phase4_module

# --- LOGGING ---
LOG_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter(
    "%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
file_handler = UnbufferedFileHandler(LOG_PATH, mode="w")
file_handler.setFormatter(formatter)
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logger.addHandler(file_handler)
logger.addHandler(stream_handler)

# Force phase4_leqe module logger to propagate to our file handler
phase4_logger = logging.getLogger("phase4_leqe")
phase4_logger.setLevel(logging.INFO)
for h in logger.handlers:
    phase4_logger.addHandler(h)
logger.setLevel(logging.INFO)
formatter = logging.Formatter(
    "%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
file_handler = UnbufferedFileHandler(LOG_PATH, mode="w")
file_handler.setFormatter(formatter)
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logger.addHandler(file_handler)
logger.addHandler(stream_handler)


def load_real_jets(max_samples: int = N_SAMPLES) -> tuple[torch.Tensor, torch.Tensor]:
    """Load real jet kinematics from canonical shards.
    
    Computes invariant mass from [pT, eta, phi, E, PID] -> [pT, eta, phi, m].
    
    Returns:
        padded: Tensor of shape (n_jets, max_particles, 4) with [pT, eta, phi, m].
        mask:   Bool tensor of shape (n_jets, max_particles), True for real particles.
    """
    from data_pipeline import CanonicalJetTDADataset
    
    logger.info(f"Loading real jets from {REAL_DATA_DIR}...")
    dataset = CanonicalJetTDADataset(
        shard_dir=REAL_DATA_DIR,
        cache_dir=CACHE_DIR,
        allow_synthetic=False,
    )
    
    n = min(max_samples, len(dataset))
    raw_samples = []
    for i in range(n):
        data = dataset[i]
        pT = data.x[:, 0]
        eta = data.x[:, 1]
        phi = data.x[:, 2]
        E = data.x[:, 3]
        px = pT * torch.cos(phi)
        py = pT * torch.sin(phi)
        pz = pT * torch.sinh(eta)
        m_sq = E ** 2 - (px ** 2 + py ** 2 + pz ** 2)
        m = torch.sqrt(torch.clamp(m_sq, min=0.0))
        kinematics = torch.stack([pT, eta, phi, m], dim=-1)
        raw_samples.append(kinematics)
    
    max_particles = max(s.shape[0] for s in raw_samples)
    padded = torch.zeros(n, max_particles, 4, dtype=torch.float32)
    mask = torch.zeros(n, max_particles, dtype=torch.bool)
    
    for i, s in enumerate(raw_samples):
        padded[i, : s.shape[0]] = s
        mask[i, : s.shape[0]] = True
    
    logger.info(f"Loaded {n} jets | max_particles={max_particles} | padded shape={padded.shape}")
    return padded, mask


def main() -> None:
    start_time = time.time()
    logger.info("=" * 70)
    logger.info("PHASE 4 FULL RUN: QUANTUM EMBEDDING GENERATION")
    logger.info("=" * 70)
    logger.info(f"Device: {DEVICE}")
    logger.info(f"Target samples: {N_SAMPLES}")
    logger.info(f"Batch size: {BATCH_SIZE}")

    # --- Load Data ---
    jet_features, mask = load_real_jets(max_samples=N_SAMPLES)
    n_jets, n_particles, n_features = jet_features.shape
    logger.info(f"Jet features shape: {jet_features.shape}")

    # --- Initialize LEQE Layer ---
    logger.info("Initializing LEQE_Layer...")
    leqe = LEQE_Layer(
        n_qubits=N_QUBITS,
        n_layers=2,
        embedding="angle",
        device="lightning.qubit",
        output_dim=8,
    ).to(DEVICE)
    
    n_params = sum(p.numel() for p in leqe.parameters())
    logger.info(f"LEQE parameters: {n_params:,}")

    # --- Generate Embeddings ---
    logger.info("Starting quantum embedding generation...")
    embeddings = embed_jet_dataset(
        leqe_layer=leqe,
        jet_features=jet_features,
        batch_size=BATCH_SIZE,
    )
    
    logger.info(f"Embeddings shape: {embeddings.shape}")
    logger.info(f"Embeddings mean: {embeddings.mean().item():.6f}")
    logger.info(f"Embeddings std: {embeddings.std().item():.6f}")

    # --- Save to Disk ---
    logger.info(f"Saving embeddings to {OUTPUT_PATH}...")
    torch.save({
        "embeddings": embeddings.cpu(),
        "jet_features": jet_features.cpu(),
        "n_qubits": N_QUBITS,
        "n_layers": 2,
        "output_dim": 8,
        "n_jets": n_jets,
        "n_particles": n_particles,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }, OUTPUT_PATH)
    
    elapsed = time.time() - start_time
    logger.info(f"Saved {n_jets} jet embeddings in {elapsed:.2f}s")
    logger.info(f"Output size: {OUTPUT_PATH.stat().st_size / 1024 / 1024:.2f} MB")
    
    logger.info("=" * 70)
    logger.info("PHASE 4 FULL RUN COMPLETE")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
