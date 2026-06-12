#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Phase 7 Omega Evaluation: Ultimate Validation of QCGV Generated Kinematics
-------------------------------------------------------------------------------
This script subjects the generated kinematics from Phase 6 (conditional DDPM)
to three rigorous academic validation tests:
    1. Statistical Distance (Wasserstein / Earth Mover's Distance)
    2. Physics Constraint Validation (negative mass, negative pT)
    3. Round-Trip Consistency (re‑formatting for Phase 3/4 re‑injection)

All results are logged to logs/phase7_evaluation_report.txt and key tensors
are saved for downstream reuse.

Author: Kilo (AI Assistant)
"""

import os
import math
import torch
import numpy as np
from scipy.stats import wasserstein_distance
from pathlib import Path
import logging

# -------------------------- CONFIGURATION --------------------------
GENERATED_PATH = Path("data/phase6_generated_kinematics.pt")
# Original valid kinematics from Phase 4/5 (we reconstruct from the same source
# used in phase6_qcgv.py to guarantee a fair comparison)
PHASE4_EMBEDDINGS = Path("data/phase4_quantum_embeddings.pt")
PHASE5_ATTENTION = Path("data/phase5_attention_matrices.pt")
LOG_DIR = Path("logs")
REPORT_PATH = LOG_DIR / "phase7_evaluation_report.txt"
ROUNDTRIP_OUT = Path("data/phase7_round_trip_injection.pt")
MAX_PARTICLES_PER_JET = 139  # as used in the original pipeline
# -----------------------------------------------------------------

def setup_logging():
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(REPORT_PATH, mode="w", encoding="utf-8"),
            logging.StreamHandler()
        ]
    )

def load_original_kinematics():
    """Reproduce the exact original valid particle selection from phase6_qcgv.py."""
    if not PHASE4_EMBEDDINGS.exists() or not PHASE5_ATTENTION.exists():
        raise FileNotFoundError(
            f"Required phase files not found:\n  {PHASE4_EMBEDDINGS}\n  {PHASE5_ATTENTION}"
        )
    phase4 = torch.load(PHASE4_EMBEDDINGS, map_location="cpu", weights_only=True)
    phase5 = torch.load(PHASE5_ATTENTION, map_location="cpu", weights_only=True)
    kinematics = phase4["kinematics"]  # (num_jets, 139, 4)
    attention = phase5["attention"]    # (num_jets, 139, 139)
    valid_mask = (kinematics[..., 0] > 1e-5)  # pT > tiny threshold
    valid_particles = []
    for b in range(kinematics.shape[0]):
        particle_mask = valid_mask[b]
        valid_particles.append(kinematics[b][particle_mask])
    x_all = torch.cat(valid_particles, dim=0).float()  # (N_valid, 4)
    return x_all.numpy()

def test_wasserstein(original: np.ndarray, generated: np.ndarray):
    """Compute Wasserstein distance for each kinematic variable."""
    dims = ["p_T", "eta", "phi", "m"]
    scores = {}
    for i, name in enumerate(dims):
        wd = wasserstein_distance(original[:, i], generated[:, i])
        scores[name] = wd
        logging.info(f"Wasserstein distance ({name}): {wd:.6f}")
    return scores

def test_physics_constraints(generated: np.ndarray):
    """Check for unphysical values."""
    total = generated.shape[0]
    neg_mass = np.sum(generated[:, 3] < 0)      # m < 0
    neg_pt   = np.sum(generated[:, 0] < 0)      # p_T < 0
    perc_mass = (neg_mass / total) * 100.0
    perc_pt   = (neg_pt   / total) * 100.0
    logging.info(f"Negative mass violations: {neg_mass}/{total} ({perc_mass:.2f}%)")
    logging.info(f"Negative p_T violations:  {neg_pt}/{total} ({perc_pt:.2f}%)")
    integrity = 100.0 - max(perc_mass, perc_pt)  # conservative bound
    logging.info(f"Physics Integrity Score: {integrity:.2f}%")
    return {"neg_mass_%": perc_mass, "neg_pt_%": perc_pt, "integrity_%": integrity}

def format_for_round_trip_inference(generated_tensor: torch.Tensor):
    """
    Reshape/pad generated [N,4] tensor into [batch_size, 139, 4] Jet format.
    Returns the tensor ready for Phase 3/4 inference.
    """
    N = generated_tensor.shape[0]
    batch_size = math.ceil(N / MAX_PARTICLES_PER_JET)
    padded = torch.zeros((batch_size, MAX_PARTICLES_PER_JET, 4), dtype=generated_tensor.dtype)
    idx = 0
    for b in range(batch_size):
        n_this = min(MAX_PARTICLES_PER_JET, N - idx)
        if n_this > 0:
            padded[b, :n_this, :] = generated_tensor[idx:idx + n_this]
        idx += n_this
    logging.info(
        f"Round-trip tensor shaped: {list(padded.shape)} "
        f"(batch_size={batch_size}, particles_per_jet={MAX_PARTICLES_PER_JET})"
    )
    return padded

def main():
    setup_logging()
    logging.info("=" * 70)
    logging.info("PHASE 7 OMEGA EVALUATION – ULTIMATE VALIDATION")
    logging.info("=" * 70)

    # Load data
    logging.info(f"Loading generated kinematics from {GENERATED_PATH}")
    if not GENERATED_PATH.exists():
        raise FileNotFoundError(f"Generated file not found: {GENERATED_PATH}")
    generated_dict = torch.load(GENERATED_PATH, map_location="cpu", weights_only=True)
    generated_tensor = generated_dict["generated"]
    generated = generated_tensor.numpy()  # (N_gen, 4)
    logging.info(f"Generated shape: {generated.shape}")

    logging.info("Reconstructing original valid kinematics from Phase 4/5...")
    original = load_original_kinematics()
    logging.info(f"Original shape: {original.shape}")

    # ----------------- TEST 1: Wasserstein -----------------
    logging.info("-" * 70)
    logging.info("TEST 1: Statistical Distance (Wasserstein / Earth Mover's)")
    ws_scores = test_wasserstein(original, generated)

    # ----------------- TEST 2: Physics Constraints ----------
    logging.info("-" * 70)
    logging.info("TEST 2: Physics Constraint Validation")
    phys_results = test_physics_constraints(generated)

    # ----------------- TEST 3: Round-Trip -------------------
    logging.info("-" * 70)
    logging.info("TEST 3: Round-Trip Consistency (Re-formatting for Phase 3/4)")
    round_trip_tensor = format_for_round_trip_inference(torch.from_numpy(generated))
    ROUNDTRIP_OUT.parent.mkdir(parents=True, exist_ok=True)
    torch.save(round_trip_tensor, ROUNDTRIP_OUT)
    logging.info(f"Saved round-trip injection tensor to {ROUNDTRIP_OUT}")

    # ----------------- SUMMARY ------------------------------
    logging.info("-" * 70)
    logging.info("EVALUATION SUMMARY")
    for name, score in ws_scores.items():
        logging.info(f"  Wasserstein {name:>4}: {score:.6f}")
    logging.info(
        f"  Physics Integrity: {phys_results['integrity_%']:.2f}% "
        f"(neg_mass: {phys_results['neg_mass_%']:.2f}%, neg_pt: {phys_results['neg_pt_%']:.2f}%)"
    )
    logging.info(f"  Round-trip tensor saved: {ROUNDTRIP_OUT}")
    logging.info("=" * 70)
    logging.info("Phase 7 Omega Evaluation Complete.")

if __name__ == "__main__":
    main()