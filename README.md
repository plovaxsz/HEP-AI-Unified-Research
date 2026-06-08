# A Unified Topological and Generative AI Framework for High-Energy Particle Physics

**HEP-AI-Unified-Research** is a research-grade software framework implementing Iterative Progressive Refinement (IPR) training for quark/gluon discrimination using Topological Data Analysis (TDA) and Graph Attention Networks (GATv2).

## Project Roadmap

This research project follows a structured 6-phase development plan:

1. **Phase 1: Fundamental Discrimination** (Current) — TDA-GATv2 baseline training on CERN Pythia jet data. 5,000 epochs of IPR protocol completed.
2. **Phase 2: Model Interpretability** — Integration of XAI techniques (Captum) for attention visualization and particle-level feature attribution.
3. **Phase 3: Generative Augmentation** — Diffusion/GAN-based jet augmentation to improve generalization on rare event topologies.
4. **Phase 4: Multi-Modal Fusion** — Combining TDA graphs with calorimeter image representations (CNNs) for hybrid inference.
5. **Phase 5: Transfer Learning** — Domain adaptation across different MC generators (Pythia → Herwig) and detector geometries (CMS → ATLAS).
6. **Phase 6: Quantum Integration** — Exploration of quantum graph neural networks (QGNN) and quantum kernel methods for topology-aware classification.

## Current Status

**Phase 1: Fundamental Discrimination — COMPLETE**

- **Model:** TDA-GATv2 (Graph Attention Network v2 with Topological Data Analysis features)
- **Dataset:** 50,000 real CERN Pythia quark/gluon jets (canonical shards)
- **Training Protocol:** Iterative Progressive Refinement (IPR) — 50 cycles × 100 epochs = 5,000 total epochs
- **Best Validation AUC:** 0.8629
- **Checkpoints:** `models/checkpoint_cycle_{1..50}.pt` + `best_model_ever.pt`
- **Telemetry:** `models/evolution.log` (CSV with per-epoch metrics)
- **Visualization:** `visualize_training.py` (Matplotlib/Seaborn scientific plots)

### Key Features
- Strict real-data enforcement (no synthetic fallback in production pipeline)
- Fault-tolerant architecture with auto-checkpoint every 5 cycles
- Graceful shutdown handling (SIGINT/SIGTERM safe)
- Mixed-precision training (FP16) on NVIDIA RTX 3060 Ti
- CPU-bound Ripser topology preprocessing with persistent caching

## Repository Structure

```
HEP-AI-Unified-Research/
├── TDA_GATv2_Research/
│   ├── data/                          # Shards and processed cache (gitignored)
│   ├── models/                        # Checkpoints and logs (gitignored)
│   ├── train.py                       # Main training entrypoint
│   ├── train_robust.py                # IPR launcher with buffered I/O
│   ├── train_ipr.py                   # Legacy IPR launcher
│   ├── data_pipeline.py               # Canonical jet shard loading & TDA preprocessing
│   ├── model.py                       # TDA-GATv2 architecture
│   ├── telemetry.py                   # WebSocket telemetry server
│   └── visualize_training.py          # Scientific visualization pipeline
├── requirements.txt                   # Python dependencies
├── .gitignore                         # Excludes large binaries, logs, venv
└── README.md                          # This file
```

## Installation

```bash
git clone https://github.com/plovaxsz/HEP-AI-Unified-Research.git
cd HEP-AI-Unified-Research
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# or
.venv\Scripts\activate     # Windows
pip install -r requirements.txt
```

## Quick Start

```bash
# Run 100-epoch training on real CERN data
python TDA_GATv2_Research/train.py \
  --epochs 100 \
  --batch-size 128 \
  --num-data 50000 \
  --shard-dir TDA_GATv2_Research/data/canonical_shards_real \
  --cache-dir TDA_GATv2_Research/data/processed_cache_real \
  --resume TDA_GATv2_Research/models/gatv2_final.pth

# Launch 50-cycle IPR sprint
python TDA_GATv2_Research/train_robust.py

# Visualize training metrics
python TDA_GATv2_Research/visualize_training.py
```

## Citation

If this framework contributes to your research, please cite:

```bibtex
@misc{hep-ai-unified-2026,
  title={A Unified Topological and Generative AI Framework for High-Energy Particle Physics},
  author={Plovaxsz},
  year={2026},
  note={Phase 1: TDA-GATv2 Fundamental Discrimination — 5,000 Epochs IPR Training}
}
```

## License

MIT License — see [LICENSE](LICENSE) for details.
