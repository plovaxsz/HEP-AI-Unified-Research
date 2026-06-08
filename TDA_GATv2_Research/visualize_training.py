"""Publication-quality visualization for TDA-GATv2 training convergence.

Reads evolution.log, extracts cycle-level summaries, and exports
both PNG (high-DPI) and PDF (vector) figures for research papers.
"""

from __future__ import annotations

import pathlib

import matplotlib.pyplot as plt
import pandas as pd

LOG_PATH = pathlib.Path(r"D:\PISS\TDA_GATv2_Research\models\evolution.log")
OUTPUT_DIR = pathlib.Path(r"D:\PISS\TDA_GATv2_Research\models")

plt.style.use("seaborn-v0_8-darkgrid")


def load_cycle_summaries(log_path: pathlib.Path) -> pd.DataFrame:
    """Load evolution.log and return only the cycle-level summary rows.

    The log contains two row types:
      1. Epoch-level: starts with 'RUN_<timestamp>'
      2. Cycle-level: starts with a digit (cycle number)

    We only want the cycle-level rows for the 3-subplot figure.
    """

    rows: list[dict[str, float | int | str]] = []
    with open(log_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("run_id,"):
                continue
            parts = line.split(",")
            if len(parts) < 8:
                continue
            try:
                int(parts[0])  # cycle number — first column is an int
                rows.append(
                    {
                        "cycle": int(parts[0]),
                        "val_auc": float(parts[1]),
                        "train_loss": float(parts[2]),
                        "train_accuracy": float(parts[3]),
                        "val_accuracy": float(parts[4]),
                        "learning_rate": float(parts[5]),
                        "epochs_completed": int(parts[6]),
                        "status": parts[7],
                    }
                )
            except ValueError:
                continue  # skip epoch-level rows (start with RUN_...)

    df = pd.DataFrame(rows).sort_values("cycle").reset_index(drop=True)
    return df


def main() -> None:
    if not LOG_PATH.exists():
        raise FileNotFoundError(f"Log file not found: {LOG_PATH}")

    df = load_cycle_summaries(LOG_PATH)
    if df.empty:
        raise ValueError("No cycle-level summary rows found in evolution.log")

    cycles = df["cycle"].values
    final_auc = float(df["val_auc"].iloc[-1])
    best_auc = float(df["val_auc"].max())
    best_cycle = int(df.loc[df["val_auc"].idxmax(), "cycle"])

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(
        "TDA-GATv2 Training Convergence — 50-Cycle IPR Protocol (5,000 Epochs)",
        fontsize=14,
        fontweight="bold",
    )

    # --- Subplot 1: Train Loss ---
    ax1 = axes[0]
    ax1.plot(cycles, df["train_loss"].values, color="crimson", linewidth=2.0, marker="o", markersize=4)
    ax1.set_xlabel("Cycle (×100 epochs)", fontsize=11)
    ax1.set_ylabel("Train Loss", fontsize=11)
    ax1.set_title("Training Loss per Cycle", fontsize=12, fontweight="bold")
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0.18, 0.25)

    # --- Subplot 2: Accuracy (train vs val) ---
    ax2 = axes[1]
    ax2.plot(cycles, df["train_accuracy"].values, color="green", linewidth=2.0, marker="s", markersize=4, label="Train Accuracy")
    ax2.plot(cycles, df["val_accuracy"].values, color="dodgerblue", linewidth=2.0, marker="^", markersize=4, label="Val Accuracy")
    ax2.set_xlabel("Cycle (×100 epochs)", fontsize=11)
    ax2.set_ylabel("Accuracy", fontsize=11)
    ax2.set_title("Train vs Validation Accuracy", fontsize=12, fontweight="bold")
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0.75, 0.93)

    # --- Subplot 3: Validation AUC ---
    ax3 = axes[2]
    ax3.plot(cycles, df["val_auc"].values, color="navy", linewidth=2.0, marker="D", markersize=4)
    ax3.axhline(y=best_auc, color="gold", linestyle="--", linewidth=1.5, alpha=0.8, label=f"Best AUC = {best_auc:.4f}")
    ax3.axvline(x=best_cycle, color="red", linestyle="--", linewidth=1.5, alpha=0.8, label=f"Best @ Cycle {best_cycle}")
    ax3.set_xlabel("Cycle (×100 epochs)", fontsize=11)
    ax3.set_ylabel("Validation AUC", fontsize=11)
    ax3.set_title("Validation AUC Convergence", fontsize=12, fontweight="bold")
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(0.84, 0.87)

    plt.tight_layout()

    png_path = OUTPUT_DIR / "training_convergence.png"
    pdf_path = OUTPUT_DIR / "training_convergence.pdf"
    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.close()

    print("=" * 60)
    print("TDA-GATv2 Training Convergence Summary")
    print("=" * 60)
    print(f"Total cycles analyzed : {len(df)}")
    print(f"Final validation AUC  : {final_auc:.4f}")
    print(f"Best validation AUC   : {best_auc:.4f} (Cycle {best_cycle})")
    print(f"AUC improvement       : {best_auc - final_auc:+.4f}")
    print(f"Final train loss      : {df['train_loss'].iloc[-1]:.6f}")
    print(f"Final train accuracy  : {df['train_accuracy'].iloc[-1]:.4f}")
    print(f"Final val accuracy    : {df['val_accuracy'].iloc[-1]:.4f}")
    print(f"Generalization gap    : {df['train_accuracy'].iloc[-1] - df['val_accuracy'].iloc[-1]:.4f}")
    print("=" * 60)
    print(f"PNG saved : {png_path}")
    print(f"PDF saved : {pdf_path}")


if __name__ == "__main__":
    main()
