"""Generate publication-ready research plots for TDA-GATv2 training analysis.

Produces 5 figures from evolution.log cycle-level summaries:
  - Figure 1: Convergence Overview (loss, accuracy, AUC)
  - Figure 2: Computational Resource Utilization (CPU, RAM)
  - Figure 3: Learning Stability (box/violin plot of val_auc)
  - Figure 4: Correlation Matrix (feature correlations)
  - Figure 5: Training Efficiency (accuracy vs loss scatter + regression)

All figures saved as high-DPI PNG and vector PDF.
"""

from __future__ import annotations

import pathlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec

LOG_PATH = pathlib.Path(r"D:\PISS\TDA_GATv2_Research\models\evolution.log")
OUTPUT_DIR = pathlib.Path(r"D:\PISS\TDA_GATv2_Research\models")

try:
    plt.style.use("seaborn-v0_8-darkgrid")
except OSError:
    plt.style.use("seaborn-darkgrid")

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


def load_cycle_summaries(log_path: pathlib.Path) -> pd.DataFrame:
    """Parse evolution.log and extract cycle-level summary rows only."""

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
                cycle = int(parts[0])
                rows.append(
                    {
                        "cycle": cycle,
                        "val_auc": float(parts[1]),
                        "train_loss": float(parts[2]),
                        "train_accuracy": float(parts[3]),
                        "val_accuracy": float(parts[4]),
                        "learning_rate": float(parts[5]),
                        "epochs_completed": int(parts[6]),
                        "status": parts[7],
                        "cpu_percent": float(parts[8]) if len(parts) > 8 else 0.0,
                        "ram_percent": float(parts[9]) if len(parts) > 9 else 0.0,
                    }
                )
            except ValueError:
                continue

    df = pd.DataFrame(rows).sort_values("cycle").reset_index(drop=True)
    return df


def save_figure(fig: plt.Figure, name: str) -> None:
    """Save figure as PNG and PDF with consistent naming."""

    png_path = OUTPUT_DIR / f"{name}.png"
    pdf_path = OUTPUT_DIR / f"{name}.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")


def figure_1_convergence(df: pd.DataFrame) -> None:
    """Figure 1: 3-panel convergence overview (loss, accuracy, AUC)."""

    cycles = df["cycle"].values
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    fig.suptitle(
        "Figure 1: TDA-GATv2 Training Convergence Overview (50 Cycles, 5,000 Epochs)",
        fontsize=13,
        fontweight="bold",
        y=1.02,
    )

    # Loss
    ax = axes[0]
    ax.plot(cycles, df["train_loss"].values, color="crimson", linewidth=2, marker="o", markersize=5)
    ax.set_xlabel("Cycle (×100 epochs)")
    ax.set_ylabel("Train Loss")
    ax.set_title("(a) Training Loss Convergence")
    ax.grid(True, alpha=0.3)

    # Accuracy
    ax = axes[1]
    ax.plot(cycles, df["train_accuracy"].values, color="seagreen", linewidth=2, marker="s", markersize=5, label="Train Accuracy")
    ax.plot(cycles, df["val_accuracy"].values, color="dodgerblue", linewidth=2, marker="^", markersize=5, label="Validation Accuracy")
    ax.set_xlabel("Cycle (×100 epochs)")
    ax.set_ylabel("Accuracy")
    ax.set_title("(b) Train vs Validation Accuracy")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # AUC
    ax = axes[2]
    best_idx = int(df["val_auc"].idxmax())
    best_cycle = int(df.loc[best_idx, "cycle"])
    best_auc = float(df.loc[best_idx, "val_auc"])
    ax.plot(cycles, df["val_auc"].values, color="navy", linewidth=2, marker="D", markersize=5)
    ax.axhline(y=best_auc, color="goldenrod", linestyle="--", linewidth=1.5, alpha=0.8)
    ax.axvline(x=best_cycle, color="crimson", linestyle="--", linewidth=1.5, alpha=0.8)
    ax.annotate(f"Best AUC = {best_auc:.4f}\n@ Cycle {best_cycle}", xy=(best_cycle, best_auc),
                xytext=(best_cycle - 8, best_auc + 0.002), fontsize=9,
                arrowprops=dict(arrowstyle="->", color="black", lw=0.8))
    ax.set_xlabel("Cycle (×100 epochs)")
    ax.set_ylabel("Validation AUC")
    ax.set_title("(c) Validation AUC Convergence")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_figure(fig, "figure1_convergence")
    plt.close(fig)


def figure_2_resources(df: pd.DataFrame) -> None:
    """Figure 2: CPU and RAM utilization per cycle."""

    cycles = df["cycle"].values
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(
        "Figure 2: Computational Resource Utilization per Cycle",
        fontsize=13,
        fontweight="bold",
        y=1.02,
    )

    # CPU
    ax = axes[0]
    ax.bar(cycles, df["cpu_percent"].values, color="steelblue", edgecolor="black", linewidth=0.5)
    ax.set_xlabel("Cycle (×100 epochs)")
    ax.set_ylabel("CPU Usage (%)")
    ax.set_title("(a) CPU Utilization")
    ax.grid(True, alpha=0.3, axis="y")

    # RAM
    ax = axes[1]
    ax.bar(cycles, df["ram_percent"].values, color="darkorange", edgecolor="black", linewidth=0.5)
    ax.set_xlabel("Cycle (×100 epochs)")
    ax.set_ylabel("RAM Usage (%)")
    ax.set_title("(b) RAM Utilization")
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    save_figure(fig, "figure2_resources")
    plt.close(fig)


def figure_3_stability(df: pd.DataFrame) -> None:
    """Figure 3: Box + Violin plot of val_auc distribution across cycles."""

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(
        "Figure 3: Learning Stability — Validation AUC Distribution (50 Cycles)",
        fontsize=13,
        fontweight="bold",
        y=1.02,
    )

    data = df["val_auc"].values

    # Box plot
    ax = axes[0]
    bp = ax.boxplot(data, vert=True, patch_artist=True, showmeans=True, meanline=True)
    for patch in bp["boxes"]:
        patch.set_facecolor("lightblue")
        patch.set_edgecolor("navy")
    ax.set_ylabel("Validation AUC")
    ax.set_title("(a) Box Plot")
    ax.grid(True, alpha=0.3, axis="y")

    # Violin plot
    ax = axes[1]
    parts = ax.violinplot(data, positions=[1], showmeans=True, showmedians=True)
    for pc in parts["bodies"]:
        pc.set_facecolor("lightcoral")
        pc.set_edgecolor("crimson")
        pc.set_alpha(0.7)
    ax.set_xticks([1])
    ax.set_xticklabels(["Val AUC"])
    ax.set_ylabel("Validation AUC")
    ax.set_title("(b) Violin Plot")
    ax.grid(True, alpha=0.3, axis="y")

    # Annotate stats
    mean_auc = np.mean(data)
    std_auc = np.std(data)
    ax.text(0.95, 0.95, f"Mean = {mean_auc:.4f}\nStd = {std_auc:.4f}",
            transform=ax.transAxes, fontsize=10, verticalalignment="top", horizontalalignment="right",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8))

    plt.tight_layout()
    save_figure(fig, "figure3_stability")
    plt.close(fig)


def figure_4_correlation(df: pd.DataFrame) -> None:
    """Figure 4: Correlation heatmap of key metrics."""

    corr_cols = ["train_loss", "train_accuracy", "val_accuracy", "val_auc"]
    corr_df = df[corr_cols].astype(float)
    corr_matrix = corr_df.corr()

    fig, ax = plt.subplots(figsize=(7, 6))
    fig.suptitle(
        "Figure 4: Feature Correlation Matrix (Cycle-Level Metrics)",
        fontsize=13,
        fontweight="bold",
        y=1.02,
    )

    im = ax.imshow(corr_matrix, cmap="coolwarm", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(corr_cols)))
    ax.set_yticks(range(len(corr_cols)))
    ax.set_xticklabels(["Train Loss", "Train Acc", "Val Acc", "Val AUC"], rotation=45, ha="right")
    ax.set_yticklabels(["Train Loss", "Train Acc", "Val Acc", "Val AUC"])

    for i in range(len(corr_cols)):
        for j in range(len(corr_cols)):
            ax.text(j, i, f"{corr_matrix.iloc[i, j]:.3f}", ha="center", va="center", fontsize=10,
                    color="white" if abs(corr_matrix.iloc[i, j]) > 0.5 else "black")

    fig.colorbar(im, ax=ax, label="Pearson Correlation")
    ax.set_title("Correlation Heatmap")
    plt.tight_layout()
    save_figure(fig, "figure4_correlation")
    plt.close(fig)


def figure_5_efficiency(df: pd.DataFrame) -> None:
    """Figure 5: Training efficiency — accuracy vs loss scatter with regression."""

    fig, ax = plt.subplots(figsize=(8, 6))
    fig.suptitle(
        "Figure 5: Training Efficiency — Accuracy vs Loss Trajectory",
        fontsize=13,
        fontweight="bold",
        y=1.02,
    )

    x = df["train_loss"].values
    y = df["train_accuracy"].values
    cycles = df["cycle"].values

    # Scatter colored by cycle progression
    sc = ax.scatter(x, y, c=cycles, cmap="viridis", s=60, edgecolors="black", linewidth=0.5, zorder=3)
    ax.plot(x, y, color="gray", linewidth=1, alpha=0.6, zorder=2)

    # Linear regression
    coeffs = np.polyfit(x, y, deg=1)
    x_fit = np.linspace(x.min(), x.max(), 100)
    y_fit = np.polyval(coeffs, x_fit)
    ax.plot(x_fit, y_fit, color="crimson", linewidth=2, linestyle="--", label=f"Linear Fit: y = {coeffs[0]:.2f}x + {coeffs[1]:.3f}")

    # Colorbar for cycle progression
    cbar = fig.colorbar(sc, ax=ax, label="Cycle Number")
    cbar.set_ticks([1, 10, 20, 30, 40, 50])

    ax.set_xlabel("Train Loss")
    ax.set_ylabel("Train Accuracy")
    ax.set_title("Accuracy vs Loss with Linear Regression")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_figure(fig, "figure5_efficiency")
    plt.close(fig)


def print_summary(df: pd.DataFrame) -> None:
    """Print console summary of training results."""

    final_auc = float(df["val_auc"].iloc[-1])
    best_auc = float(df["val_auc"].max())
    best_cycle = int(df.loc[df["val_auc"].idxmax(), "cycle"])
    worst_auc = float(df["val_auc"].min())
    mean_auc = float(df["val_auc"].mean())
    std_auc = float(df["val_auc"].std())

    print("=" * 70)
    print("TDA-GATv2 Research Summary — 50-Cycle IPR Protocol")
    print("=" * 70)
    print(f"Total cycles         : {len(df)}")
    print(f"Total epochs         : {int(df['epochs_completed'].sum()):,}")
    print(f"Final val_auc        : {final_auc:.4f}")
    print(f"Best  val_auc        : {best_auc:.4f} (Cycle {best_cycle})")
    print(f"Worst val_auc        : {worst_auc:.4f}")
    print(f"Mean  val_auc        : {mean_auc:.4f}")
    print(f"Std   val_auc        : {std_auc:.4f}")
    print(f"AUC range            : {worst_auc:.4f} – {best_auc:.4f}")
    print(f"Final train loss     : {df['train_loss'].iloc[-1]:.6f}")
    print(f"Final train accuracy : {df['train_accuracy'].iloc[-1]:.4f}")
    print(f"Final val accuracy   : {df['val_accuracy'].iloc[-1]:.4f}")
    print(f"Generalization gap   : {df['train_accuracy'].iloc[-1] - df['val_accuracy'].iloc[-1]:.4f}")
    print("=" * 70)


def main() -> None:
    if not LOG_PATH.exists():
        raise FileNotFoundError(f"Log file not found: {LOG_PATH}")

    df = load_cycle_summaries(LOG_PATH)
    if df.empty:
        raise ValueError("No cycle-level data found in evolution.log")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    figure_1_convergence(df)
    figure_2_resources(df)
    figure_3_stability(df)
    figure_4_correlation(df)
    figure_5_efficiency(df)
    print_summary(df)


if __name__ == "__main__":
    main()
