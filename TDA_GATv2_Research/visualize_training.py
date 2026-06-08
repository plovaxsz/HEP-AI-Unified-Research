"""Scientific visualization pipeline for TDA-GATv2 training metrics."""

import pathlib

import matplotlib.pyplot as plt
import pandas as pd

LOG_PATH = pathlib.Path(r"D:\PISS\TDA_GATv2_Research\models\evolution.log")
OUTPUT_PATH = pathlib.Path(r"D:\PISS\TDA_GATv2_Research\models\training_visualization.png")

plt.style.use("seaborn-v0_8-darkgrid")


def main() -> None:
    df = pd.read_csv(LOG_PATH)
    df.columns = [c.strip() for c in df.columns]

    data_rows = df[df["epoch"].notna() & (df["epoch"] != "epoch")]
    if data_rows.empty:
        print(f"No training data found in {LOG_PATH}")
        print("Run training first: python train.py --epochs 100 ...")
        return

    data_rows = data_rows.copy()
    data_rows["epoch"] = data_rows["epoch"].astype(int)
    data_rows["train_loss"] = data_rows["train_loss"].astype(float)
    data_rows["val_auc"] = data_rows["val_auc"].astype(float)
    data_rows["val_accuracy"] = data_rows["val_accuracy"].astype(float)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("TDA-GATv2 Training Metrics — IPR Protocol", fontsize=14, fontweight="bold")

    epochs = data_rows["epoch"].values

    ax1.plot(epochs, data_rows["train_loss"].values, color="red", linewidth=1.2, label="Train Loss (raw)")
    data_rows["loss_ma"] = data_rows["train_loss"].rolling(window=5, min_periods=1).mean()
    ax1.plot(epochs, data_rows["loss_ma"].values, color="darkred", linewidth=2.0, label="Train Loss (MA-5)")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training Loss Convergence")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    best_idx = int(data_rows["val_auc"].idxmax())
    best_epoch = int(data_rows.loc[best_idx, "epoch"])
    best_auc = float(data_rows.loc[best_idx, "val_auc"])

    ax2.plot(epochs, data_rows["val_auc"].values, color="blue", linewidth=1.5, label="Val AUC")
    ax2.plot(epochs, data_rows["val_accuracy"].values, color="green", linewidth=1.5, label="Val Accuracy")
    ax2.axvline(x=best_epoch, color="black", linestyle="--", linewidth=1.2, alpha=0.7, label=f"Best AUC @ epoch {best_epoch}")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Metric Value")
    ax2.set_title("Validation Metrics")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_PATH, dpi=300, bbox_inches="tight")
    print(f"Visualization saved to: {OUTPUT_PATH}")
    print(f"Best validation AUC: {best_auc:.4f} at epoch {best_epoch}")
    plt.show()


if __name__ == "__main__":
    main()
