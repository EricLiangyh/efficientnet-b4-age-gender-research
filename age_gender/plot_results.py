"""Generate focused research-stage figures directly from historical CSV logs."""
import csv
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

GROUPS = [
    ("backbone_selection_15", "1. Backbone selection", [("b4_15", "EfficientNet-B4"), ("resnet50_15", "ResNet-50"), ("vgg16_15", "VGG-16")], "Compare model families under a 15-epoch budget."),
    ("efficientnet_scaling_15", "2. EfficientNet scaling", [("b4_15", "EfficientNet-B4"), ("b0_15", "EfficientNet-B0")], "Compare B4 and B0 under a 15-epoch budget."),
    ("augmentation_ablation_15", "3a. Data augmentation", [("b4_15", "Full augmentation"), ("b4_no_augmentation_15", "No augmentation")], "Hold the B4 baseline fixed and compare augmentation settings."),
    ("loss_weight_ablation_15", "3b. Task-loss weighting", [("b4_15", "Gender loss weight = 6"), ("b4_gender_weight_1_15", "Gender loss weight = 1")], "Compare gender-loss coefficients for the 15-epoch B4 runs."),
    ("alignment_ablation_15", "3c. Face alignment", [("b4_15", "With alignment"), ("b4_no_alignment_15", "Without alignment")], "Compare the archived aligned and unaligned B4 experiments."),
]
COLORS = ["#1768ac", "#df7b24", "#7656a4"]


def decorate(ax, epochs, ylabel):
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.set_xlim(1, epochs)
    ax.set_xticks([1, 3, 5, 7, 9, 11, 13, 15] if epochs == 15 else [1, 5, 10, 15, 20, 25, 30, 35])
    ax.grid(axis="y", color="#dbe3eb", linewidth=0.7)
    ax.spines[["top", "right"]].set_visible(False)


def compare(histories, filename, title, members, destination):
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.9))
    for color, (key, label) in zip(COLORS, members):
        rows = histories[key]
        if len(rows) != 15:
            raise ValueError(f"Expected 15 epochs for {key}")
        epochs = [int(r["epoch"]) for r in rows]
        for ax, field in zip(axes, ["val_mae", "val_acc"]):
            values = [float(r[field]) for r in rows]
            ax.plot(epochs, values, color=color, label=label, linewidth=2.1,
                    marker="o", markersize=3.2)
            ax.scatter(epochs[-1], values[-1], color=color, s=35, zorder=3)
    for ax, label, field in zip(axes, ["Age MAE (years; lower is better)", "Gender accuracy (%; higher is better)"], ["val_mae", "val_acc"]):
        decorate(ax, 15, label)
        ax.legend(loc="best", frameon=True, fontsize=9)
        detail = " | ".join(f"{label}: {float(histories[key][-1][field]):.4f}" for key, label in members)
        ax.text(0, -0.25, "Epoch 15 — " + detail, transform=ax.transAxes, fontsize=8.2)
    fig.suptitle(title + " · 15 epochs", fontsize=15, fontweight="bold", y=0.98)
    fig.text(0.5, 0.015, "Historical validation logs · single-run observations · not independent test results", ha="center", fontsize=9, color="#596675")
    fig.subplots_adjust(left=0.07, right=0.98, top=0.87, bottom=0.27, wspace=0.23)
    fig.savefig(destination / (filename + ".png"), dpi=180, facecolor="white", bbox_inches="tight", pad_inches=0.18)
    plt.close(fig)


def final_training(rows, destination):
    epochs = [int(r["epoch"]) for r in rows]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.7))
    axes[0].plot(epochs, [float(r["train_loss"]) for r in rows], color=COLORS[0], label="Training loss", linewidth=2)
    axes[0].plot(epochs, [float(r["val_loss"]) for r in rows], color=COLORS[1], label="Validation loss", linewidth=2)
    axes[0].legend(fontsize=9)
    for ax, field, ylabel, color in [(axes[1], "val_mae", "Validation age MAE (years)", COLORS[0]), (axes[2], "val_acc", "Validation gender accuracy (%)", COLORS[1])]:
        values = [float(r[field]) for r in rows]
        ax.plot(epochs, values, color=color, linewidth=2)
        ax.scatter(epochs[-1], values[-1], color=color, s=35)
        ax.text(0.04, 0.07, f"Epoch {epochs[-1]}: {values[-1]:.4f}", transform=ax.transAxes, fontsize=10,
                bbox={"facecolor": "white", "edgecolor": "#dbe3eb", "alpha": 0.9})
        decorate(ax, epochs[-1], ylabel)
    decorate(axes[0], epochs[-1], "Weighted joint loss")
    fig.suptitle("4. Final EfficientNet-B4 training · full 35-epoch run", fontsize=15, fontweight="bold", y=0.98)
    fig.text(0.5, 0.025, "C3 historical record · selected configuration · validation results, not a new test evaluation", ha="center", fontsize=9, color="#596675")
    fig.subplots_adjust(left=0.055, right=0.985, top=0.86, bottom=0.17, wspace=0.28)
    fig.savefig(destination / "final_training_35.png", dpi=180, facecolor="white", bbox_inches="tight", pad_inches=0.18)
    plt.close(fig)


def main():
    root = Path(__file__).resolve().parents[1]
    histories, records = {}, []
    for path in sorted((root / "experiments/logs").glob("*.csv")):
        with path.open(encoding="utf-8-sig") as stream:
            rows = list(csv.DictReader(stream))
        histories[path.stem] = rows
        last = rows[-1]
        records.append({"run": path.stem, "epochs": int(last["epoch"]), "val_mae_years": float(last["val_mae"]), "val_accuracy_percent": float(last["val_acc"]), "source": path.relative_to(root).as_posix()})
    destination = root / "experiments/figures"
    destination.mkdir(exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    sections = ["# Historical experiments by research stage", "", "Figures are organized by research question. This is a presentation order, not a claim about the chronological order of archived runs. All values are historical validation observations, not independently remeasured test results.", ""]
    for filename, title, members, description in GROUPS:
        compare(histories, filename, title, members, destination)
        sections += ["## " + title, "", description, "", "| Configuration | Epochs | Age MAE (years) | Gender accuracy (%) | Source |", "| --- | ---: | ---: | ---: | --- |"]
        for key, label in members:
            row = histories[key][-1]
            sections.append(f"| {label} | 15 | {float(row['val_mae']):.4f} | {float(row['val_acc']):.4f} | [CSV](../experiments/logs/{key}.csv) |")
        sections += ["", f"![{title}](../experiments/figures/{filename}.png)", ""]
    final_training(histories["b4_35"], destination)
    sections += ["## 4. Final training: selected EfficientNet-B4 configuration", "", "The final figure uses the complete 35-epoch C3 record. It separates training/validation joint loss, validation age MAE and validation gender accuracy. It is not mixed with the 15-epoch comparisons.", "", "![Final training](../experiments/figures/final_training_35.png)", "", "[Final-run CSV](../experiments/logs/b4_35.csv) · [Provenance and comparison limitations](provenance.md)", "", "Archived runs do not establish fully controlled comparisons or statistical significance. Original split membership is unavailable; preprocessing and architecture details should be audited before drawing causal conclusions."]
    (root / "docs/experiments.md").write_text("\n".join(sections) + "\n", encoding="utf-8")
    (root / "experiments/summary.json").write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
    print("Generated 6 focused figures from 8 historical CSV logs")


if __name__ == "__main__":
    main()
