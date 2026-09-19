"""Plot final-round accuracy for AdamK EF variants and baselines."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from plot_adamk_ef_variants_comparison import (
    ATTACKS,
    METHODS,
    STYLES,
    VARIANT_ROOT,
    read_curve,
)


OUTPUT = VARIANT_ROOT / "adamk_ef_variants_vs_fed_dproc_rosdhb_final_acc"


def main() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10.5,
            "axes.titlesize": 15,
            "axes.titleweight": "bold",
            "axes.labelsize": 12,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
        }
    )

    attack_labels = [title for _, title in ATTACKS]
    x = np.arange(len(ATTACKS), dtype=float)
    width = 0.115
    offsets = (np.arange(len(METHODS)) - (len(METHODS) - 1) / 2) * width

    fig, ax = plt.subplots(figsize=(14.5, 7.2), dpi=180)
    for offset, (label, root, folder_template) in zip(offsets, METHODS):
        final_accuracies = []
        for attack, _ in ATTACKS:
            folder = root / folder_template.format(attack=attack)
            rounds, accuracy = read_curve(folder)
            if rounds[-1] != 100:
                raise RuntimeError(f"Final round is not 100 in {folder}")
            final_accuracies.append(accuracy[-1] * 100.0)

        bars = ax.bar(
            x + offset,
            final_accuracies,
            width=width * 0.92,
            label=label,
            color=STYLES[label]["color"],
            edgecolor="white",
            linewidth=0.7,
        )
        ax.bar_label(
            bars,
            labels=[f"{value:.1f}" for value in final_accuracies],
            padding=2,
            fontsize=8.2,
            rotation=90,
        )

    ax.set_title(r"CIFAR-10 Non-IID ($\alpha=0.1$): Final Accuracy at Round 100")
    ax.set_ylabel("Top-1 Accuracy (%)")
    ax.set_xticks(x, attack_labels)
    ax.set_ylim(0, 52)
    ax.set_yticks(np.arange(0, 51, 5))
    ax.grid(axis="y", color="#d9d9d9", linewidth=0.7, alpha=0.8)
    ax.set_axisbelow(True)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.095),
        ncol=4,
        fontsize=9.5,
        columnspacing=1.4,
        handlelength=1.6,
    )
    fig.tight_layout(rect=(0, 0.09, 1, 1))
    fig.savefig(OUTPUT.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(OUTPUT.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(OUTPUT.with_suffix(".png"))
    print(OUTPUT.with_suffix(".pdf"))


if __name__ == "__main__":
    main()
