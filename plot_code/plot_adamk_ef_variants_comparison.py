"""Plot AdamK EF variants against AdamK, Fed-DPRoC, and RoSDHB."""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
VARIANT_ROOT = ROOT / "results" / "results_adamk_ef_variants_noniid"
BASELINE_ROOT = ROOT / "results" / "results_main_noniid_omniscient_krum"
OUTPUT = VARIANT_ROOT / "adamk_ef_variants_vs_fed_dproc_rosdhb_acc"

METRIC_PATTERN = re.compile(
    r"Round\s+(?P<round>\d+)\s+Metric\s+top1_accuracy:\s*"
    r"(?P<accuracy>[-+0-9.eE]+)"
)

ATTACKS = (
    ("withoutatt", "No Attack"),
    ("omniscient_foe", "Omniscient FoE"),
    ("omniscient_signflipping", "Omniscient Sign Flipping"),
)

METHODS = (
    (
        "Original AdamK",
        BASELINE_ROOT,
        "cifar10_adamk_compress4_clip_{attack}_noniid_alpha01",
    ),
    (
        "V1: EF, No WD",
        VARIANT_ROOT,
        "cifar10_adamk_v1_ef_nowd_{attack}_noniid_alpha01",
    ),
    (
        "V2: EF, No WD/ARC",
        VARIANT_ROOT,
        "cifar10_adamk_v2_ef_nowd_noclip_{attack}_noniid_alpha01",
    ),
    (
        r"V3: Client Momentum, $\beta_1=0$",
        VARIANT_ROOT,
        "cifar10_adamk_v3_ef_clientmom_b10_{attack}_noniid_alpha01",
    ),
    (
        "V4: Active-Mask Update",
        VARIANT_ROOT,
        "cifar10_adamk_v4_ef_masked_active_{attack}_noniid_alpha01",
    ),
    (
        "Fed-DPRoC",
        BASELINE_ROOT,
        "cifar10_fed_dproc_krum_{attack}_noniid_alpha01",
    ),
    (
        "RoSDHB",
        BASELINE_ROOT,
        "cifar10_rosdhb_krum_{attack}_noniid_alpha01",
    ),
)

STYLES = {
    "Original AdamK": dict(color="#111111", linewidth=2.8, linestyle="-"),
    "V1: EF, No WD": dict(color="#1f77b4", linewidth=2.0, linestyle="-"),
    "V2: EF, No WD/ARC": dict(
        color="#2ca02c", linewidth=2.0, linestyle="--"
    ),
    r"V3: Client Momentum, $\beta_1=0$": dict(
        color="#d62728", linewidth=2.4, linestyle="-"
    ),
    "V4: Active-Mask Update": dict(
        color="#ff7f0e", linewidth=2.0, linestyle="-."
    ),
    "Fed-DPRoC": dict(color="#9467bd", linewidth=2.3, linestyle="--"),
    "RoSDHB": dict(color="#8c564b", linewidth=2.3, linestyle=":"),
}


def read_curve(folder: Path) -> tuple[np.ndarray, np.ndarray]:
    """Merge resumed logs, with later logs replacing duplicate rounds."""
    values: dict[int, float] = {}
    logs = sorted(folder.glob("experiment_*.log"), key=lambda p: p.stat().st_mtime)
    for log in logs:
        with log.open("r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                match = METRIC_PATTERN.search(line)
                if match:
                    values[int(match.group("round"))] = float(
                        match.group("accuracy")
                    )
    if not values:
        raise RuntimeError(f"No top1_accuracy data found in {folder}")
    rounds = np.asarray(sorted(values), dtype=int)
    accuracy = np.asarray([values[round_] for round_ in rounds], dtype=float)
    if len(rounds) != 101 or rounds[0] != 0 or rounds[-1] != 100:
        raise RuntimeError(
            f"Incomplete curve in {folder}: {len(rounds)} points, "
            f"rounds {rounds[0]}-{rounds[-1]}"
        )
    return rounds, accuracy


def main() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10.5,
            "axes.titlesize": 13,
            "axes.titleweight": "bold",
            "axes.labelsize": 11.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
        }
    )

    fig, axes = plt.subplots(1, 3, figsize=(19.2, 6.3), dpi=180, sharey=True)
    handles = None
    labels = None

    for ax, (attack, title) in zip(axes, ATTACKS):
        for label, root, folder_template in METHODS:
            folder = root / folder_template.format(attack=attack)
            rounds, accuracy = read_curve(folder)
            ax.plot(
                rounds,
                accuracy * 100.0,
                label=label,
                **STYLES[label],
            )

        ax.set_title(title)
        ax.set_xlabel("Communication Round")
        ax.set_xlim(0, 100)
        ax.set_ylim(10, 50)
        ax.set_xticks(np.arange(0, 101, 20))
        ax.set_yticks(np.arange(10, 51, 5))
        ax.grid(True, color="#d9d9d9", linewidth=0.7, alpha=0.75)
        if handles is None:
            handles, labels = ax.get_legend_handles_labels()

    axes[0].set_ylabel("Top-1 Accuracy (%)")
    fig.suptitle(
        r"CIFAR-10 Non-IID ($\alpha=0.1$): AdamK EF Variants and Baselines",
        fontsize=16,
        fontweight="bold",
        y=0.995,
    )
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.01),
        ncol=4,
        fontsize=10,
        columnspacing=1.5,
        handlelength=3.0,
    )
    fig.tight_layout(rect=(0, 0.105, 1, 0.95))
    fig.savefig(OUTPUT.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(OUTPUT.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(OUTPUT.with_suffix(".png"))
    print(OUTPUT.with_suffix(".pdf"))


if __name__ == "__main__":
    main()
