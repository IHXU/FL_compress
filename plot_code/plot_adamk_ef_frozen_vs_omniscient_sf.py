"""Compare AdamK EF frozen-zero-momentum against Non-IID omniscient SF baselines."""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = ROOT / "results"
BASELINE_ROOT = RESULTS_ROOT / "results_main_noniid_omniscient_krum"
NEW_ROOT = RESULTS_ROOT / "results_adamk_ef_frozen_zero_momentum"
OUTPUT = NEW_ROOT / "adamk_ef_frozen_vs_omniscient_sf_acc"

METRIC_PATTERN = re.compile(
    r"Round\s+(?P<round>\d+)\s+Metric\s+top1_accuracy:\s*"
    r"(?P<accuracy>[-+0-9.eE]+)"
)

METHODS = (
    ("adamk_compress4_clip", "ADAMK + Compress4-Clip"),
    ("byz_dm21_krum", "Byz-DM21 + Krum"),
    ("byz_vr_dm21_krum", "Byz-VR-DM21 + Krum"),
    ("byz_ef21_sgdm_krum", "Byz-EF21-SGDM + Krum"),
    ("byz_sgdm_broadcast_krum", "Byz-SGDM-Broadcast + Krum"),
    ("d_byz_sgdm_krum", "D-Byz-SGDM + Krum"),
    ("fed_dproc_krum", "Fed-DPRoC + Krum"),
    ("rosdhb_krum", "RoSDHB + Krum"),
)


def read_curve(folder: Path) -> tuple[np.ndarray, np.ndarray]:
    """Merge logs chronologically; later entries replace duplicate rounds."""
    values: dict[int, float] = {}
    logs = sorted(folder.glob("experiment_*.log"), key=lambda p: p.stat().st_mtime)
    for log in logs:
        with log.open("r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                match = METRIC_PATTERN.search(line)
                if match:
                    values[int(match.group("round"))] = float(match.group("accuracy"))
    if not values:
        raise RuntimeError(f"No top1_accuracy entries found in {folder}")
    rounds = np.asarray(sorted(values), dtype=int)
    accuracy = np.asarray([values[round_] for round_ in rounds], dtype=float)
    return rounds, accuracy


def main() -> None:
    curves: list[tuple[str, np.ndarray, np.ndarray, bool]] = []
    new_folder = (
        NEW_ROOT
        / "cifar10_adamk_ef_frozen_zero_momentum_omniscient_signflipping_noniid_alpha01"
    )
    rounds, accuracy = read_curve(new_folder)
    curves.append(("AdamK + EF (Frozen Zero Momentum)", rounds, accuracy, True))

    for method, label in METHODS:
        folder = (
            BASELINE_ROOT
            / f"cifar10_{method}_omniscient_signflipping_noniid_alpha01"
        )
        rounds, accuracy = read_curve(folder)
        curves.append((label, rounds, accuracy, False))

    for label, rounds, _, _ in curves:
        if len(rounds) != 101 or rounds[0] != 0 or rounds[-1] != 100:
            raise RuntimeError(
                f"{label} is incomplete: {len(rounds)} points, rounds {rounds[0]}-{rounds[-1]}"
            )

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10.5,
            "axes.titlesize": 14,
            "axes.titleweight": "bold",
            "axes.labelsize": 12,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
        }
    )

    baseline_colors = plt.cm.plasma(np.linspace(0.05, 0.92, len(METHODS)))
    baseline_styles = ("--", "-.", ":", "--", "-.", ":", "--", "-.")
    fig, ax = plt.subplots(figsize=(13.2, 7.4), dpi=180)

    baseline_index = 0
    for label, rounds, accuracy, is_new in curves:
        values = accuracy * 100.0
        if is_new:
            ax.plot(
                rounds,
                values,
                color="#d62728",
                linewidth=3.2,
                linestyle="-",
                label=label,
                zorder=20,
            )
            ax.scatter(
                rounds[-1], values[-1], color="#d62728", s=42, zorder=21
            )
        else:
            ax.plot(
                rounds,
                values,
                color=baseline_colors[baseline_index],
                linewidth=1.55,
                linestyle=baseline_styles[baseline_index],
                label=label,
                alpha=0.9,
            )
            baseline_index += 1

    all_values = np.concatenate([accuracy * 100.0 for _, _, accuracy, _ in curves])
    lower = max(0, 5 * np.floor((all_values.min() - 2) / 5))
    upper = min(100, 5 * np.ceil((all_values.max() + 2) / 5))
    ax.set_title(r"CIFAR-10 Non-IID ($\alpha=0.1$), Omniscient Sign-Flipping Attack")
    ax.set_xlabel("Communication Round")
    ax.set_ylabel("Top-1 Accuracy (%)")
    ax.set_xlim(0, 100)
    ax.set_ylim(lower, upper)
    ax.set_xticks(np.arange(0, 101, 10))
    ax.grid(True, color="#d9d9d9", linewidth=0.7, alpha=0.75)
    ax.legend(loc="best", ncol=2, fontsize=9.2, columnspacing=1.2)

    fig.tight_layout()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(OUTPUT.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)

    print(f"Saved {OUTPUT.with_suffix('.png')}")
    print(f"Saved {OUTPUT.with_suffix('.pdf')}")
    for label, rounds, accuracy, _ in curves:
        print(
            f"{label}: {len(rounds)} rounds, final={accuracy[-1] * 100:.2f}%, "
            f"best={accuracy.max() * 100:.2f}%"
        )


if __name__ == "__main__":
    main()
