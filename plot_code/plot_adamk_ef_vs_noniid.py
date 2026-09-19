"""Plot per-round accuracy for AdamK+EF and Non-IID baselines."""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "results_main_noniid"
EF_RESULTS = ROOT / "results" / "results_adamk_ef_ablation"
OUTPUT = EF_RESULTS / "figures" / "adamk_ef_vs_noniid_signflipping_acc"

METRIC_PATTERN = re.compile(
    r"Round\s+(?P<round>\d+)\s+Metric\s+top1_accuracy:\s*"
    r"(?P<accuracy>[-+0-9.eE]+)"
)


def read_curve(*logs: Path) -> tuple[np.ndarray, np.ndarray]:
    """Merge resumed logs into one sorted per-round accuracy curve."""
    values: dict[int, float] = {}
    for log in logs:
        with log.open("r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                match = METRIC_PATTERN.search(line)
                if match:
                    values[int(match.group("round"))] = float(
                        match.group("accuracy")
                    )
    rounds = np.asarray(sorted(values), dtype=int)
    accuracy = np.asarray([values[round_] for round_ in rounds], dtype=float)
    return rounds, accuracy


def old_log(folder: str) -> Path:
    return RESULTS / folder / "experiment_20260829_142714.log"


CURVES = {
    "AdamK + Error Feedback": (
        EF_RESULTS
        / "b_error_feedback_signflipping"
        / "experiment_20260830_160345.log",
    ),
    "Original AdamK": (
        old_log("cifar10_adamk_compress4_clip_signflipping_noniid_alpha01"),
    ),
    "Byz-DM21": (
        old_log("cifar10_byz_dm21_krum_signflipping_noniid_alpha01"),
    ),
    # These two experiments have identical loss and accuracy at every round.
    "Byz-EF21 / Byz-VR-DM21 (identical)": (
        old_log("cifar10_byz_ef21_sgdm_krum_signflipping_noniid_alpha01"),
    ),
    "Byz-SGDM-Broadcast": (
        old_log("cifar10_byz_sgdm_broadcast_krum_signflipping_noniid_alpha01"),
        RESULTS
        / "cifar10_byz_sgdm_broadcast_krum_signflipping_noniid_alpha01"
        / "experiment_20260830_101126.log",
    ),
    "D-Byz-SGDM": (
        old_log("cifar10_d_byz_sgdm_krum_signflipping_noniid_alpha01"),
    ),
    "Fed-DPRoC": (
        old_log("cifar10_fed_dproc_krum_signflipping_noniid_alpha01"),
        RESULTS
        / "cifar10_fed_dproc_krum_signflipping_noniid_alpha01"
        / "experiment_20260830_100850.log",
    ),
    "RoSD-HB": (
        old_log("cifar10_rosdhb_krum_signflipping_noniid_alpha01"),
    ),
}

STYLES = {
    "AdamK + Error Feedback": dict(
        color="#d62728", linewidth=3.0, linestyle="-", zorder=20
    ),
    "Original AdamK": dict(
        color="#4d4d4d", linewidth=2.2, linestyle="--", zorder=16
    ),
    "Byz-DM21": dict(color="#1f77b4", linewidth=1.35, linestyle="-"),
    "Byz-EF21 / Byz-VR-DM21 (identical)": dict(
        color="#9467bd", linewidth=1.45, linestyle="-"
    ),
    "Byz-SGDM-Broadcast": dict(
        color="#8c564b", linewidth=1.2, linestyle=":"
    ),
    "D-Byz-SGDM": dict(color="#e377c2", linewidth=1.2, linestyle="-."),
    "Fed-DPRoC": dict(color="#2ca02c", linewidth=1.8, linestyle="-"),
    "RoSD-HB": dict(color="#ff7f0e", linewidth=1.8, linestyle="-"),
}


def main() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.titlesize": 14,
            "axes.titleweight": "bold",
            "axes.labelsize": 12,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
        }
    )

    fig, ax = plt.subplots(figsize=(12.8, 7.2), dpi=180)
    for label, logs in CURVES.items():
        rounds, accuracy = read_curve(*logs)
        if len(rounds) != 101 or rounds[0] != 0 or rounds[-1] != 100:
            raise RuntimeError(
                f"{label} is incomplete: {len(rounds)} points, "
                f"rounds {rounds[0]}-{rounds[-1]}"
            )
        ax.plot(rounds, accuracy * 100.0, label=label, **STYLES[label])

    ax.set_title("CIFAR-10 Non-IID ($\\alpha=0.1$), Sign Flipping Attack")
    ax.set_xlabel("Communication Round")
    ax.set_ylabel("Top-1 Accuracy (%)")
    ax.set_xlim(0, 100)
    ax.set_ylim(10, 48)
    ax.set_xticks(np.arange(0, 101, 10))
    ax.set_yticks(np.arange(10, 49, 5))
    ax.grid(True, color="#d9d9d9", linewidth=0.7, alpha=0.75)
    ax.legend(loc="upper left", ncol=2, fontsize=9.5, columnspacing=1.3)

    fig.tight_layout()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(OUTPUT.with_suffix(".pdf"), bbox_inches="tight")
    print(OUTPUT.with_suffix(".png"))
    print(OUTPUT.with_suffix(".pdf"))


if __name__ == "__main__":
    main()
