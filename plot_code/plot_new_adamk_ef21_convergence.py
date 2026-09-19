"""Compare the completed EF/EF21 runs in cfg_adamk_ef_variants_noniid."""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = ROOT / "results" / "results_adamk_ef_variants_noniid"
OUTPUT = RESULT_ROOT / "adamk_ef21_convergence_comparison"

LOSS_PATTERN = re.compile(
    r"Round\s+(?P<round>\d+)\s+Loss:\s*(?P<value>[-+0-9.eE]+)"
)
ACCURACY_PATTERN = re.compile(
    r"Round\s+(?P<round>\d+)\s+Metric\s+top1_accuracy:\s*"
    r"(?P<value>[-+0-9.eE]+)"
)

SCENARIOS = (
    ("withoutatt", "withoutatt", "No Attack"),
    ("omniscient_foe", "foe", "FoE"),
    ("omniscient_signflipping", "signflipping", "Sign Flipping"),
)

METHODS = (
    (
        "V1: EF, no WD",
        "cifar10_adamk_v1_ef_nowd_{old_attack}_noniid_alpha01",
        dict(color="#4c78a8", linewidth=1.9, linestyle="-"),
    ),
    (
        "V2: EF, no WD/ARC",
        "cifar10_adamk_v2_ef_nowd_noclip_{old_attack}_noniid_alpha01",
        dict(color="#59a14f", linewidth=1.9, linestyle="--"),
    ),
    (
        r"V3: EF + client momentum ($\beta_1=0$)",
        "cifar10_adamk_v3_ef_clientmom_b10_{old_attack}_noniid_alpha01",
        dict(color="#e15759", linewidth=1.9, linestyle="-."),
    ),
    (
        "V4: EF + active-mask update",
        "cifar10_adamk_v4_ef_masked_active_{old_attack}_noniid_alpha01",
        dict(color="#f28e2b", linewidth=1.9, linestyle=":"),
    ),
    (
        "New: RP-EF21-Adam",
        "cifar10_new_adamk_ef21_{new_attack}_noniid_alpha01",
        dict(color="#111111", linewidth=3.0, linestyle="-"),
    ),
)


def read_history(folder: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    losses: dict[int, float] = {}
    accuracies: dict[int, float] = {}
    logs = sorted(folder.glob("experiment_*.log"), key=lambda path: path.stat().st_mtime)
    for log in logs:
        with log.open("r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                loss_match = LOSS_PATTERN.search(line)
                if loss_match:
                    losses[int(loss_match.group("round"))] = float(
                        loss_match.group("value")
                    )
                accuracy_match = ACCURACY_PATTERN.search(line)
                if accuracy_match:
                    accuracies[int(accuracy_match.group("round"))] = float(
                        accuracy_match.group("value")
                    )

    rounds = np.asarray(sorted(set(losses) & set(accuracies)), dtype=int)
    if len(rounds) != 101 or rounds[0] != 0 or rounds[-1] != 100:
        raise RuntimeError(
            f"Incomplete history in {folder}: {len(rounds)} shared points"
        )
    loss = np.asarray([losses[round_] for round_ in rounds], dtype=float)
    accuracy = np.asarray(
        [accuracies[round_] * 100.0 for round_ in rounds], dtype=float
    )
    return rounds, loss, accuracy


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

    fig, axes = plt.subplots(2, 3, figsize=(18.5, 10.2), dpi=180, sharex=True)
    handles = labels = None
    for column, (old_attack, new_attack, title) in enumerate(SCENARIOS):
        loss_axis = axes[0, column]
        accuracy_axis = axes[1, column]
        for label, folder_template, style in METHODS:
            folder = RESULT_ROOT / folder_template.format(
                old_attack=old_attack, new_attack=new_attack
            )
            rounds, loss, accuracy = read_history(folder)
            loss_axis.plot(rounds, loss, label=label, **style)
            accuracy_axis.plot(rounds, accuracy, label=label, **style)

        loss_axis.set_title(title)
        loss_axis.grid(True, color="#d9d9d9", linewidth=0.7, alpha=0.75)
        accuracy_axis.grid(True, color="#d9d9d9", linewidth=0.7, alpha=0.75)
        accuracy_axis.set_xlabel("Communication Round")
        accuracy_axis.set_xlim(0, 100)
        accuracy_axis.set_xticks(np.arange(0, 101, 20))
        accuracy_axis.set_ylim(8, 46)
        accuracy_axis.set_yticks(np.arange(10, 46, 5))
        if handles is None:
            handles, labels = loss_axis.get_legend_handles_labels()

    axes[0, 0].set_ylabel("Test Loss")
    axes[1, 0].set_ylabel("Top-1 Accuracy (%)")
    fig.suptitle(
        r"CIFAR-10 Non-IID ($\alpha=0.1$): Existing EF Variants vs. New EF21",
        fontsize=16,
        fontweight="bold",
        y=0.985,
    )
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.012),
        ncol=3,
        fontsize=10.5,
        columnspacing=1.7,
        handlelength=3.2,
    )
    fig.text(
        0.5,
        0.065,
        "Curves use each YAML configuration's recorded attack implementation; "
        "all runs contain 101 evaluation points.",
        ha="center",
        fontsize=9.5,
        color="#555555",
    )
    fig.tight_layout(rect=(0.02, 0.105, 0.99, 0.95), h_pad=2.2, w_pad=1.5)
    fig.savefig(OUTPUT.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(OUTPUT.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(OUTPUT.with_suffix(".png"))
    print(OUTPUT.with_suffix(".pdf"))


if __name__ == "__main__":
    main()
