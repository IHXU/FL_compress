"""Compare EF+No-WD AdamK with every non-AdamK non-IID baseline."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from plot_adamk_ef_variants_comparison import read_curve


ROOT = Path(__file__).resolve().parents[1]
VARIANT_ROOT = ROOT / "results" / "results_adamk_ef_variants_noniid"
BASELINE_ROOT = ROOT / "results" / "results_main_noniid_omniscient_krum"
OUTPUT = VARIANT_ROOT / "ef_nowd_vs_all_noniid_curves_and_final_acc"

ATTACKS = (
    ("withoutatt", "No Attack"),
    ("omniscient_foe", "Omniscient FoE"),
    ("omniscient_signflipping", "Omniscient Sign Flipping"),
)

# Original AdamK is intentionally excluded.
METHODS = (
    (
        "AdamK V1: EF, No WD",
        VARIANT_ROOT,
        "cifar10_adamk_v1_ef_nowd_{attack}_noniid_alpha01",
    ),
    (
        "Byz-DM21 + Krum",
        BASELINE_ROOT,
        "cifar10_byz_dm21_krum_{attack}_noniid_alpha01",
    ),
    (
        "Byz-VR-DM21 + Krum",
        BASELINE_ROOT,
        "cifar10_byz_vr_dm21_krum_{attack}_noniid_alpha01",
    ),
    (
        "Byz-EF21-SGDM + Krum",
        BASELINE_ROOT,
        "cifar10_byz_ef21_sgdm_krum_{attack}_noniid_alpha01",
    ),
    (
        "Byz-SGDM-Broadcast + Krum",
        BASELINE_ROOT,
        "cifar10_byz_sgdm_broadcast_krum_{attack}_noniid_alpha01",
    ),
    (
        "D-Byz-SGDM + Krum",
        BASELINE_ROOT,
        "cifar10_d_byz_sgdm_krum_{attack}_noniid_alpha01",
    ),
    (
        "Fed-DPRoC + Krum",
        BASELINE_ROOT,
        "cifar10_fed_dproc_krum_{attack}_noniid_alpha01",
    ),
    (
        "RoSDHB + Krum",
        BASELINE_ROOT,
        "cifar10_rosdhb_krum_{attack}_noniid_alpha01",
    ),
)

COLORS = (
    "#d62728",
    "#1f77b4",
    "#2ca02c",
    "#ff7f0e",
    "#9467bd",
    "#17becf",
    "#e377c2",
    "#8c564b",
)
LINESTYLES = ("-", "--", "-.", ":", "--", "-.", "--", ":")


def main() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 12.5,
            "axes.titleweight": "bold",
            "axes.labelsize": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
        }
    )

    curves: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]] = {}
    for attack, _ in ATTACKS:
        for label, root, template in METHODS:
            folder = root / template.format(attack=attack)
            curves[(attack, label)] = read_curve(folder)

    fig, axes = plt.subplots(
        2,
        3,
        figsize=(20, 11.5),
        dpi=180,
        gridspec_kw={"height_ratios": (1.25, 1.0), "hspace": 0.27},
    )

    handles = None
    labels = None
    for column, (attack, title) in enumerate(ATTACKS):
        curve_ax = axes[0, column]
        bar_ax = axes[1, column]

        final_values = []
        for index, (label, _, _) in enumerate(METHODS):
            rounds, accuracy = curves[(attack, label)]
            accuracy_pct = accuracy * 100.0
            final_values.append(accuracy_pct[-1])
            is_adamk = index == 0
            curve_ax.plot(
                rounds,
                accuracy_pct,
                label=label,
                color=COLORS[index],
                linestyle=LINESTYLES[index],
                linewidth=3.0 if is_adamk else 1.9,
                zorder=5 if is_adamk else 2,
            )

        curve_ax.set_title(title)
        curve_ax.set_xlabel("Communication Round")
        curve_ax.set_xlim(0, 100)
        curve_ax.set_ylim(10, 50)
        curve_ax.set_xticks(np.arange(0, 101, 20))
        curve_ax.set_yticks(np.arange(10, 51, 5))
        curve_ax.grid(True, color="#d9d9d9", linewidth=0.7, alpha=0.8)
        curve_ax.set_axisbelow(True)
        if handles is None:
            handles, labels = curve_ax.get_legend_handles_labels()

        positions = np.arange(len(METHODS))
        bars = bar_ax.bar(
            positions,
            final_values,
            color=COLORS,
            edgecolor="white",
            linewidth=0.8,
        )
        bars[0].set_edgecolor("#8b0000")
        bars[0].set_linewidth(2.0)
        bar_ax.bar_label(
            bars,
            labels=[f"{value:.1f}" for value in final_values],
            padding=2,
            fontsize=8.5,
        )
        bar_ax.set_title(f"{title}: Final Accuracy (Round 100)")
        bar_ax.set_xticks(positions, [str(i + 1) for i in positions])
        bar_ax.set_xlabel("Method number (see legend)")
        bar_ax.set_ylim(0, 52)
        bar_ax.set_yticks(np.arange(0, 51, 5))
        bar_ax.grid(axis="y", color="#d9d9d9", linewidth=0.7, alpha=0.8)
        bar_ax.set_axisbelow(True)

    axes[0, 0].set_ylabel("Top-1 Accuracy (%)")
    axes[1, 0].set_ylabel("Final Top-1 Accuracy (%)")
    fig.suptitle(
        r"CIFAR-10 Non-IID ($\alpha=0.1$): AdamK EF No-WD vs. All Baselines",
        fontsize=17,
        fontweight="bold",
        y=0.985,
    )
    numbered_labels = [f"{index + 1}. {label}" for index, label in enumerate(labels)]
    fig.legend(
        handles,
        numbered_labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.012),
        ncol=4,
        fontsize=9.7,
        columnspacing=1.5,
        handlelength=3.0,
    )
    fig.subplots_adjust(left=0.055, right=0.99, top=0.92, bottom=0.125)
    fig.savefig(OUTPUT.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(OUTPUT.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(OUTPUT.with_suffix(".png"))
    print(OUTPUT.with_suffix(".pdf"))


if __name__ == "__main__":
    main()
