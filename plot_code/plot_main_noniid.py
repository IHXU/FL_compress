"""Plot the main CIFAR-10 Non-IID comparison experiments.

The script reads accuracy directly from experiment logs, so it does not need to
load the (much larger) PyTorch checkpoints.  By default it writes a four-panel
learning-curve figure, a final-accuracy bar chart, and a CSV summary.
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.collections import PolyCollection
from matplotlib.lines import Line2D
import numpy as np


DEFAULT_ROOT = Path(__file__).resolve().parents[1] / "results" / "results_main_noniid"

ATTACK_ORDER = ["withoutatt", "foe", "labelflipping", "signflipping"]
ATTACK_DISPLAY = {
    "withoutatt": "No Attack",
    "foe": "FoE",
    "labelflipping": "Label Flipping",
    "signflipping": "Sign Flipping",
}

# Keep the proposed method first and use names consistent with the optimizer
# class names in configs/cfg_main_exp_noniid.
METHOD_ORDER = [
    "adamk_compress4_clip",
    "byz_dm21_krum",
    "byz_vr_dm21_krum",
    "byz_ef21_sgdm_krum",
    "byz_sgdm_broadcast_krum",
    "d_byz_sgdm_krum",
    "fed_dproc_krum",
    "rosdhb_krum",
]
METHOD_DISPLAY = {
    "adamk_compress4_clip": "ADAMK + Compress4-Clip",
    "byz_dm21_krum": "Byz-DM21 + Krum",
    "byz_vr_dm21_krum": "Byz-VR-DM21 + Krum",
    "byz_ef21_sgdm_krum": "Byz-EF21-SGDM + Krum",
    "byz_sgdm_broadcast_krum": "Byz-SGDM-Broadcast + Krum",
    "d_byz_sgdm_krum": "D-Byz-SGDM + Krum",
    "fed_dproc_krum": "Fed-DPRoC + Krum",
    "rosdhb_krum": "RoSDHB + Krum",
}
METHOD_SHORT = {
    "adamk_compress4_clip": "ADAMK-C4C",
    "byz_dm21_krum": "Byz-DM21",
    "byz_vr_dm21_krum": "Byz-VR-DM21",
    "byz_ef21_sgdm_krum": "Byz-EF21-SGDM",
    "byz_sgdm_broadcast_krum": "Byz-SGDM-BC",
    "d_byz_sgdm_krum": "D-Byz-SGDM",
    "fed_dproc_krum": "Fed-DPRoC",
    "rosdhb_krum": "RoSDHB",
}

# Plasma-like colors match the existing figures while remaining distinguishable.
COLORS = [
    "#0d0887",
    "#5302a3",
    "#8b0aa5",
    "#b83289",
    "#db5c68",
    "#f48849",
    "#fdb42f",
    "#f0f921",
]
LINESTYLES = ["-", "--", "-.", ":", "--", "-.", ":", "--"]
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "h"]

DIR_PATTERN = re.compile(
    r"^cifar10_(?P<method>.+)_(?P<attack>withoutatt|foe|labelflipping|signflipping)"
    r"_noniid_alpha01$"
)
METRIC_PATTERN = re.compile(
    r"Round\s+(?P<round>\d+)\s+Metric\s+top1_accuracy:\s*"
    r"(?P<accuracy>[-+0-9.eE]+)"
)


def parse_log(log_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return sorted, de-duplicated rounds and top-1 accuracies."""
    values: dict[int, float] = {}
    with log_path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            match = METRIC_PATTERN.search(line)
            if match:
                values[int(match.group("round"))] = float(match.group("accuracy"))
    rounds = np.asarray(sorted(values), dtype=int)
    accuracies = np.asarray([values[r] for r in rounds], dtype=float)
    return rounds, accuracies


def best_log(folder: Path) -> tuple[str | None, np.ndarray, np.ndarray]:
    """Merge resumed logs, with newer logs taking precedence for duplicate rounds."""
    log_paths = sorted(folder.glob("experiment_*.log"), key=lambda path: path.stat().st_mtime)
    if not log_paths:
        empty = np.asarray([], dtype=float)
        return None, empty.astype(int), empty

    values: dict[int, float] = {}
    contributing_logs = []
    for log_path in log_paths:
        rounds, accuracies = parse_log(log_path)
        if len(rounds):
            contributing_logs.append(log_path)
            values.update(zip(rounds.tolist(), accuracies.tolist()))

    rounds = np.asarray(sorted(values), dtype=int)
    accuracies = np.asarray([values[round] for round in rounds], dtype=float)
    logs = ";".join(str(path) for path in contributing_logs)
    return logs or None, rounds, accuracies


def load_results(root: Path) -> dict[str, dict[str, dict[str, object]]]:
    data: dict[str, dict[str, dict[str, object]]] = {}
    for folder in sorted(root.iterdir()):
        if not folder.is_dir():
            continue
        match = DIR_PATTERN.match(folder.name)
        if not match:
            if folder.name != "figures":
                print(f"[WARN] Skipping unrecognized directory: {folder.name}")
            continue
        method = match.group("method")
        attack = match.group("attack")
        log_path, rounds, accuracies = best_log(folder)
        if log_path is None or not len(rounds):
            print(f"[WARN] No accuracy data: {folder.name}")
            continue
        data.setdefault(attack, {})[method] = {
            "rounds": rounds,
            "accuracies": accuracies,
            "log": log_path,
        }
        print(
            f"[INFO] Loaded {method:28s} | {attack:14s} | "
            f"{len(rounds):3d} points | last round {rounds[-1]:3d}"
        )
    return data


def apply_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.labelsize": 11,
            "axes.titlesize": 13,
            "axes.titleweight": "bold",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "savefig.dpi": 300,
        }
    )


def _style_3d_axis(ax) -> None:
    """Match the 3-D styling used by the existing ablation plots."""
    ax.computed_zorder = False
    ax.view_init(elev=25, azim=-55)
    ax.set_box_aspect((1.35, 1.0, 1.0))
    ax.xaxis.pane.set_facecolor((0.96, 0.97, 1.00, 0.42))
    ax.yaxis.pane.set_facecolor((0.95, 1.00, 0.97, 0.34))
    ax.zaxis.pane.set_facecolor((1.00, 0.98, 0.91, 0.28))
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_edgecolor((0.78, 0.80, 0.86, 0.45))
        axis._axinfo["grid"]["color"] = (0.62, 0.66, 0.74, 0.28)
        axis._axinfo["grid"]["linewidth"] = 0.55
    ax.tick_params(labelsize=7.4, pad=0)


def _plot_attack_waterfall(ax, attack_data: dict) -> None:
    methods = [method for method in METHOD_ORDER if method in attack_data]
    if not methods:
        return

    all_acc = np.concatenate(
        [np.asarray(attack_data[method]["accuracies"], dtype=float) for method in methods]
    )
    z_floor = max(0.0, float(np.nanmin(all_acc)) - 0.08)
    z_top = min(1.0, float(np.nanmax(all_acc)) + 0.08)
    max_round = max(int(attack_data[method]["rounds"][-1]) for method in methods)
    y_positions = np.arange(len(methods), dtype=float)

    curve_rows = []
    for y_pos, method in zip(y_positions, methods):
        entry = attack_data[method]
        rounds = np.asarray(entry["rounds"], dtype=float)
        acc = np.asarray(entry["accuracies"], dtype=float)
        color = COLORS[METHOD_ORDER.index(method)]
        curve_rows.append((y_pos, method, rounds, acc, color))

    connector_rounds = np.unique(np.r_[np.arange(0, max_round + 1, 20), max_round]).astype(int)
    for round_value in connector_rounds:
        points = []
        for y_pos, _, rounds, acc, _ in curve_rows:
            indices = np.flatnonzero(rounds == round_value)
            if len(indices):
                points.append((y_pos, float(acc[indices[0]])))
        if len(points) >= 2:
            ys, zs = zip(*points)
            ax.plot(
                np.full(len(ys), round_value), ys, zs,
                color="black", linewidth=0.65, linestyle="--",
                alpha=0.36, zorder=6,
            )

    for y_pos, method, rounds, acc, color in reversed(curve_rows):
        row_zorder = 10 + (len(curve_rows) - y_pos) * 3
        verts = [[(rounds[0], z_floor), *zip(rounds, acc), (rounds[-1], z_floor)]]
        poly = PolyCollection(
            verts,
            facecolors=[colors.to_rgba(color, 0.28)],
            edgecolors=[colors.to_rgba(color, 0.74)],
            linewidths=0.55,
            zorder=row_zorder,
        )
        ax.add_collection3d(poly, zs=[y_pos], zdir="y")
        ax.plot(
            rounds, np.full_like(rounds, y_pos), acc,
            color=color,
            linewidth=1.35 if method == METHOD_ORDER[0] else 1.05,
            alpha=0.98,
            zorder=row_zorder + 1,
        )

        point_x, point_z = [], []
        for round_value in connector_rounds:
            indices = np.flatnonzero(rounds == round_value)
            if len(indices):
                point_x.append(round_value)
                point_z.append(float(acc[indices[0]]))
        if point_x:
            ax.scatter(
                point_x, np.full(len(point_x), y_pos), point_z,
                color=color, edgecolors="none", marker="o", s=20,
                depthshade=False, zorder=row_zorder + 1.2,
            )

    ax.set_xlim(0, max_round)
    ax.set_ylim(-0.45, len(methods) - 0.55)
    ax.set_zlim(z_floor, z_top)
    ax.set_yticks(y_positions)
    ax.set_yticklabels([METHOD_SHORT[method] for method in methods])
    ax.set_xlabel("Round", fontsize=9.5, labelpad=2)
    ax.set_ylabel("Method", fontsize=9.5, labelpad=3)
    ax.set_zlabel("Accuracy", fontsize=9.5, labelpad=3)
    _style_3d_axis(ax)

    legend_handles = [
        Line2D(
            [0], [0], marker="o", linestyle="None",
            markerfacecolor=COLORS[METHOD_ORDER.index(method)],
            markeredgecolor="none", markersize=5.5,
            label=METHOD_SHORT[method],
        )
        for method in methods
    ]
    ax.legend(
        handles=legend_handles,
        loc="upper left",
        bbox_to_anchor=(0.01, 0.99),
        bbox_transform=ax.transAxes,
        frameon=False,
        fontsize=6.8,
        handlelength=0.7,
        handletextpad=0.3,
        borderaxespad=0.0,
        labelspacing=0.25,
    )


def plot_waterfall(
    data: dict,
    output_base: Path,
    title: str = r"CIFAR-10, Non-IID ($\alpha=0.1$)",
) -> None:
    """Create the 1x4 waterfall layout used by the reference scripts."""
    apply_style()
    fig = plt.figure(figsize=(26, 7.2), dpi=300)
    axes = [fig.add_subplot(1, 4, index + 1, projection="3d") for index in range(4)]

    for index, (ax, attack) in enumerate(zip(axes, ATTACK_ORDER)):
        attack_data = data.get(attack, {})
        if attack_data:
            _plot_attack_waterfall(ax, attack_data)
            if index == len(axes) - 1:
                ax.set_zlabel("")
                ax.text2D(
                    1.035, 0.52, "Accuracy", transform=ax.transAxes,
                    rotation=90, ha="center", va="center", fontsize=9.5,
                )
        else:
            ax.text2D(0.34, 0.5, "No data", transform=ax.transAxes, fontsize=12)
            ax.set_axis_off()
        ax.text2D(
            0.5, -0.025, ATTACK_DISPLAY[attack], transform=ax.transAxes,
            ha="center", va="top", fontsize=15, weight="bold",
        )

    fig.suptitle(title, y=0.975, fontsize=15, weight="bold")
    fig.subplots_adjust(left=0.015, right=0.95, bottom=0.07, top=0.93, wspace=0.08)
    for suffix in ("png", "pdf"):
        path = output_base.with_suffix(f".{suffix}")
        fig.savefig(path, bbox_inches="tight", pad_inches=0.08)
        print(f"[INFO] Saved {path}")
    plt.close(fig)


def plot_learning_curves(
    data: dict,
    output_base: Path,
    title: str = r"CIFAR-10, Non-IID ($\alpha=0.1$)",
) -> None:
    apply_style()
    fig, axes = plt.subplots(2, 2, figsize=(14.2, 9.0), sharex=True, sharey=True)
    handles_by_method = {}

    all_accuracies = [
        entry["accuracies"]
        for attack_data in data.values()
        for entry in attack_data.values()
    ]
    y_min = max(0.0, min(float(np.nanmin(values)) for values in all_accuracies) - 0.035)
    y_max = min(1.0, max(float(np.nanmax(values)) for values in all_accuracies) + 0.035)

    for panel_index, (ax, attack) in enumerate(zip(axes.flat, ATTACK_ORDER)):
        attack_data = data.get(attack, {})
        for method_index, method in enumerate(METHOD_ORDER):
            if method not in attack_data:
                continue
            entry = attack_data[method]
            is_primary = method == METHOD_ORDER[0]
            (line,) = ax.plot(
                entry["rounds"],
                entry["accuracies"],
                color=COLORS[method_index],
                linestyle=LINESTYLES[method_index],
                linewidth=2.6 if is_primary else 1.65,
                marker=MARKERS[method_index],
                markersize=4.7 if is_primary else 3.8,
                markevery=10,
                markeredgecolor="white",
                markeredgewidth=0.45,
                alpha=1.0 if is_primary else 0.92,
                label=METHOD_DISPLAY.get(method, method),
                zorder=10 if is_primary else 3,
            )
            handles_by_method[method] = line

        ax.set_title(f"({chr(97 + panel_index)}) {ATTACK_DISPLAY[attack]}", pad=8)
        ax.set_xlim(left=0)
        ax.set_ylim(y_min, y_max)
        ax.grid(True, linestyle="--", linewidth=0.65, color="#aeb4bd", alpha=0.48)
        ax.set_axisbelow(True)

    for ax in axes[-1, :]:
        ax.set_xlabel("Communication Round")
    for ax in axes[:, 0]:
        ax.set_ylabel("Top-1 Accuracy")

    ordered_methods = [method for method in METHOD_ORDER if method in handles_by_method]
    fig.legend(
        [handles_by_method[method] for method in ordered_methods],
        [METHOD_DISPLAY.get(method, method) for method in ordered_methods],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=4,
        fontsize=9.4,
        columnspacing=1.4,
        handlelength=3.2,
    )
    fig.suptitle(title, y=0.925, fontsize=14, weight="bold")
    fig.subplots_adjust(left=0.07, right=0.985, bottom=0.075, top=0.865, hspace=0.27, wspace=0.12)
    for suffix in ("png", "pdf"):
        path = output_base.with_suffix(f".{suffix}")
        fig.savefig(path, bbox_inches="tight", pad_inches=0.06)
        print(f"[INFO] Saved {path}")
    plt.close(fig)


def tail_mean(entry: dict, window: int) -> float:
    values = np.asarray(entry["accuracies"], dtype=float)
    return float(np.nanmean(values[-window:]))


def mean_ending_at(entry: dict, end_round: int, window: int) -> float:
    rounds = np.asarray(entry["rounds"], dtype=int)
    values = np.asarray(entry["accuracies"], dtype=float)
    selected = values[(rounds <= end_round) & (rounds > end_round - window)]
    return float(np.nanmean(selected)) if len(selected) else np.nan


def plot_final_accuracy(
    data: dict,
    output_base: Path,
    window: int,
    title: str = "CIFAR-10, Non-IID (α=0.1): Common-Round Performance",
) -> None:
    apply_style()
    methods = [method for method in METHOD_ORDER if any(method in data.get(a, {}) for a in ATTACK_ORDER)]
    x = np.arange(len(ATTACK_ORDER), dtype=float)
    width = 0.82 / max(len(methods), 1)
    fig, ax = plt.subplots(figsize=(14.2, 5.4))

    # Compare every method at the same endpoint while some runs are incomplete.
    common_end = min(
        int(data[attack][method]["rounds"][-1])
        for attack in ATTACK_ORDER
        for method in methods
        if method in data.get(attack, {})
    )

    for method_index, method in enumerate(methods):
        heights = [
            mean_ending_at(data[attack][method], common_end, window)
            if method in data.get(attack, {})
            else np.nan
            for attack in ATTACK_ORDER
        ]
        offset = (method_index - (len(methods) - 1) / 2) * width
        ax.bar(
            x + offset,
            heights,
            width=width * 0.92,
            color=COLORS[METHOD_ORDER.index(method)],
            edgecolor="white",
            linewidth=0.45,
            label=METHOD_DISPLAY.get(method, method),
        )

    ax.set_xticks(x, [ATTACK_DISPLAY[attack] for attack in ATTACK_ORDER])
    ax.set_ylabel(f"Mean Top-1 Accuracy (rounds {max(0, common_end - window + 1)}–{common_end})")
    ax.set_title(title, pad=12)
    ax.grid(axis="y", linestyle="--", linewidth=0.65, color="#aeb4bd", alpha=0.5)
    ax.set_axisbelow(True)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=4, fontsize=9)
    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.25, top=0.91)
    for suffix in ("png", "pdf"):
        path = output_base.with_suffix(f".{suffix}")
        fig.savefig(path, bbox_inches="tight", pad_inches=0.06)
        print(f"[INFO] Saved {path}")
    plt.close(fig)


def write_summary(data: dict, output_path: Path, window: int) -> None:
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["attack", "method", "points", "last_round", "last_accuracy", f"mean_last_{window}", "log"]
        )
        for attack in ATTACK_ORDER:
            for method in METHOD_ORDER:
                if method not in data.get(attack, {}):
                    continue
                entry = data[attack][method]
                writer.writerow(
                    [
                        attack,
                        method,
                        len(entry["rounds"]),
                        int(entry["rounds"][-1]),
                        f"{entry['accuracies'][-1]:.6f}",
                        f"{tail_mean(entry, window):.6f}",
                        entry["log"],
                    ]
                )
    print(f"[INFO] Saved {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT, help="results_main_noniid directory")
    parser.add_argument("--output-dir", type=Path, default=None, help="figure output directory")
    parser.add_argument("--tail-window", type=int, default=10, help="rounds averaged in final summary")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    output_dir = (args.output_dir or (root / "figures")).resolve()
    if args.tail_window < 1:
        raise ValueError("--tail-window must be at least 1")
    if not root.is_dir():
        raise FileNotFoundError(f"Results directory does not exist: {root}")
    output_dir.mkdir(parents=True, exist_ok=True)

    data = load_results(root)
    if not data:
        raise RuntimeError(f"No valid experiment logs found under {root}")
    plot_waterfall(data, output_dir / "main_noniid_comparison")
    plot_learning_curves(data, output_dir / "main_noniid_comparison_2d")
    plot_final_accuracy(data, output_dir / "main_noniid_final_accuracy", args.tail_window)
    write_summary(data, output_dir / "main_noniid_summary.csv", args.tail_window)


if __name__ == "__main__":
    main()
