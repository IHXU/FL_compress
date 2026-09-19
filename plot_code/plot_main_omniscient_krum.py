"""Plot the IID and Non-IID omniscient-Krum main experiments.

The omniscient experiment directories contain No Attack, omniscient FoE, and
omniscient Sign Flipping.  Label Flipping is loaded from the corresponding
results_main_new/results_main_noniid directory to complete the four panels.
Resumed logs belonging to one method/attack are merged by communication round.
"""

from __future__ import annotations

import re
from pathlib import Path

import plot_main_noniid as reference


RESULTS_ROOT = Path(__file__).resolve().parents[1] / "results"
TAIL_WINDOW = 10

EXPERIMENTS = (
    {
        "root": RESULTS_ROOT / "results_main_omniscient_krum",
        "label_root": RESULTS_ROOT / "results_main_new",
        "suffix": "",
        "stem": "main_omniscient_krum",
        "title": "CIFAR-10, IID: Omniscient Krum Attacks",
    },
    {
        "root": RESULTS_ROOT / "results_main_noniid_omniscient_krum",
        "label_root": RESULTS_ROOT / "results_main_noniid",
        "suffix": "_noniid_alpha01",
        "stem": "main_noniid_omniscient_krum",
        "title": r"CIFAR-10, Non-IID ($\alpha=0.1$): Omniscient Krum Attacks",
    },
)

ATTACK_MAP = {
    "withoutatt": "withoutatt",
    "omniscient_foe": "foe",
    "omniscient_signflipping": "signflipping",
    "labelflipping": "labelflipping",
}


def load_matching(root: Path, pattern: re.Pattern[str]) -> dict:
    """Load folders selected by *pattern* using the reference log parser."""
    data: dict[str, dict[str, dict[str, object]]] = {}
    for folder in sorted(root.iterdir()):
        if not folder.is_dir() or folder.name == "figures":
            continue
        match = pattern.fullmatch(folder.name)
        if not match:
            continue

        method = match.group("method")
        attack_token = match.group("attack")
        attack = ATTACK_MAP[attack_token]
        logs, rounds, accuracies = reference.best_log(folder)
        if logs is None or not len(rounds):
            print(f"[WARN] No accuracy data: {folder}")
            continue
        data.setdefault(attack, {})[method] = {
            "rounds": rounds,
            "accuracies": accuracies,
            "log": logs,
        }
        print(
            f"[INFO] Loaded {method:28s} | {attack:14s} | "
            f"{len(rounds):3d} points | last round {rounds[-1]:3d} | {root.name}"
        )
    return data


def load_experiment(root: Path, label_root: Path, suffix: str) -> dict:
    if not root.is_dir():
        raise FileNotFoundError(f"Results directory does not exist: {root}")
    if not label_root.is_dir():
        raise FileNotFoundError(f"Label-Flipping source does not exist: {label_root}")

    escaped_suffix = re.escape(suffix)
    main_pattern = re.compile(
        r"^cifar10_(?P<method>.+)_"
        r"(?P<attack>withoutatt|omniscient_foe|omniscient_signflipping)"
        rf"{escaped_suffix}$"
    )
    label_pattern = re.compile(
        r"^cifar10_(?P<method>.+)_(?P<attack>labelflipping)"
        rf"{escaped_suffix}$"
    )

    data = load_matching(root, main_pattern)
    label_data = load_matching(label_root, label_pattern)
    data["labelflipping"] = label_data.get("labelflipping", {})

    missing = {
        attack: [method for method in reference.METHOD_ORDER if method not in data.get(attack, {})]
        for attack in reference.ATTACK_ORDER
    }
    missing = {attack: methods for attack, methods in missing.items() if methods}
    if missing:
        details = "; ".join(f"{attack}: {', '.join(methods)}" for attack, methods in missing.items())
        raise RuntimeError(f"Incomplete plotting data ({root.name}): {details}")
    return data


def plot_experiment(config: dict) -> None:
    root = config["root"]
    output_dir = root / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    data = load_experiment(root, config["label_root"], config["suffix"])
    stem = config["stem"]
    title = config["title"]

    reference.plot_waterfall(data, output_dir / f"{stem}_comparison", title)
    reference.plot_learning_curves(data, output_dir / f"{stem}_comparison_2d", title)
    reference.plot_final_accuracy(
        data,
        output_dir / f"{stem}_final_accuracy",
        TAIL_WINDOW,
        f"{title}: Common-Round Performance",
    )
    reference.write_summary(data, output_dir / f"{stem}_summary.csv", TAIL_WINDOW)


def main() -> None:
    # Make the two omniscient attack panels explicit while retaining the shared
    # reference plotting layout and method styling.
    reference.ATTACK_DISPLAY["foe"] = "Omniscient FoE"
    reference.ATTACK_DISPLAY["signflipping"] = "Omniscient Sign Flipping"
    for config in EXPERIMENTS:
        plot_experiment(config)


if __name__ == "__main__":
    main()
