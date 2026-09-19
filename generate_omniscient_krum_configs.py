"""Generate isolated IID and Non-IID Krum-space omniscient experiments."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parent

FAMILIES = (
    {
        "source": ROOT / "configs" / "cfg_main_exp",
        "destination": ROOT / "configs" / "cfg_main_exp_omniscient_krum",
        "results_root": "./results/results_main_omniscient_krum",
    },
    {
        "source": ROOT / "configs" / "cfg_main_exp_noniid",
        "destination": ROOT / "configs" / "cfg_main_exp_noniid_omniscient_krum",
        "results_root": "./results/results_main_noniid_omniscient_krum",
    },
)

PASSTHROUGH = "fl_framework.components.attacks.passthrough.PassthroughAttack"
OMNISCIENT_KRUM = (
    "fl_framework.components.aggregators.omniscient_krum."
    "OmniscientKrumAggregator"
)
ADAMK_EF2 = (
    "fl_framework.components.aggregators.compress4_clip_ef2."
    "OmniscientEF2CompressAggregator"
)


def load(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def save(path: Path, config: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False, allow_unicode=True)


def configure(
    config: dict,
    attacked: bool,
    results_root: str,
    attack_scale: float = 1.0,
) -> dict:
    config = deepcopy(config)
    num_byzantine = int(config["byzantine"]["num_clients"])
    total_clients = int(config["honest"]["num_clients"]) + num_byzantine
    name = config["experiment"]["name"]

    config["byzantine"]["attack"] = {
        "class_path": PASSTHROUGH,
        "params": {},
    }

    aggregator_path = config["aggregator"]["class_path"]
    if "compress4_clip" in aggregator_path:
        params = dict(config["aggregator"].get("params", {}))
        params["k"] = 461
        params["byzantine_alpha"] = (
            num_byzantine / total_clients if total_clients else 0.0
        )
        params["attack_scale"] = attack_scale
        params["attack_enabled"] = attacked
        config["aggregator"] = {"class_path": ADAMK_EF2, "params": params}
    else:
        config["aggregator"] = {
            "class_path": OMNISCIENT_KRUM,
            "params": {
                "num_byzantine": num_byzantine,
                "attack_scale": attack_scale,
                "attack_enabled": attacked,
            },
        }

    # Source names end in either the attack name or withoutatt, optionally
    # followed by the Non-IID suffix. Callers set the final name explicitly.
    config["experiment"]["name"] = name
    config["experiment"]["save_path"] = f"{results_root}/{name}"
    return config


def generate_family(source: Path, destination: Path, results_root: str) -> int:
    destination.mkdir(parents=True, exist_ok=True)
    # This directory is generated exclusively by this script. Remove the
    # previous generated layout before replacing it with the corrected one.
    for old_config in destination.glob("*.yaml"):
        old_config.unlink()

    foe_files = sorted(source.glob("*foe*.yaml"))
    sign_files = sorted(source.glob("*signflipping*.yaml"))
    without_files = sorted(source.glob("*withoutatt*.yaml"))
    if not foe_files or not sign_files or not (
        len(foe_files) == len(sign_files) == len(without_files)
    ):
        raise RuntimeError(
            f"Expected paired foe/signflipping/withoutatt configs in {source}"
        )

    created = 0
    for source_file in foe_files:
        config = load(source_file)
        old_name = config["experiment"]["name"]
        new_name = old_name.replace("foe", "omniscient_foe")
        config["experiment"]["name"] = new_name
        config = configure(
            config,
            attacked=True,
            results_root=results_root,
            attack_scale=10.0,
        )
        config["experiment"]["name"] = new_name
        config["experiment"]["save_path"] = f"{results_root}/{new_name}"
        output_name = source_file.name.replace("foe", "omniscient_foe")
        save(destination / output_name, config)
        created += 1

    for source_file in sign_files:
        config = load(source_file)
        old_name = config["experiment"]["name"]
        new_name = old_name.replace(
            "signflipping", "omniscient_signflipping"
        )
        config["experiment"]["name"] = new_name
        config = configure(
            config,
            attacked=True,
            results_root=results_root,
            attack_scale=1.0,
        )
        config["experiment"]["name"] = new_name
        config["experiment"]["save_path"] = f"{results_root}/{new_name}"
        output_name = source_file.name.replace(
            "signflipping", "omniscient_signflipping"
        )
        save(destination / output_name, config)
        created += 1

    for source_file in without_files:
        config = load(source_file)
        name = config["experiment"]["name"]
        config = configure(config, attacked=False, results_root=results_root)
        config["experiment"]["name"] = name
        config["experiment"]["save_path"] = f"{results_root}/{name}"
        save(destination / source_file.name, config)
        created += 1

    return created


def main() -> None:
    total = 0
    for family in FAMILIES:
        count = generate_family(**family)
        total += count
        print(f"{family['destination']}: {count} configs")
    print(f"total: {total} configs")


if __name__ == "__main__":
    main()
