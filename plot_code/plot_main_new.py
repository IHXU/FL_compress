"""Plot the CIFAR-10 IID experiments in results_main_new.

The plotting style and method ordering follow plot_main_noniid.py. Resumed log
files belonging to the same experiment are merged by communication round.
"""

from pathlib import Path

import plot_main_noniid as reference


ROOT = Path(__file__).resolve().parents[1] / "results" / "results_main_new"
OUTPUT_DIR = ROOT / "figures"
TITLE = "CIFAR-10, IID"
TAIL_WINDOW = 10

# results_main_new uses the same experiment names without the Non-IID suffix.
reference.DIR_PATTERN = reference.re.compile(
    r"^cifar10_(?P<method>.+)_(?P<attack>withoutatt|foe|labelflipping|signflipping)$"
)


def main() -> None:
    if not ROOT.is_dir():
        raise FileNotFoundError(f"Results directory does not exist: {ROOT}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    data = reference.load_results(ROOT)
    if not data:
        raise RuntimeError(f"No valid experiment logs found under {ROOT}")

    reference.plot_waterfall(data, OUTPUT_DIR / "main_new_comparison", TITLE)
    reference.plot_learning_curves(data, OUTPUT_DIR / "main_new_comparison_2d", TITLE)
    reference.plot_final_accuracy(
        data,
        OUTPUT_DIR / "main_new_final_accuracy",
        TAIL_WINDOW,
        f"{TITLE}: Final-Round Performance",
    )
    reference.write_summary(data, OUTPUT_DIR / "main_new_summary.csv", TAIL_WINDOW)


if __name__ == "__main__":
    main()
