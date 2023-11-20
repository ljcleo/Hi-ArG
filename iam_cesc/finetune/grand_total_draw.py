from json import load
from pathlib import Path

import numpy as np
from matplotlib import colormaps
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.colors import Colormap
from matplotlib.figure import Figure

metrics: dict[str, list[float]] = {
    "macro_F1": [0.5, 0.7],
    "micro_F1": [0.8, 1.0],
    "support_F1": [0.3, 0.5],
    "contest_F1": [0.35, 0.55],
}

modes: dict[int, str] = {
    1: "text only",
    3: "joint & no freeze",
    4: "joint & half freeze",
    5: "joint & full freeze",
}

comments: tuple[str, ...] = ("(no pretrain)", "w/o comment", "with comment", "mixed comment")


if __name__ == "__main__":
    summary_dir: Path = Path("summary") / "grand_total"
    color_map: Colormap = colormaps["tab10"]

    for metric, y_lim in metrics.items():
        with (summary_dir / f"test_{metric}.json").open(encoding="utf8") as f:
            data: dict[str, dict[str, float]] = {
                k: v["per_seed_by_dev"] for k, v in load(f).items()
            }

        fig: Figure = plt.figure(figsize=(3 * len(modes), 6))
        ax: Axes = fig.subplots()

        if len(data) == 0:
            continue

        grouped: dict[int, list[dict[str, float]]] = {
            mode: [{} for _ in comments] for mode in modes.keys()
        }

        for name, value in data.items():
            components: list[str] = name.split("_")
            mode: int = int(components[0])
            comment: int = 0

            if len(components) > 1:
                comment = int(components[-1]) + 1

            grouped[mode][comment] = value

        x: np.ndarray = np.arange(len(modes))
        width: float = 0.2

        for i, comment in enumerate(comments):
            cur_x: np.ndarray = x + width * i

            cur_y: dict[str, np.ndarray] = {
                stat: np.array([grouped[mode][i].get(stat, 0) for mode in modes.keys()])
                for stat in ("mean", "min", "max")
            }

            cur_yerr: list[np.ndarray] = [
                cur_y["mean"] - cur_y["min"],
                cur_y["max"] - cur_y["mean"],
            ]

            ax.bar(
                cur_x,
                cur_y["mean"],
                width,
                label=comment,
                color=color_map((i + len(comment) - 1) % len(comment)),
            )

            ax.errorbar(cur_x, cur_y["mean"], yerr=cur_yerr, capsize=5, fmt="none", ecolor="black")

        ax.set_ylim(y_lim)
        ax.set_xticks(x + width * (len(comments) - 1) / 2, modes.values())
        ax.grid(which="both")
        ax.set_title(f"GreaseArg IAM CESC test {metric}")
        ax.set_ylabel(metric)
        ax.legend()

        fig.tight_layout()
        fig.savefig(summary_dir / f"test_{metric}.png", dpi=300)
        print(f"test_{metric}.png saved")
        plt.close(fig)
