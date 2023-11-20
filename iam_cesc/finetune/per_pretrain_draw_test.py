from json import load
from pathlib import Path

import numpy as np
from matplotlib import colormaps
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.colors import Colormap
from matplotlib.figure import Figure

metrics: dict[str, list[float]] = {
    "macro_F1": [0.3, 0.7],
    "micro_F1": [0.6, 1.0],
    "support_F1": [0.2, 0.6],
    "contest_F1": [0.2, 0.6],
}

modes: dict[int, str] = {
    1: "text only",
    3: "joint & no freeze",
    4: "joint & half freeze",
    5: "joint & full freeze",
}

comments: tuple[str, ...] = ("w/o comment", "with comment", "mixed comment")


if __name__ == "__main__":
    summary_dir: Path = Path("summary") / "per_pretrain"
    color_map: Colormap = colormaps["tab10"]

    for metric, y_lim in metrics.items():
        with (summary_dir / f"test_{metric}.json").open(encoding="utf8") as f:
            data: dict[str, dict[str, float]] = {
                k: v["per_seed_by_dev"] for k, v in load(f).items()
            }

        fig: Figure = plt.figure(figsize=(8, len(modes) * 3))
        axes: np.ndarray[Axes] = fig.subplots(len(modes), 1, sharex=True)

        grouped: dict[int, list[tuple[list[int], list[float], list[float]]]] = {
            mode: [([], [], []) for _ in comments] for mode in modes.keys()
        }

        for name, value in data.items():
            components: list[str] = name.split("_")
            comment: int = -1
            x: int = 0

            if len(components) > 1:
                comment = int(components[-2])
                x = int(components[-1])

            y_mean: float = value["mean"]
            y_max: float = value["max"]
            target: list[tuple[list[int], list[float], list[float]]] = grouped[int(components[0])]

            if comment == -1:
                for all_x, all_y_mean, all_y_max in target:
                    all_x.append(x)
                    all_y_mean.append(y_mean)
                    all_y_max.append(y_max)
            else:
                all_x, all_y_mean, all_y_max = target[comment]
                all_x.append(x)
                all_y_mean.append(y_mean)
                all_y_max.append(y_max)

        max_x: int = 0

        for (mode, mode_name), ax in zip(modes.items(), axes):
            ax: Axes

            for i, (comment, (x, y_mean, y_max)) in enumerate(zip(comments, grouped[mode])):
                if len(x) > 0:
                    max_x = max(max(x), max_x)

                for name, y, line_style in (("(mean)", y_mean, "-"), ("(max)", y_max, "--")):
                    ax.plot(
                        x,
                        y,
                        label=f"{comment} {name}",
                        marker="o",
                        linestyle=line_style,
                        color=color_map(i),
                    )

            ax.set_ylabel(metric)
            ax.set_xlim([-1, max_x + 1])
            ax.set_xticks(np.arange(0, max_x + 1, 2))
            ax.set_ylim(y_lim)
            ax.grid(which="both")
            ax.set_title(mode_name)

            if mode == list(modes.keys())[-1]:
                ax.set_xlabel("pretrain steps ($\\times10^4$)")
                ax.legend()

        fig.suptitle(f"GreaseArg IAM CESC test {metric}")
        fig.tight_layout()
        fig.savefig(summary_dir / f"test_{metric}.png", dpi=300)
        print(f"test_{metric}.png saved")
        plt.close(fig)
