from json import load
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.figure import Figure

metrics: dict[str, list[float]] = {
    "strict_mAP": [0.4, 0.8],
    "relaxed_mAP": [0.6, 1.0],
    "mean_mAP": [0.5, 0.9],
    "strict_p@50%": [0.5, 0.9],
}

formats: tuple[str, ...] = ("sc", "pl", "cl")

modes: dict[int, str] = {
    1: "text only",
    3: "joint & no freeze",
    4: "joint & half freeze",
    5: "joint & full freeze",
}

comments: tuple[str, ...] = ("w/o comment", "with comment", "mixed comment")


def extract(
    name: str, content: dict[str, dict[str, float]]
) -> tuple[str, int, int, np.ndarray, np.ndarray, list[np.ndarray]]:
    keys: list[str] = sorted(content.keys())
    x: np.ndarray = np.array([int(x) for x in keys])
    x //= x[0]

    y: dict[str, np.ndarray] = {
        stat: np.array([content[key][stat] for key in keys]) for stat in ("mean", "min", "max")
    }

    e_min: np.ndarray = y["mean"] - y["min"]
    e_max: np.ndarray = y["max"] - y["mean"]

    comment: int = -1
    checkpoint: int = 0

    if name.count("_") > 1:
        comment, checkpoint = [int(s) for s in name.rsplit("_", 2)[1:]]

    return name, comment, checkpoint, x, y["mean"], [e_min, e_max]


if __name__ == "__main__":
    summary_dir: Path = Path("summary") / "per_pretrain"

    for metric, y_lim in metrics.items():
        with (summary_dir / f"dev_{metric}.json").open(encoding="utf8") as f:
            data: dict[str, dict[str, dict[str, float]]] = load(f)

        mappable = ScalarMappable(
            norm=Normalize(
                vmin=0,
                vmax=max(
                    [int(key.rsplit("_", 2)[-1]) for key in data.keys() if key.count("_") > 1]
                ),
            ),
            cmap="cool",
        )

        for format in formats:
            fig: Figure = plt.figure(figsize=(len(comments) << 2 | 2, len(modes) * 3))
            max_x: bool = 0

            axes: np.ndarray[Axes] = fig.subplots(
                len(modes), len(comments), sharex=True, sharey=True
            )

            for pos, mode in enumerate(modes.keys()):
                sub_data: list[
                    tuple[str, int, int, np.ndarray, np.ndarray, list[np.ndarray]]
                ] = sorted(
                    [
                        extract(name, content)
                        for name, content in data.items()
                        if name[0] == str(mode) and name[2:4] == format
                    ],
                    key=lambda x: x[0],
                )

                if len(sub_data) == 0:
                    continue

                for detail in sub_data:
                    max_x = max(detail[3].max().item(), max_x)

                    def draw(ax: Axes) -> None:
                        ax.errorbar(
                            detail[3],
                            detail[4],
                            yerr=detail[5],
                            label=detail[0],
                            capsize=5,
                            marker="o",
                            color=mappable.to_rgba(detail[2]),
                        )

                    if detail[1] == -1:
                        for ax in axes[pos]:
                            draw(ax)
                    else:
                        draw(axes[pos, detail[1]])

            if max_x > 0:
                for pos, (mode, mode_name) in enumerate(modes.items()):
                    for comment, ax in zip(comments, axes[pos]):
                        ax: Axes
                        ax.set_xlim([0, max_x + 1])
                        ax.set_xticks(np.arange(0, max_x + 1, 2))
                        ax.set_ylim(y_lim)
                        ax.grid(which="both")
                        ax.set_title(f"{mode_name}, {comment}")

                        if mode == list(modes.keys())[-1]:
                            ax.set_xlabel("finetune epochs")
                        if comment == comments[0]:
                            ax.set_ylabel(metric)

                fig.suptitle(f"GreaseArg ArgKP dev {metric}, {format} format")
                fig.tight_layout()
                fig.subplots_adjust(bottom=0.1, right=0.9, top=0.9)

                plt.colorbar(
                    mappable=mappable,
                    cax=fig.add_axes([0.925, 0.1, 0.025, 0.8]),
                    label="pretrain steps ($\\times10^4$)",
                )

                fig.savefig(summary_dir / f"dev_{metric}_{format}.png", dpi=300)
                print(f"dev_{metric}_{format}.png saved")

            plt.close(fig)
