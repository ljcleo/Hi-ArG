from json import dump
from pathlib import Path

import numpy as np
import pandas as pd


def calc_ap(label: np.ndarray, relax: bool) -> float:
    label = label.copy()
    label[label < 0] = relax

    return (
        np.sum(label * np.cumsum(label) / (np.arange(label.shape[0]) + 1.0)).item() / label.shape[0]
    )


if __name__ == "__main__":
    data_dir = Path("converted")
    pred_dir = Path("prediction")
    eval_dir = Path("evaluation")
    eval_dir.mkdir(exist_ok=True)

    inputs: pd.DataFrame = pd.read_json(data_dir / "input.jsonl", lines=True)
    outputs: pd.DataFrame = pd.read_json(data_dir / "output.jsonl", lines=True)

    for prompt_method in ("direct", "explain"):
        predictions: pd.DataFrame = (
            pd.merge(
                outputs,
                pd.read_json(pred_dir / f"{prompt_method}.jsonl", lines=True).rename(
                    columns={"label": "score"}
                ),
                left_index=True,
                right_index=True,
            )
            .sort_values(["arg_id", "score", "key_point_id"], ascending=[True, False, True])
            .groupby("arg_id")
            .head(1)
            .merge(inputs[["arg_id", "topic", "stance"]].drop_duplicates())
            .sort_values(["score", "arg_id"], ascending=[False, True])
        )

        groups = predictions.groupby(["topic", "stance"])
        predictions = predictions.loc[groups.cumcount() <= groups["arg_id"].transform("size") * 0.5]

        strict_map: float = (
            predictions.groupby(["topic", "stance"])["label"]
            .aggregate(lambda x: calc_ap(x, False))
            .mean()
        )

        relaxed_map: float = (
            predictions.groupby(["topic", "stance"])["label"]
            .aggregate(lambda x: calc_ap(x, True))
            .mean()
        )

        mean_map: float = (strict_map + relaxed_map) / 2
        strict_p50: float = (predictions["label"] == 1).mean()

        with open(eval_dir / f"{prompt_method}.json", "w", encoding="utf8") as f:
            dump(
                {
                    "strict_mAP": strict_map,
                    "relaxed_mAP": relaxed_map,
                    "mean_mAP": mean_map,
                    "strict_p@50%": strict_p50,
                },
                f,
            )
