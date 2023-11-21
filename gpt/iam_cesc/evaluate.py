from json import dump
from pathlib import Path

import numpy as np
import pandas as pd


if __name__ == "__main__":
    data_dir = Path("converted")
    pred_dir = Path("prediction")
    eval_dir = Path("evaluation")
    eval_dir.mkdir(exist_ok=True)

    labels: np.ndarray = pd.read_json(data_dir / "output.jsonl", lines=True)["label"].to_numpy()

    for prompt_method in ("direct", "explain"):
        predictions: np.ndarray = pd.read_json(pred_dir / f"{prompt_method}.jsonl", lines=True)[
            "label"
        ].to_numpy()

        accuracy: float = np.mean(predictions == labels).item()
        all_f1: list[float] = []

        for i in range(3):
            ip: int = np.sum(predictions == i).item()
            il: int = np.sum(labels == i).item()
            tp: int = np.sum((predictions == i) & (labels == i)).item()
            all_f1.append((tp << 1) / (ip + il))

        with open(eval_dir / f"{prompt_method}.json", "w", encoding="utf8") as f:
            dump(
                {
                    "macro_F1": np.mean(all_f1).item(),
                    "micro_F1": accuracy,
                    "support_F1": all_f1[2],
                    "contest_F1": all_f1[0],
                },
                f,
            )
