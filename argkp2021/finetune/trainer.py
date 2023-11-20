from collections import defaultdict
from json import dump
from os import path

import numpy as np
from torch.utils.data import Dataset
from transformers import Trainer
from transformers.trainer_utils import EvalPrediction


def calc_ap(label: np.ndarray, relax: bool) -> float:
    label = label.copy()
    label[label < 0] = relax

    return (
        np.sum(label * np.cumsum(label) / (np.arange(label.shape[0]) + 1.0)).item() / label.shape[0]
    )


def compute_metrics(output: EvalPrediction) -> dict[str, float]:
    predictions: list[np.ndarray] = output.predictions
    labels: list[np.ndarray] = output.label_ids
    pred_assign: dict[int, tuple[float, int]] = {}

    for scores, label in zip(predictions, labels):
        label = label.reshape(-1, 2, label.shape[-1])
        meta_id: np.ndarray = label[:, 1, :]
        label = label[:, 0, :]
        pivot: int = 0

        for group, ids in zip(label, meta_id):
            mask: np.ndarray = ids != -100
            group = group[mask]
            ids = ids[mask]
            aid: int = ids[0].item()
            count: int = mask.sum().item() - 1

            for i in range(count):
                score: float = scores[pivot].item()
                gold: int = group[i + 1].item()
                pivot += 1

                if aid not in pred_assign or score > pred_assign[aid][0]:
                    pred_assign[aid] = score, gold

    group_pred: defaultdict[int, list[tuple[float, int]]] = defaultdict(list)

    for aid, (score, gold) in pred_assign.items():
        gid: int = aid // 1000
        group_pred[gid].append((score, gold))

    strict_ap: list[float] = []
    relaxed_ap: list[float] = []

    for group in group_pred.values():
        group = sorted(group, key=lambda x: -x[0])[: len(group) >> 1]
        pred_label: np.ndarray = np.array([x[1] for x in group])
        strict_ap.append(calc_ap(pred_label, False))
        relaxed_ap.append(calc_ap(pred_label, True))

    all_pred: list[int] = [
        max(x[1], 0) for x in sorted(sum(group_pred.values(), []), key=lambda x: -x[0])
    ]

    strict_map: float = np.mean(strict_ap).item()
    relaxed_map: float = np.mean(relaxed_ap).item()

    return {
        "strict_mAP": strict_map,
        "relaxed_mAP": relaxed_map,
        "mean_mAP": (strict_map + relaxed_map) / 2,
        "strict_p@50%": np.mean(all_pred[: len(all_pred) >> 1]).item(),
    }


class ArgKP2021Trainer(Trainer):
    def __init__(self, dump_name: str, test_dataset: Dataset = None, **kwargs) -> None:
        super().__init__(**kwargs)
        self.dump_name: str = dump_name
        self.test_dataset = test_dataset

    def train(self, **kwargs):
        self.cp_metrics: defaultdict[str, dict[str, float]] = defaultdict(dict)
        return super().train(**kwargs)

    def save_model(self, output_dir: str | None = None, _: bool = False):
        if output_dir is None:
            output_dir = self.args.output_dir

        for x in self.state.log_history:
            self.cp_metrics[self.ff(x["step"])].update({k: v for k, v in x.items() if k != "step"})

        if self.args.should_save:
            loc: str = path.join(output_dir, f"{self.dump_name}.json")
            with open(loc, "w", encoding="utf8") as f:
                dump(self.cp_metrics, f, indent=4, sort_keys=True)

    def _save_checkpoint(self, model, trial, metrics=None):
        self.cp_metrics[self.ff(self.state.global_step)].update(
            self.evaluate(self.test_dataset, metric_key_prefix="test")
        )

    @staticmethod
    def ff(x: int) -> str:
        return f"{x:04d}"
