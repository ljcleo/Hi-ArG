from collections import defaultdict
from json import dump
from os import path

import numpy as np
from torch.utils.data import Dataset
from transformers import Trainer, TrainerCallback, TrainerControl, TrainerState, TrainingArguments
from transformers.trainer_utils import EvalPrediction


def compute_metrics(output: EvalPrediction) -> dict[str, float]:
    predictions: np.ndarray = np.argmax(output.predictions, axis=1)
    labels: np.ndarray = output.label_ids
    accuracy: float = np.mean(predictions == labels).item()
    all_f1: list[float] = []

    for i in range(3):
        ip: int = np.sum(predictions == i).item()
        il: int = np.sum(labels == i).item()
        tp: int = np.sum((predictions == i) & (labels == i)).item()
        all_f1.append((tp << 1) / (ip + il))

    return {
        "macro_F1": np.mean(all_f1).item(),
        "micro_F1": accuracy,
        "support_F1": all_f1[2],
        "contest_F1": all_f1[0],
    }


class IAMCESCTrainer(Trainer):
    def __init__(self, dump_name: str, test_dataset: Dataset = None, **kwargs) -> None:
        super().__init__(**kwargs)
        self.dump_name: str = dump_name
        self.test_dataset = test_dataset
        self.add_callback(ResamplingCallback(self))

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


class ResamplingCallback(TrainerCallback):
    def __init__(self, trainer: IAMCESCTrainer) -> None:
        super().__init__()
        self.trainer = trainer

    def on_epoch_begin(
        self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs
    ) -> None:
        self.trainer.train_dataset.resample()
