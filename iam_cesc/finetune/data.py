from dataclasses import dataclass
from pathlib import Path
from pickle import load
from random import choices, shuffle

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass
class IAMCESCDataset(Dataset):
    data: list[dict[str, np.ndarray]]
    neg_ratio: float | None = None

    def __post_init__(self) -> None:
        if self.neg_ratio is not None and self.neg_ratio < 0:
            self.neg_ratio = None

        self.pos: list[int] = []
        self.neg: list[int] = []
        self.id_list: list[int] = []

        for i, sample in enumerate(self.data):
            (self.neg if sample["labels"][0] == 1 else self.pos).append(i)

        self.resample()

    def resample(self) -> None:
        self.id_list = self.pos.copy()

        self.id_list.extend(
            self.neg
            if self.neg_ratio is None
            else choices(self.neg, k=round(len(self.pos) * self.neg_ratio))
        )

        shuffle(self.id_list)

    @classmethod
    def load_train_dev_test(
        cls, dataset: str, neg_ratio: float | None = 5.0
    ) -> tuple["IAMCESCDataset", "IAMCESCDataset", "IAMCESCDataset"]:
        data_dir: Path = Path("data") / dataset
        datasets: list["IAMCESCDataset"] = []

        for mode in ("train", "dev", "test"):
            cur_ratio: float | None = neg_ratio if mode == "train" else None
            with (data_dir / f"{mode}.pkl").open("rb") as f:
                datasets.append(IAMCESCDataset(load(f), neg_ratio=cur_ratio))

        return tuple(datasets)

    def __len__(self) -> int:
        return len(self.id_list)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            key: torch.tensor(value if key == "pos" else value.astype(int))
            for key, value in self.data[self.id_list[index]].items()
        }
