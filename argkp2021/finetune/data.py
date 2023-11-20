from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
from pickle import load

import numpy as np
import torch
from torch.utils.data import Dataset


class ContextFormat(Enum):
    SPLIT = auto()
    PAIR = auto()
    CONCAT = auto()


@dataclass
class ArgKP2021Dataset(Dataset):
    data: list[tuple[dict[str, np.ndarray], ...]]

    @classmethod
    def load_train_dev_test(
        cls, dataset: str, format: ContextFormat
    ) -> tuple["ArgKP2021Dataset", "ArgKP2021Dataset", "ArgKP2021Dataset"]:
        data_dir: Path = Path("data") / dataset
        datasets: list["ArgKP2021Dataset"] = []

        for mode in ("train", "dev", "test"):
            with (data_dir / f"{format.value}_{mode}.pkl").open("rb") as f:
                datasets.append(ArgKP2021Dataset(load(f)))

        return tuple(datasets)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> tuple[dict[str, torch.Tensor], ...]:
        return tuple(
            {
                key: torch.tensor(value if key == "pos" else value.astype(int))
                for key, value in slice.items()
            }
            for slice in self.data[index]
        )
