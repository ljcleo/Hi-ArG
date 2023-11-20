from json import loads
from pathlib import Path

import numpy as np
import pandas as pd
from utils.args import get_target_dataset
from utils.io import dump_component, load_component
from utils.np import cat_ragged

if __name__ == "__main__":
    data_dir: Path = Path("data")
    source_dir: Path = data_dir / "paired"
    target_dir: Path = data_dir / "tokenized"
    target_dir.mkdir(exist_ok=True)
    target_dataset: str | None = get_target_dataset()

    for dataset_dir in source_dir.iterdir():
        dataset_name: str = dataset_dir.stem
        if target_dataset is not None and dataset_name != target_dataset:
            continue

        aligns: pd.DataFrame = load_component(dataset_dir, "", "aligns")
        print(f"{dataset_name} aligns loaded.")

        graph_align_cat: np.ndarray
        graph_align_start: np.ndarray

        graph_align_cat, graph_align_start = cat_ragged(
            [
                np.array(loads(x), dtype=np.uint16).reshape(-1, 2)
                for x in aligns["aligns"].cat.categories
            ]
        )

        aligns["id"] = aligns["id"].astype(np.uint32)
        aligns["node"] = aligns["node"].astype(np.uint32)
        aligns["aligns"] = aligns["aligns"].cat.codes.astype(np.uint32)
        graph_align_start = graph_align_start.astype(np.uint32)

        print(f"{dataset_name} graph align concatenated.")

        target_sub_dir: Path = target_dir / dataset_name
        target_sub_dir.mkdir(exist_ok=True)

        dump_component(aligns, target_sub_dir, "", "aligns")
        print(f"{dataset_name} aligns dumped.")

        np.savez_compressed(target_sub_dir / "graph_align.npz", data=graph_align_cat)
        np.savez_compressed(target_sub_dir / "graph_align_start.npz", data=graph_align_start)
        print(f"{dataset_name} graph align dumped.")
