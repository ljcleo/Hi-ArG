from pathlib import Path

import numpy as np
import pandas as pd
from utils.args import get_target_dataset
from utils.io import dump_component, load_component

RANDOM_SEED: int = 19260817

if __name__ == "__main__":
    data_dir: Path = Path("data")
    source_dir: Path = data_dir / "tokenized"
    target_dir: Path = data_dir / "splitted"
    target_dir.mkdir(exist_ok=True)
    target_dataset: str | None = get_target_dataset()

    for dataset_dir in source_dir.iterdir():
        dataset_name: str = dataset_dir.stem
        if target_dataset is not None and dataset_name != target_dataset:
            continue

        aligns: pd.DataFrame = load_component(dataset_dir, "", "aligns")
        tops: pd.DataFrame = load_component(dataset_dir, "", "tops")
        print(f"{dataset_name} graph loaded.")

        top_snt: np.ndarray = np.load(dataset_dir / "top_snt.npz")["data"]
        print(f"{dataset_name} text loaded.")

        tops["end"] = np.append(tops["start"].to_numpy()[1:], top_snt.shape[0]).astype(np.uint32)
        tops["arg"] = (np.cumsum(tops["boa"]) - 1).astype(np.uint32)
        num_args: int = tops["arg"].max() + 1

        eval_args: np.ndarray = np.random.default_rng(seed=RANDOM_SEED).choice(
            num_args, round(num_args * 0.05)
        )

        tops_eval_flag: pd.Series = tops["arg"].isin(eval_args)
        eval_tops: pd.DataFrame = tops[tops_eval_flag].copy()
        train_tops: pd.DataFrame = tops[~tops_eval_flag].copy()
        del tops

        aligns_eval_flag: pd.DataFrame = aligns["id"].isin(eval_tops.index)
        eval_aligns: pd.DataFrame = aligns[aligns_eval_flag].reset_index(drop=True)
        train_aligns: pd.DataFrame = aligns[~aligns_eval_flag].reset_index(drop=True)
        del aligns

        target_sub_dir: Path = target_dir / dataset_name
        target_sub_dir.mkdir(exist_ok=True)

        for split, tops, aligns in (
            ("eval", eval_tops, eval_aligns),
            ("train", train_tops, train_aligns),
        ):
            top_map = pd.Series(np.arange(tops.shape[0], dtype=np.uint32), index=tops.index)
            tops["pair"] = tops["pair"].map(lambda x: -1 if x not in tops.index else top_map[x])
            tops.loc[(tops["pair"] == -1).values, "against"] = False
            aligns["id"] = aligns["id"].map(top_map)
            tops.reset_index(drop=True, inplace=True)
            print(f"{dataset_name} top indices reset.")

            num_tops: int = tops.shape[0]
            good_tops: np.ndarray
            sub_count: np.ndarray
            good_tops, sub_count = np.unique(aligns["id"], return_counts=True)
            align_count: np.ndarray = np.zeros(num_tops, dtype=np.uint32)
            align_count[good_tops] = sub_count.astype(np.uint32)
            tops["align_start"] = np.cumsum(align_count, dtype=np.uint32) - align_count
            print(f"{dataset_name} align list prepared.")

            target_split_dir: Path = target_sub_dir / split
            target_split_dir.mkdir(exist_ok=True)

            dump_component(aligns, target_split_dir, "", "aligns")
            dump_component(tops, target_split_dir, "", "tops")
            print(f"{dataset_name} {split} graph dumped.")
