from pathlib import Path
from shutil import copy

import numpy as np
import pandas as pd
from utils.args import get_target_dataset
from utils.io import load_component

if __name__ == "__main__":
    data_dir: Path = Path("data")
    source_dir: Path = data_dir / "tokenized"
    split_source_dir: Path = data_dir / "aligned"
    vocab_dir: Path = data_dir / "vocab"
    target_dir: Path = data_dir / "simplified"
    target_dir.mkdir(exist_ok=True)

    copy(vocab_dir / "tokens.json", target_dir)
    print("vocab data copied.")
    target_dataset: str | None = get_target_dataset()

    for dataset_dir in source_dir.iterdir():
        dataset_name: str = dataset_dir.stem
        split_dataset_dir: Path = split_source_dir / dataset_name

        if target_dataset is not None and dataset_name != target_dataset:
            continue

        nodes: pd.DataFrame = load_component(dataset_dir, "", "nodes")
        edges: pd.DataFrame = load_component(dataset_dir, "", "edges")
        top_snt: np.ndarray = np.load(dataset_dir / "top_snt.npz")["data"]
        print(f"{dataset_name} main data loaded.")

        nodes_text: np.ndarray = nodes["text"].to_numpy()
        nodes_start: np.ndarray = nodes["start"].to_numpy()
        nodes_end: np.ndarray = np.append(nodes_start[1:], edges.shape[0])
        edges_data: np.ndarray = edges.to_numpy()
        print(f"{dataset_name} main data simplified.")

        target_sub_dir: Path = target_dir / dataset_name
        target_sub_dir.mkdir(exist_ok=True)

        np.savez_compressed(
            target_sub_dir / "main.npz",
            nodes_text=nodes_text,
            nodes_start=nodes_start,
            nodes_end=nodes_end,
            edges_data=edges_data,
            top_snt=top_snt,
        )

        print(f"{dataset_name} main data dumped.")

        for split in ("train", "eval"):
            sub_dir: Path = split_dataset_dir / split
            aligns: pd.DataFrame = load_component(sub_dir, "", "aligns")
            tops: pd.DataFrame = load_component(sub_dir, "", "tops")
            cross_align: np.ndarray = np.load(sub_dir / "cross_align.npz")["data"]
            print(f"{dataset_name} {split} data loaded.")

            aligns_node: np.ndarray = aligns["node"].to_numpy()
            tops_top: np.ndarray = tops["top"].to_numpy()
            tops_boa: np.ndarray = tops["boa"].to_numpy()
            tops_pair: np.ndarray = tops["pair"].to_numpy()
            tops_against: np.ndarray = tops["against"].to_numpy()
            tops_start: np.ndarray = tops["start"].to_numpy()
            tops_end: np.ndarray = tops["end"].to_numpy()
            tops_align_start: np.ndarray = tops["align_start"].to_numpy()
            tops_align_end: np.ndarray = np.append(tops_align_start[1:], aligns.shape[0])
            tops_cross_start: np.ndarray = tops["cross_start"].to_numpy()
            tops_cross_end: np.ndarray = np.append(tops_cross_start[1:], cross_align.shape[0])
            print(f"{dataset_name} {split} data simplified.")

            np.savez_compressed(
                target_sub_dir / f"{split}.npz",
                aligns_node=aligns_node,
                tops_top=tops_top,
                tops_boa=tops_boa,
                tops_pair=tops_pair,
                tops_against=tops_against,
                tops_start=tops_start,
                tops_end=tops_end,
                tops_align_start=tops_align_start,
                tops_align_end=tops_align_end,
                tops_cross_start=tops_cross_start,
                tops_cross_end=tops_cross_end,
                cross_align=cross_align,
            )

            print(f"{dataset_name} {split} data dumped.")
