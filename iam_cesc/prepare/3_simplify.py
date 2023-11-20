from pathlib import Path
from shutil import copy

import numpy as np
import pandas as pd
from utils.args import get_target_dataset
from utils.io import load_graphs

if __name__ == "__main__":
    data_dir: Path = Path("data")
    source_dir: Path = data_dir / "tokenized"
    target_dir: Path = data_dir / "simplified"
    target_dir.mkdir(exist_ok=True)

    copy(source_dir / "tokens.json", target_dir)
    target_dataset: str | None = get_target_dataset()

    for dataset_dir in source_dir.iterdir():
        dataset_name: str = dataset_dir.stem

        if not dataset_dir.is_dir():
            continue
        if target_dataset is not None and dataset_name != target_dataset:
            continue

        nodes: pd.DataFrame
        edges: pd.DataFrame
        aligns: pd.DataFrame
        tops: pd.DataFrame
        nodes, edges, aligns, tops = load_graphs(dataset_dir, "")
        print(f"{dataset_name} graph loaded.")

        top_snt: np.ndarray = np.load(dataset_dir / "top_snt.npz")["data"]
        print(f"{dataset_name} text loaded.")

        nodes_text: np.ndarray = nodes["text"].to_numpy()
        nodes_start: np.ndarray = nodes["start"].to_numpy()
        nodes_end: np.ndarray = np.append(nodes_start[1:], edges.shape[0])
        edges_data: np.ndarray = edges.to_numpy()
        aligns_node: np.ndarray = aligns["node"].to_numpy()
        tops_top: np.ndarray = tops["top"].to_numpy()
        tops_start: np.ndarray = tops["start"].to_numpy()
        tops_end: np.ndarray = np.append(tops_start[1:], top_snt.shape[0])
        tops_align_start: np.ndarray = tops["align_start"].to_numpy()
        tops_align_end: np.ndarray = np.append(tops_align_start[1:], aligns.shape[0])
        tops_end: np.ndarray = np.append(tops_start[1:], top_snt.shape[0])
        print(f"{dataset_name} data simplified.")

        target_sub_dir: Path = target_dir / dataset_name
        target_sub_dir.mkdir(exist_ok=True)

        np.savez_compressed(
            target_sub_dir / "main.npz",
            nodes_text=nodes_text,
            nodes_start=nodes_start,
            nodes_end=nodes_end,
            edges_data=edges_data,
            aligns_node=aligns_node,
            tops_top=tops_top,
            tops_start=tops_start,
            tops_end=tops_end,
            tops_align_start=tops_align_start,
            tops_align_end=tops_align_end,
            top_snt=top_snt,
        )

        print(f"{dataset_name} data dumped.")

        for mode in ("train", "dev", "test"):
            tasks: pd.DataFrame = pd.read_parquet(dataset_dir / f"{mode}.bin")
            tasks = tasks.astype(int)

            tasks_topic: np.ndarray = tasks["topic"].to_numpy()
            tasks_candidate: np.ndarray = tasks["candidate"].to_numpy()
            tasks_label: np.ndarray = tasks["label"].to_numpy()
            print(f"{dataset_name} {mode} task loaded.")

            np.savez_compressed(
                target_sub_dir / f"{mode}.npz",
                tasks_topic=tasks_topic,
                tasks_candidate=tasks_candidate,
                tasks_label=tasks_label,
            )

            print(f"{dataset_name} {mode} task dumped.")
