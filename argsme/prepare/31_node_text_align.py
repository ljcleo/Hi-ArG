from pathlib import Path

import numpy as np
import pandas as pd
from pandarallel import pandarallel
from utils.args import get_target_dataset
from utils.io import dump_component, load_component

if __name__ == "__main__":
    data_dir: Path = Path("data")
    source_dir: Path = data_dir / "tokenized"
    split_source_dir: Path = data_dir / "splitted"
    target_dir: Path = data_dir / "aligned"
    target_dir.mkdir(exist_ok=True)

    target_dataset: str | None = get_target_dataset()
    pandarallel.initialize(progress_bar=True)

    for dataset_dir in source_dir.iterdir():
        dataset_name: str = dataset_dir.stem
        if target_dataset is not None and dataset_name != target_dataset:
            continue

        top_snt: np.ndarray = np.load(dataset_dir / "top_snt.npz")["data"]
        text_align: np.ndarray = np.load(dataset_dir / "text_align.npz")["data"]
        print(f"{dataset_name} text loaded.")

        nodes: pd.DataFrame = load_component(dataset_dir, "", "nodes")
        edges: pd.DataFrame = load_component(dataset_dir, "", "edges")
        nodes_start: np.ndarray = nodes["start"].to_numpy()
        nodes_end: np.ndarray = np.append(nodes_start[1:], edges.shape[0])
        graph_align: np.ndarray = np.load(dataset_dir / "graph_align.npz")["data"]
        graph_align_start: np.ndarray = np.load(dataset_dir / "graph_align_start.npz")["data"]
        graph_align_end: np.ndarray = np.append(graph_align_start[1:], graph_align.shape[0])

        def get_graph_align(align_index: int) -> np.ndarray:
            align_start: int = graph_align_start[align_index]
            align_end: int = graph_align_end[align_index]
            return graph_align[align_start:align_end, :]

        print(f"{dataset_name} graph loaded.")

        target_sub_dir: Path = target_dir / dataset_name
        target_sub_dir.mkdir(exist_ok=True)

        for split in ("eval", "train"):
            split_dir: Path = split_source_dir / dataset_name / split
            aligns: pd.DataFrame = load_component(split_dir, "", "aligns")
            tops: pd.DataFrame = load_component(split_dir, "", "tops")

            tops_start: np.ndarray = tops["start"].to_numpy()
            tops_end: np.ndarray = tops["end"].to_numpy()
            tops_align_start: np.ndarray = tops["align_start"].to_numpy()
            tops_align_end: np.ndarray = np.append(tops_align_start[1:], aligns.shape[0])
            print(f"{dataset_name} split loaded.")

            grouped_aligns: pd.Series = aligns["aligns"].parallel_map(get_graph_align)

            align_pos_end: np.ndarray = np.cumsum(
                aligns["aligns"]
                .parallel_map(lambda x: graph_align_end[x] - graph_align_start[x])
                .to_numpy()
            )

            align_pos_start: np.ndarray = np.concatenate([[0], align_pos_end[:-1]])
            align_pos: np.ndarray = np.concatenate(grouped_aligns.tolist(), axis=0)
            print(f"{dataset_name} graph align reset.")

            def calc_cross(data: pd.Series) -> pd.Series:
                top_index: int = data.name

                snt_start: int = tops_start[top_index]
                snt_end: int = tops_end[top_index]
                text_align_pos: np.ndarray = text_align[snt_start:snt_end, :]

                align_start: int = tops_align_start[top_index]
                align_end: int = tops_align_end[top_index]

                if align_start == align_end:
                    return np.array([], dtype=int).reshape(-1, 2)

                node_indices: list[int] = aligns["node"].iloc[align_start:align_end].tolist()
                node_align_pos_start: np.ndarray = align_pos_start[align_start:align_end]
                node_align_pos_end: np.ndarray = align_pos_end[align_start:align_end]

                node_align_pos: np.ndarray = align_pos[
                    node_align_pos_start[0] : node_align_pos_end[-1]
                ]

                node_align_pos_end -= node_align_pos_start[0]
                node_align_pos_start -= node_align_pos_start[0]

                cross_link: np.ndarray = np.greater.outer(
                    node_align_pos[:, 1], text_align_pos[:, 0]
                ) & np.less.outer(node_align_pos[:, 0], text_align_pos[:, 1])

                merged_link: np.ndarray = np.vstack(
                    [
                        np.any(cross_link[s:t, :], axis=0)
                        for s, t in zip(node_align_pos_start, node_align_pos_end)
                    ]
                )

                return np.array(
                    [[node_indices[node], token] for node, token in zip(*merged_link.nonzero())]
                ).reshape(-1, 2)

            cross_align: list[np.ndarray] = tops.parallel_apply(calc_cross, axis=1).tolist()
            cross_align_end: np.ndarray = np.cumsum([x.shape[0] for x in cross_align])
            tops["cross_start"] = np.concatenate([[0], cross_align_end[:-1]]).astype(np.uint32)
            aligns.drop(columns=["aligns"], inplace=True)
            print(f"{dataset_name} cross align calculated.")

            target_split_dir: Path = target_sub_dir / split
            target_split_dir.mkdir(exist_ok=True)

            dump_component(tops, target_split_dir, "", "tops")
            dump_component(aligns, target_split_dir, "", "aligns")
            print(f"{dataset_name} {split} split dumped.")

            np.savez_compressed(
                target_split_dir / "cross_align.npz",
                data=np.concatenate(cross_align, axis=0).astype(np.uint32),
            )

            print(f"{dataset_name} {split} cross align dumped.")
