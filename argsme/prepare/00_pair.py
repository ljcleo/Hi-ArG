from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from pandarallel import pandarallel
from utils.args import get_target_dataset
from utils.io import dump_graphs, load_graphs

MAX_DEGREE: int = 500
MIN_DIST: int = 32


if __name__ == "__main__":
    data_dir: Path = Path("data")
    source_dir: Path = data_dir / "dataset"
    target_dir: Path = data_dir / "paired"
    target_dir.mkdir(exist_ok=True)

    target_dataset: str | None = get_target_dataset()
    pandarallel.initialize(progress_bar=True)

    for dataset_dir in source_dir.iterdir():
        dataset_name: str = dataset_dir.stem
        if target_dataset is not None and dataset_name != target_dataset:
            continue

        nodes: pd.DataFrame
        edges: pd.DataFrame
        aligns: pd.DataFrame
        tops: pd.DataFrame
        nodes, edges, aligns, tops = load_graphs(dataset_dir, "")

        tops_arg: list[str] = []
        for i, (x, y) in enumerate(zip(tops["snt"], tops["boa"])):
            tops_arg.append(x if y else tops_arg[-1])

        tops["topic"] = pd.Series([x[1:-11] for x in tops_arg], dtype="category")
        tops["stance"] = [x[-2] == "t" for x in tops_arg]
        print(f"{dataset_name} graph loaded.", flush=True)

        align_tn: pd.DataFrame = aligns[["id", "node"]].copy()
        node_count: pd.DataFrame = align_tn.groupby("node").count()

        node_weight: pd.DataFrame = (
            node_count.query(f"1 < id <= {MAX_DEGREE}")
            .rename(columns={"id": "weight"})
            .applymap(lambda x: 1.0 / x)
            .reset_index()
        )

        cand_tn: pd.DataFrame = align_tn.merge(node_weight, how="inner", on="node")
        print(f"{dataset_name} candidate matrix calculated.", flush=True)

        tn_csr = sp.csr_array(
            (np.ones(cand_tn.shape[0]), (cand_tn["id"], cand_tn["node"])),
            shape=(tops.shape[0], nodes.shape[0]),
        )

        nt_csr = sp.csr_array(
            (cand_tn["weight"], (cand_tn["node"], cand_tn["id"])),
            shape=(nodes.shape[0], tops.shape[0]),
        )

        tt: sp.coo_array = (tn_csr @ nt_csr).tocoo()
        print(f"{dataset_name} matrix product calculated.", flush=True)

        source_top_ids: np.ndarray
        target_top_ids: np.ndarray
        source_top_ids, target_top_ids = tt.nonzero()
        pair_weight: np.ndarray = tt.data
        print(f"{dataset_name} weight extracted.", flush=True)

        no_loop: np.ndarray = source_top_ids != target_top_ids
        source_top_ids = source_top_ids[no_loop]
        target_top_ids = target_top_ids[no_loop]
        pair_weight = pair_weight[no_loop]

        top_tops: np.ndarray = tops["top"].values
        diff_top: np.ndarray = top_tops[source_top_ids] != top_tops[target_top_ids]
        source_top_ids = source_top_ids[diff_top]
        target_top_ids = target_top_ids[diff_top]
        pair_weight = pair_weight[diff_top]

        topic_codes: np.ndarray = tops["topic"].cat.codes.values
        same_topic: np.ndarray = topic_codes[source_top_ids] == topic_codes[target_top_ids]
        source_top_ids = source_top_ids[same_topic]
        target_top_ids = target_top_ids[same_topic]
        pair_weight = pair_weight[same_topic]

        far_enough: np.ndarray = np.abs(source_top_ids - target_top_ids) >= MIN_DIST
        source_top_ids = source_top_ids[far_enough]
        target_top_ids = target_top_ids[far_enough]
        pair_weight = pair_weight[far_enough]
        print(f"{dataset_name} weight filtered.", flush=True)

        source_tops: pd.DataFrame = (
            tops.loc[np.unique(source_top_ids), "stance"].rename_axis(index="source").reset_index()
        )

        target_tops: pd.DataFrame = (
            tops.loc[np.unique(target_top_ids), "stance"].rename_axis(index="target").reset_index()
        )

        cand_pairs = (
            pd.DataFrame(
                {"source": source_top_ids, "target": target_top_ids, "weight": pair_weight}
            )
            .merge(source_tops, on="source")
            .merge(target_tops, on="target")
        )

        cand_pairs["against"] = cand_pairs["stance_x"] != cand_pairs["stance_y"]
        cand_pairs.drop(columns=["stance_x", "stance_y"], inplace=True)
        cand_pairs.sort_values(["source", "target"], ignore_index=True, inplace=True)
        print(f"{dataset_name} candidate pairs selected.", flush=True)

        final_index: np.ndarray = (
            cand_pairs.groupby("source")
            .parallel_apply(
                lambda x: np.random.default_rng(x.index[0]).choice(
                    x.index, p=x["weight"] / x["weight"].sum()
                )
            )
            .values
        )

        final: pd.DataFrame = cand_pairs.loc[final_index].reset_index(drop=True)

        real_final: pd.DataFrame = final[
            ~tops.loc[final["source"].values, "boa"].values
        ].reset_index(drop=True)

        print(f"{dataset_name} final pair sampled.", flush=True)

        tops["pair"] = -1
        tops["against"] = False
        tops.loc[real_final["source"].values, "pair"] = real_final["target"].values
        tops.loc[real_final["source"].values, "against"] = real_final["against"].values

        tops["pair"] = tops["pair"].astype(np.int32)
        tops["against"] = tops["against"].astype(bool)
        tops.drop(columns=["topic", "stance"], inplace=True)
        print(f"{dataset_name} top info updated.", flush=True)

        target_sub_dir: Path = target_dir / dataset_name
        target_sub_dir.mkdir(exist_ok=True)
        dump_graphs(nodes, edges, aligns, tops, target_sub_dir, "")
        print(f"{dataset_name} graph dumped.", flush=True)
