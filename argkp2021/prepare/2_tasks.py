from pathlib import Path

import pandas as pd
from utils.args import get_target_dataset
from utils.io import load_component


def merge(row: pd.Series) -> list[list[int]]:
    return sorted(
        (list(x) for x in zip(row["kid"], row["key_point"], row["label"])), key=lambda x: x[0]
    )


if __name__ == "__main__":
    data_dir: Path = Path("data")
    source_dir: Path = data_dir / "dataset"
    target_dir: Path = data_dir / "tokenized"
    target_dir.mkdir(exist_ok=True)
    target_dataset: str | None = get_target_dataset()

    for dataset_dir in source_dir.iterdir():
        dataset_name: str = dataset_dir.stem
        if target_dataset is not None and dataset_name != target_dataset:
            continue

        tops: pd.DataFrame = load_component(dataset_dir, "", "tops")
        snt_top_map: pd.Series = tops.drop_duplicates("snt").reset_index().set_index("snt")["index"]
        print(f"{dataset_name} tops loaded.")

        for mode in ("train", "dev", "test"):
            tasks: pd.DataFrame = pd.read_json(
                dataset_dir / f"{mode}.jsonl", orient="records", lines=True
            )

            print(f"{dataset_name} {mode} tasks loaded.")

            tasks["argument"] = snt_top_map[tasks["argument"].str.strip()].reset_index(drop=True)
            tasks["key_point"] = snt_top_map[tasks["key_point"].str.strip()].reset_index(drop=True)
            tasks = tasks.groupby("aid").aggregate(list).reset_index()
            tasks["argument"] = tasks["argument"].str[0]
            tasks["key_point"] = tasks.apply(merge, axis=1)
            tasks.drop(columns=["kid", "label"], inplace=True)
            tasks = tasks[["aid", "argument", "key_point"]]
            print(f"{dataset_name} {mode} tasks prepared.")

            target_sub_dir: Path = target_dir / dataset_name
            target_sub_dir.mkdir(exist_ok=True)
            tasks.to_parquet(target_sub_dir / f"{mode}.bin")
            print(f"{dataset_name} {mode} tasks dumped.")
