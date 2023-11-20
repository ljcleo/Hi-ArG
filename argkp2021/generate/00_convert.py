from pathlib import Path
from shutil import rmtree

import pandas as pd


def get_tid(key: str) -> int:
    return int(key.split("_")[1])


def get_iid(key: str) -> int:
    return int(key.split("_")[-1])


def make_uid(row: pd.Series) -> int:
    tid: int = row["tid"]
    stance: int = row["stance"]
    iid: int = row["iid"]
    return ((tid << 1) | ((1 + stance) >> 1)) * 1000 + iid


def extract(in_dir: Path, mode: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    arguments: pd.DataFrame = pd.read_csv(in_dir / f"arguments_{mode}.csv")
    arguments["tid"] = arguments["arg_id"].apply(get_tid)
    arguments["iid"] = arguments["arg_id"].apply(get_iid)
    arguments["aid"] = arguments.apply(make_uid, axis=1)
    arguments["argument"] = arguments["argument"].str.replace(r"\s+", " ", regex=True)
    arguments.set_index("arg_id", inplace=True)
    arguments.drop(columns=["topic", "iid"], inplace=True)

    key_points: pd.DataFrame = pd.read_csv(in_dir / f"key_points_{mode}.csv")
    key_points["tid"] = key_points["key_point_id"].apply(get_tid)
    key_points["iid"] = key_points["key_point_id"].apply(get_iid)
    key_points["kid"] = key_points.apply(make_uid, axis=1)

    key_points["key_point"] = key_points.apply(
        lambda x: f'{x["topic"]}? {x["key_point"]}', axis=1
    ).str.replace(r"\s+", " ", regex=True)

    key_points.set_index("key_point_id", inplace=True)
    key_points.drop(columns=["topic", "iid"], inplace=True)

    labels: pd.DataFrame = pd.read_csv(in_dir / f"labels_{mode}.csv")
    labels["argument"] = arguments.loc[labels["arg_id"], "argument"].reset_index(drop=True)
    labels["key_point"] = key_points.loc[labels["key_point_id"], "key_point"].reset_index(drop=True)
    labels.drop(columns=["arg_id", "key_point_id"], inplace=True)
    arguments.reset_index(drop=True, inplace=True)
    key_points.reset_index(drop=True, inplace=True)

    sentences: pd.DataFrame = pd.concat(
        [
            arguments[["tid", "argument"]].rename(columns={"argument": "snt"}),
            key_points[["tid", "key_point"]].rename(columns={"key_point": "snt"}),
        ],
        axis=0,
        ignore_index=True,
    )

    candidates: pd.DataFrame = (
        pd.merge(arguments, key_points, on=["tid", "stance"])
        .drop(columns=["tid", "stance"])
        .sort_values(["aid", "kid"])
    )

    labels = pd.merge(
        candidates, labels, how="inner" if mode == "train" else "left", on=["argument", "key_point"]
    ).fillna(-1)

    labels["label"] = labels["label"].astype(int)
    return sentences, labels[["aid", "kid", "argument", "key_point", "label"]]


if __name__ == "__main__":
    data_dir = Path("data")
    in_dir: Path = data_dir / "raw"
    out_dir: Path = data_dir / "arranged"
    task_dir: Path = data_dir / "task"

    for dir in (out_dir, task_dir):
        if dir.exists():
            rmtree(dir, ignore_errors=True)

        dir.mkdir(exist_ok=True)

    data: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {
        mode: extract(in_dir, mode) for mode in ("train", "dev", "test")
    }

    offset: int = max((x["tid"].max() for x, _ in data.values())) + 1
    data["test"][0]["tid"] += offset

    for key in ("aid", "kid"):
        data["test"][1][key] += offset * 2000

    sentences: pd.DataFrame = pd.concat([x for x, _ in data.values()], axis=0, ignore_index=True)

    for tid, group in sentences.groupby("tid"):
        (out_dir / f"{tid:04d}.txt").write_text(
            "\n".join(group.to_dict(orient="list")["snt"]), encoding="utf8"
        )

    for mode, (_, labels) in data.items():
        labels.to_json(task_dir / f"{mode}.jsonl", orient="records", force_ascii=False, lines=True)
