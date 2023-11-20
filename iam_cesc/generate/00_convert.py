from pathlib import Path
from shutil import rmtree

import pandas as pd


def extract(in_dir: Path, mode: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    data: pd.DataFrame = pd.read_table(
        in_dir / f"{mode}.txt",
        delimiter="\t",
        usecols=[1, 2, 4],
        names=["topic", "candidate", "stance"],
    ).rename(columns={"stance": "label"})

    for key in ("topic", "candidate"):
        data[key] = data[key].astype("str").str.replace(r"\s+", " ", regex=True).str.strip()

    data = data.query('topic != "" and candidate != ""')
    data["label"] += 1

    data["tid"] = (
        data["topic"]
        .drop_duplicates()
        .reset_index()
        .set_index("topic")["index"][data["topic"]]
        .reset_index(drop=True)
    )

    sentences: pd.DataFrame = pd.concat(
        [
            data[["tid", "topic"]].rename(columns={"topic": "snt"}),
            data[["tid", "candidate"]].rename(columns={"candidate": "snt"}),
        ],
        axis=0,
    )

    data = data[["tid", "topic", "candidate", "label"]]
    return sentences.drop_duplicates(ignore_index=True), data


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

    pivot: int = 0

    for mode, (sentences, labels) in data.items():
        for tid, group in sentences.groupby("tid"):
            (out_dir / f"{pivot:04d}.txt").write_text(
                "\n".join(group.to_dict(orient="list")["snt"]), encoding="utf8"
            )

            pivot += 1

        labels.to_json(task_dir / f"{mode}.jsonl", orient="records", force_ascii=False, lines=True)
