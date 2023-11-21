from pathlib import Path

import pandas as pd

if __name__ == "__main__":
    raw_dir = Path("raw")
    conv_dir = Path("converted")
    conv_dir.mkdir(exist_ok=True)

    data: pd.DataFrame = pd.read_table(
        raw_dir / "test.txt",
        delimiter="\t",
        usecols=[1, 2, 4],
        names=["topic", "candidate", "stance"],
    ).rename(columns={"stance": "label"})

    for key in ("topic", "candidate"):
        data[key] = data[key].astype("str").str.replace(r"\s+", " ", regex=True).str.strip()

    data = data.query('topic != "" and candidate != ""')
    data["label"] += 1

    data[["topic", "candidate"]].to_json(conv_dir / "input.jsonl", orient="records", lines=True)
    data[["label"]].to_json(conv_dir / "output.jsonl", orient="records", lines=True)
