from pathlib import Path

import pandas as pd

if __name__ == "__main__":
    raw_dir = Path("raw")
    conv_dir = Path("converted")
    simp_dir = Path("simplified")
    conv_dir.mkdir(exist_ok=True)
    simp_dir.mkdir(exist_ok=True)

    arguments: pd.DataFrame = pd.read_csv(raw_dir / "arguments_test.csv")
    key_points: pd.DataFrame = pd.read_csv(raw_dir / "key_points_test.csv")
    labels: pd.DataFrame = pd.read_csv(raw_dir / "labels_test.csv")

    arguments["argument"] = arguments["argument"].str.replace(r"\s+", " ", regex=True)
    key_points["key_point"] = key_points["key_point"].str.replace(r"\s+", " ", regex=True)

    inputs: pd.DataFrame = arguments.merge(key_points, on=["tid", "stance"])
    inputs.to_json(conv_dir / "input.jsonl", orient="records", lines=True)

    inputs[["topic", "argument", "key_point"]].to_json(
        simp_dir / "input.jsonl", orient="records", lines=True
    )

    outputs: pd.DataFrame = inputs[["arg_id", "key_point_id"]].merge(labels, how="left")
    outputs["label"] = outputs["label"].fillna(-1).astype(int)
    outputs.to_json(conv_dir / "output.jsonl", orient="records", lines=True)
