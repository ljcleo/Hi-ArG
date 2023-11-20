from json import dump
from pathlib import Path

import pandas as pd


def dump_graphs(
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
    aligns: pd.DataFrame,
    tops: pd.DataFrame,
    output_file: Path,
) -> None:
    with output_file.open("w", encoding="utf8") as f:
        dump(
            {
                "nodes": nodes.to_dict(orient="records"),
                "edges": edges.to_dict(orient="records"),
                "aligns": aligns.to_dict(orient="records"),
                "tops": tops.to_dict(orient="records"),
            },
            f,
            ensure_ascii=False,
        )
