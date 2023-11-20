from json import dumps, loads
from pathlib import Path

import pandas as pd
from pandarallel import pandarallel
from utils.file import cleanup
from utils.io import dump_graphs, load_graphs


def merge_aligns(aligns: pd.DataFrame) -> str:
    events: list[tuple[int, int]] = sum(
        [[(start, 1), (stop, -1)] for piece in aligns["aligns"] for start, stop in loads(piece)], []
    )

    events.sort(key=lambda x: (x[0], -x[1]))
    results: list[list[int]] = []
    left: int = -1
    counter: int = 0

    for point, inc in events:
        if counter == 0:
            assert inc == 1
            left = point

        counter += inc
        assert counter >= 0

        if counter == 0:
            results.append([left, point])

    return dumps(results)


def merge_all_aligns(aligns: pd.DataFrame) -> pd.DataFrame:
    dup_mark: pd.Series = aligns.duplicated(["id", "node"], keep=False)
    single: pd.DataFrame = aligns[~dup_mark].copy()
    pandarallel.initialize(progress_bar=True)
    single["aligns"] = single["aligns"].parallel_map(lambda x: dumps(sorted(loads(x))))
    result: pd.DataFrame

    if dup_mark.any():
        result = pd.concat(
            [
                single,
                aligns[dup_mark]
                .groupby(["id", "node"])
                .apply(merge_aligns)
                .rename("aligns")
                .reset_index(),
            ]
        )
    else:
        result = single

    result.sort_values(["id", "node"], ignore_index=True, inplace=True)
    return result


if __name__ == "__main__":
    data_dir = Path("data")
    in_dir: Path = data_dir / "output"
    out_dir: Path = data_dir / "final"
    cleanup(out_dir)

    nodes: pd.DataFrame
    edges: pd.DataFrame
    aligns: pd.DataFrame
    tops: pd.DataFrame
    nodes, edges, aligns, tops = load_graphs(in_dir, "")
    print("graph loaded.")

    node_map: dict[int, int] = {x: i for i, x in enumerate(nodes["id"])}
    nodes["text"] = nodes["text"].astype("category")
    nodes.drop(columns=["id"], inplace=True)
    edges["type"] = edges["type"].astype("category")

    top_map: dict[int, int] = {x: i for i, x in enumerate(tops["id"])}
    tops["boa"] = ~((tops["id"] % 10000).astype(bool))
    tops.drop(columns=["id"], inplace=True)

    edges["from"] = edges["from"].map(node_map)
    edges["to"] = edges["to"].map(node_map)
    aligns["id"] = aligns["id"].map(top_map)
    aligns["node"] = aligns["node"].map(node_map)
    tops["top"] = tops["top"].map(node_map)
    print("graph reindexed.")

    aligns = merge_all_aligns(aligns)
    aligns["aligns"] = aligns["aligns"].astype("category")
    print("aligns merged.")

    dump_graphs(nodes, edges, aligns, tops, out_dir, "")
    print("graph dumped.")
