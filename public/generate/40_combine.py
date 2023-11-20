from multiprocessing.pool import ApplyResult, Pool
from pathlib import Path

import pandas as pd
from utils.file import cleanup
from utils.io import dump_graphs, load_component
from utils.pool import NestablePool


def merge_chunks(
    in_dir: Path,
    name_list: list[str],
    component: str,
    sort_keys: list[str],
    category_keys: list[str],
) -> pd.DataFrame:
    with Pool() as pool:
        results: list[ApplyResult] = [
            pool.apply_async(load_component, (in_dir, name, component)) for name in name_list
        ]

        data: pd.DataFrame = pd.concat([result.get() for result in results])
        print(f"{component} data loaded.")

    for key in category_keys:
        data[key] = data[key].astype("category")

    print(f"{component} category created.")
    data.sort_values(sort_keys, inplace=True, ignore_index=True)
    data.reset_index(drop=True)
    print(f"{component} sorted.")
    return data


def convert_top_id(old: str) -> int:
    aid, sid = tuple(int(x) for x in old.split("-"))
    return aid * 10000 + sid


if __name__ == "__main__":
    data_dir = Path("data")
    in_dir: Path = data_dir / "refined"
    out_dir: Path = data_dir / "combined"
    cleanup(out_dir)

    name_list: list[str] = sorted(
        (x.stem[:4] for x in in_dir.iterdir() if x.stem.endswith("_tops")), key=lambda x: int(x)
    )

    with NestablePool() as pool:
        results: list[ApplyResult] = [
            pool.apply_async(merge_chunks, (in_dir, name_list, component, sort_keys, category_keys))
            for component, sort_keys, category_keys in (
                ("nodes", ["id"], ["id"]),
                ("edges", ["from", "to", "type"], ["from", "to", "type"]),
                ("aligns", ["id", "node"], ["node"]),
                ("tops", ["id"], ["top"]),
            )
        ]

        nodes: pd.DataFrame
        edges: pd.DataFrame
        aligns: pd.DataFrame
        tops: pd.DataFrame
        nodes, edges, aligns, tops = tuple(result.get() for result in results)

    edges["from"] = edges["from"].cat.set_categories(nodes["id"]).cat.codes
    edges["to"] = edges["to"].cat.set_categories(nodes["id"]).cat.codes
    aligns["node"] = aligns["node"].cat.set_categories(nodes["id"]).cat.codes
    tops["top"] = tops["top"].cat.set_categories(nodes["id"]).cat.codes
    nodes["id"] = nodes["id"].cat.codes
    tops["id"] = tops["id"].map(convert_top_id)
    aligns["id"] = aligns["id"].map(convert_top_id)

    for data in (nodes, edges, aligns, tops):
        print(data.shape, data.memory_usage(deep=True), sep="\n")

    dump_graphs(nodes, edges, aligns, tops, out_dir, "")
