from json import dumps, loads
from multiprocessing.pool import ApplyResult, Pool
from pathlib import Path

import pandas as pd
from utils.file import cleanup
from utils.io import dump_graphs, load_graphs


def align(data: pd.Series) -> pd.Series:
    snt: str = data["snt"]
    tokens: list[str] = loads(data["tokens"])
    result: list[tuple[int, int]] = []
    pivot: int = 0

    for token in tokens:
        while snt != "" and not snt.startswith(token):
            pivot += 1
            snt = snt[1:]

        if snt == "":
            raise RuntimeError(token)

        result.append((pivot, pivot + len(token)))
        snt = snt[len(token) :]
        pivot += len(token)

    return pd.Series({"id": data["id"], "tokens": dumps(result)})


def work(in_dir: Path, out_dir: Path, name: str, pid: int) -> int:
    nodes: pd.DataFrame
    edges: pd.DataFrame
    aligns: pd.DataFrame
    tops: pd.DataFrame
    nodes, edges, aligns, tops = load_graphs(in_dir, name)
    print(f"GROUP {name} LOADED.")

    top_tokens: pd.Series = tops.apply(align, axis=1).set_index("id")["tokens"]
    tops.drop(columns=["tokens"], inplace=True)

    aligns["aligns"] = aligns.apply(
        lambda x: dumps([loads(top_tokens[x["id"]])[y] for y in loads(x["aligns"])]), axis=1
    )

    print(f"GROUP {name} REFINED.")
    dump_graphs(nodes, edges, aligns, tops, out_dir, name)
    print(f"GROUP {name} DUMPED.", flush=True)
    return pid


if __name__ == "__main__":
    data_dir = Path("data")
    in_dir: Path = data_dir / "merged"
    out_dir: Path = data_dir / "refined"
    cleanup(out_dir)

    name_list: list[str] = sorted(
        (x.stem[:4] for x in in_dir.iterdir() if x.stem.endswith("_tops")), key=lambda x: int(x)
    )

    with Pool() as p:
        results: list[ApplyResult] = [
            p.apply_async(work, (in_dir, out_dir, name, pid)) for pid, name in enumerate(name_list)
        ]

        for result in results:
            print(f"PROCESS {result.get()} ENDED.", flush=True)
