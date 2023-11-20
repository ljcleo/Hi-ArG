from multiprocessing.pool import ApplyResult, Pool
from pathlib import Path

import pandas as pd
from utils.file import cleanup
from utils.io import dump_graphs, load_graphs
from utils.merge import merge


def work(in_dir: Path, out_dir: Path, name: str, pid: int) -> int:
    print(f"PROCESS {pid} STARTED.", flush=True)

    nodes: pd.DataFrame
    edges: pd.DataFrame
    aligns: pd.DataFrame
    tops: pd.DataFrame
    nodes, edges, aligns, tops = load_graphs(in_dir, name)
    print(f"[{pid:05d}] GROUP {name} LOADED.", flush=True)

    nodes, edges, aligns, tops = merge(nodes, edges, aligns, tops, parallel=False, pid=pid)
    print(f"[{pid:05d}] GROUP {name} MERGED.", flush=True)

    dump_graphs(nodes, edges, aligns, tops, out_dir, name)
    print(f"[{pid:05d}] GROUP {name} DUMPED.", flush=True)
    return pid


if __name__ == "__main__":
    data_dir = Path("data")
    in_dir: Path = data_dir / "converted"
    out_dir: Path = data_dir / "merged"
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
