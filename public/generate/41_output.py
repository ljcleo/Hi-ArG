from pathlib import Path

import pandas as pd
from utils.file import cleanup
from utils.io import dump_graphs, load_graphs
from utils.merge import merge

if __name__ == "__main__":
    data_dir = Path("data")
    in_dir: Path = data_dir / "combined"
    out_dir: Path = data_dir / "output"
    cleanup(out_dir)

    nodes: pd.DataFrame
    edges: pd.DataFrame
    aligns: pd.DataFrame
    tops: pd.DataFrame
    nodes, edges, aligns, tops = load_graphs(in_dir, "")
    print("graph loaded.")

    nodes, edges, aligns, tops = merge(nodes, edges, aligns, tops)
    print("graph merged.")

    dump_graphs(nodes, edges, aligns, tops, out_dir, "")
    print("graph dumped.")
