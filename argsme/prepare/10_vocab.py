from pathlib import Path

import pandas as pd
from utils.io import load_component

if __name__ == "__main__":
    data_dir = Path("data")
    graph_dir: Path = data_dir / "paired"
    vocab_dir: Path = data_dir / "vocab"

    with (vocab_dir / "predicates.txt").open() as f:
        candidates: set[str] = {x.strip() for x in f}
    with (vocab_dir / "additions.txt").open() as f:
        candidates |= {x.strip() for x in f}

    actual: set[str] = set()

    for dataset in graph_dir.iterdir():
        nodes: pd.DataFrame = load_component(dataset, "", "nodes")
        edges: pd.DataFrame = load_component(dataset, "", "edges")

        texts: pd.Series = pd.concat(
            [
                nodes["text"].cat.categories.to_series(),
                edges["type"].cat.categories.to_series().map(lambda x: f":{x}"),
            ]
        )

        texts = texts[texts.isin(candidates)]
        actual |= set(texts.to_list())

    actual |= {f"{x}-of" for x in actual if x[0] == ":" and not x.endswith("-of")}
    actual |= {x[:-3] for x in actual if x[0] == ":" and x.endswith("-of")}
    out: list[str] = list(actual)
    out.sort()

    with (vocab_dir / "new_vocab.txt").open("w", encoding="utf8") as f:
        for token in out:
            print(token, file=f)
