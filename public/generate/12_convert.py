from json import loads
from multiprocessing.pool import ApplyResult, Pool
from pathlib import Path
from pickle import load

import pandas as pd
from penman.constant import evaluate
from penman.constant import type as const_type
from penman.graph import Graph, Triple
from utils.file import cleanup
from utils.io import dump_graphs
from utils.pool import NestablePool


def mark_const(attribute: Triple) -> tuple[str, str]:
    target: str = attribute.target
    evaled: str = evaluate(target)
    return f"CONST#{const_type(target).name}#{target}#{evaled}", str(evaled)


def post_mark(
    nodes: pd.DataFrame, edges: pd.DataFrame, aligns: pd.DataFrame, tops: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    node_text: pd.Series = nodes.set_index("id")["text"]
    node_repr: dict[str, str] = node_text.to_dict()

    for edge in edges.rename(columns={"from": "u", "to": "v", "type": "t"}).itertuples(index=False):
        node_repr[edge.u] = f"{node_repr[edge.u]}:{node_text[edge.v]}|{edge.t}"

    counter: dict[str, int] = {}
    new_nodes: list[dict[str, str]] = []

    for node, r in node_repr.items():
        cur_text: str = node_text[node]

        if not r.startswith("CONST#"):
            if r in counter:
                counter[r] += 1
                cur_text = f"{cur_text}${counter[r]}"
            else:
                counter[r] = 0

        new_nodes.append({"id": node, "text": cur_text})

    return pd.DataFrame(new_nodes), edges, aligns, tops


def graph_to_pandas(graph: Graph) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    instance_map: dict[str, str] = {
        instance.source: instance.target for instance in graph.instances()
    }

    const_map: dict[str, tuple[str, str]] = {
        attribute.target: mark_const(attribute) for attribute in graph.attributes()
    }

    nodes: list[dict[str, str]] = [
        {"id": const_id, "text": const_value} for const_id, const_value in const_map.values()
    ]

    edges: list[dict[str, str]] = []

    for variable in graph.variables():
        nodes.append({"id": variable, "text": instance_map[variable]})

        edges.extend(
            [
                {"from": edge.source, "to": edge.target, "type": edge.role[1:]}
                for edge in graph.edges(source=variable)
            ]
        )

        edges.extend(
            [
                {
                    "from": attribute.source,
                    "to": const_map[attribute.target][0],
                    "type": attribute.role[1:],
                }
                for attribute in graph.attributes(source=variable)
            ]
        )

    cur_id: str = graph.metadata["id"]
    for node in nodes:
        node["id"] = f"{cur_id}-{node['id']}"

    for edge in edges:
        edge["from"] = f"{cur_id}-{edge['from']}"
        edge["to"] = f"{cur_id}-{edge['to']}"

    try:
        aligns: list[dict[str, str]] = [
            {"id": cur_id, "node": f"{cur_id}-{node}", "aligns": alignments}
            for node, alignments in loads(graph.metadata["alignments"]).items()
        ]
    except Exception:
        print(graph.metadata["alignments"])
        raise

    return post_mark(
        pd.DataFrame(nodes),
        pd.DataFrame(edges).drop_duplicates(),
        pd.DataFrame(aligns),
        pd.DataFrame(
            [
                {
                    "id": cur_id,
                    "snt": graph.metadata["snt"],
                    "top": f"{cur_id}-{graph.top}",
                    "tokens": graph.metadata["tokens"],
                }
            ]
        ),
    )


def work(in_file: Path, out_dir: Path) -> int:
    name: str = f"{int(in_file.stem):04d}"
    with in_file.open("rb") as f:
        graphs: list[Graph] = load(f)

    print(f"GROUP {name} LOADED.", flush=True)
    nodes: pd.DataFrame
    edges: pd.DataFrame
    aligns: pd.DataFrame
    tops: pd.DataFrame

    with Pool(processes=8) as pool:
        results: list[ApplyResult] = [
            pool.apply_async(graph_to_pandas, (graph,)) for graph in graphs
        ]

        nodes, edges, aligns, tops = tuple(
            pd.concat(data_list, ignore_index=True)
            for data_list in zip(*[result.get() for result in results])
        )

    print(f"GROUP {name} GENERATED.", flush=True)
    dump_graphs(nodes, edges, aligns, tops, out_dir, name)
    print(f"GROUP {name} DUMPED.", flush=True)
    return int(name)


if __name__ == "__main__":
    data_dir = Path("data")
    in_dir: Path = data_dir / "aligned"
    out_dir: Path = data_dir / "converted"
    cleanup(out_dir)
    file_list: list[Path] = sorted(in_dir.iterdir(), key=lambda x: int(x.stem))

    with NestablePool(processes=24) as p:
        results: list[ApplyResult] = [
            p.apply_async(work, (in_file, out_dir)) for in_file in file_list
        ]

        for result in results:
            print(f"PROCESS {result.get()} ENDED.", flush=True)
