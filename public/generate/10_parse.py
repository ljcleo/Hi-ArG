from argparse import ArgumentParser
from dataclasses import dataclass
from multiprocessing import Manager
from multiprocessing.pool import ApplyResult, Pool
from multiprocessing.queues import Queue
from pathlib import Path
from pickle import dump
from queue import Empty
from typing import Optional

from amrlib import load_stog_model
from amrlib.graph_processing.annotator import add_lemmas
from amrlib.models.inference_bases import STOGInferenceBase
from penman import DecodeError
from penman.graph import Graph, Triple, Variable
from penman.models.noop import NoOpModel
from utils.file import cleanup


@dataclass
class ParseArgs:
    restart: bool = False
    use_gpu: bool = False
    num_proc: int = 4
    batch_size: int = 12
    num_beams: int = 4


def to_acyclic(graph: Graph) -> Graph:
    model = NoOpModel()
    edges: list[Triple] = []
    status: dict[Variable, int] = {var: 0 for var in graph.variables()}

    def dfs(source: Variable) -> None:
        status[source] = 1

        for edge in graph.edges(source=source):
            target: Variable = edge.target
            if target == source:
                continue

            if status[target] == 1:
                edges.append(model.invert(edge))
            else:
                edges.append(edge)
                if status[target] == 0:
                    dfs(target)

        status[source] = 2

    dfs(graph.top)

    return Graph(
        graph.instances() + graph.attributes() + edges, top=graph.top, metadata=graph.metadata
    )


def work(pid: int, queue: Queue, out_dir: Path, args: ParseArgs) -> int:
    model: STOGInferenceBase = load_stog_model(
        device=f"cuda:{pid}" if args.use_gpu else "cpu",
        batch_size=args.batch_size,
        num_beams=args.num_beams,
    )

    name: str = f"WORKER {pid:02d}"
    file_count: int = 0
    snt_count: int = 0
    good: int = 0

    while True:
        try:
            in_file: Path = queue.get(timeout=3)
            file_count += 1
            graphs: list[Graph] = []

            with in_file.open("r", encoding="utf8") as f:
                lines: list[str] = [x.strip() for x in f]
                snt_count += len(lines)

            for i in range(0, len(lines), args.batch_size):
                for graph in model.parse_sents(lines[i : i + args.batch_size]):
                    graph: Optional[str]
                    if graph is None:
                        continue

                    try:
                        penman: Graph = add_lemmas(graph, snt_key="snt")
                    except DecodeError:
                        continue

                    if penman.top is not None:
                        graphs.append(to_acyclic(penman))
                        good += 1

            with (out_dir / f"{in_file.stem}.pkl").open("wb") as f:
                dump(graphs, f)

            print(f"[{name}] COUNT {file_count}/{snt_count} GOOD {good}")
        except Empty:
            break

    return good


if __name__ == "__main__":
    parser = ArgumentParser(description="2_parse")
    parser.add_argument("-r", "--restart", action="store_true")
    parser.add_argument("-g", "--use-gpu", action="store_true")
    parser.add_argument("-p", "--num-proc", default=4, type=int)
    parser.add_argument("-b", "--batch-size", default=12, type=int)
    parser.add_argument("-n", "--num-beams", default=4, type=int)
    args: ParseArgs = parser.parse_args(namespace=ParseArgs())

    data_dir = Path("data")
    in_dir: Path = data_dir / "arranged"
    out_dir: Path = data_dir / "parsed"

    if args.restart:
        cleanup(out_dir)

    with Manager() as manager:
        excluded: set[str] = {x.stem for x in out_dir.iterdir()}
        files: list[Path] = [f for f in in_dir.iterdir() if f.stem not in excluded]
        files.sort(key=lambda x: x.stem)
        queue: Queue = manager.Queue(maxsize=len(files))

        for in_file in files:
            queue.put(in_file)

        with Pool(processes=args.num_proc) as pool:
            results: list[ApplyResult] = [
                pool.apply_async(work, args=(pid, queue, out_dir, args))
                for pid in range(args.num_proc)
            ]

            for result in results:
                print(f"OUTPUT {result.get()} GRAPHS")
