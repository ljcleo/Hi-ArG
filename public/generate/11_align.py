from argparse import ArgumentParser
from collections import defaultdict
from dataclasses import dataclass
from json import dumps
from multiprocessing.pool import ApplyResult, Pool
from pathlib import Path
from pickle import dump, load

from amrlib.alignments.faa_aligner import FAA_Aligner
from amrlib.alignments.rbw_aligner import RBWAligner
from penman import decode, encode
from penman.constant import evaluate
from penman.constant import type as const_type
from penman.epigraph import Epidatum
from penman.graph import Graph, Triple, Variable
from penman.layout import Pop
from penman.models.noop import NoOpModel
from penman.surface import AlignmentMarker
from utils.file import cleanup


@dataclass
class AlignArguments:
    use_boa: bool = False
    split_size: int = 4096


def get_rbw_graph(graph: Graph) -> Graph:
    try:
        return RBWAligner.from_penman_w_json(graph).get_penman_graph()
    except Exception as e:
        print("RBW", type(e), e)
        return graph


def is_starter(snt: str) -> bool:
    return snt.startswith('"') and (snt.endswith('" is right.') or snt.endswith('" is wrong.'))


def extract_alignments(graph: Graph, triple: Triple) -> set[int]:
    return set(
        sum(
            (
                epidatum.indices
                for epidatum in graph.epidata.get(triple, [])
                if isinstance(epidatum, AlignmentMarker)
            ),
            (),
        )
    )


def mark_const(attribute: Triple) -> tuple[str, str]:
    target: str = attribute.target
    evaled: str = evaluate(target)
    return f"CONST#{const_type(target).name}#{target}#{evaled}", str(evaled)


def to_acyclic(graph: Graph) -> Graph:
    edges: list[Triple] = []
    status: dict[Variable, int] = {var: 0 for var in graph.variables()}

    alignments: defaultdict[str, set[int]] = defaultdict(set)
    for instance in graph.instances():
        alignments[instance.source] |= extract_alignments(graph, instance)

    for attribute in graph.attributes():
        mark: str = mark_const(attribute)[0]
        alignments[mark] |= extract_alignments(graph, attribute)

    def dfs(source: Variable) -> None:
        status[source] = 1

        for edge in graph.edges(source=source):
            target: Variable = edge.target
            if target == source:
                continue

            edge_alignments: set[int] = extract_alignments(graph, edge)
            alignments[source] |= edge_alignments
            alignments[target] |= edge_alignments

            if status[target] == 1:
                edges.append(NoOpModel().invert(edge))
            else:
                edges.append(edge)
                if status[target] == 0:
                    dfs(target)

        status[source] = 2

    dfs(graph.top)

    graph.metadata["alignments"] = dumps(
        {key: dumps(list(value)) for key, value in alignments.items()}
    )

    return Graph(
        graph.instances() + graph.attributes() + edges, top=graph.top, metadata=graph.metadata
    )


def work(source: Path) -> list[Graph]:
    print(f"\nWORKING ON {source} ...\n")
    faa_model = FAA_Aligner()
    noop = NoOpModel()

    with source.open("rb") as f:
        graphs: list[Graph] = load(f)

    graph_strings: list[str] = [encode(graph, model=noop) for graph in graphs]

    rbw_graphs: list[Graph] = [
        get_rbw_graph(decode(graph_str, model=noop)) for graph_str in graph_strings
    ]

    token_snt: list[str] = [" ".join(eval(graph.metadata["tokens"])) for graph in rbw_graphs]
    faa_strings: list[str]

    try:
        faa_strings = faa_model.align_sents(token_snt, graph_strings)[0]
    except Exception:
        faa_strings = []

        for snt, graph_str in zip(token_snt, graph_strings):
            try:
                graph_str = faa_model.align_sents([snt], [graph_str])[0][0]
            except Exception as e:
                print("FAA", type(e), e)

            faa_strings.append(graph_str)

    good_graphs: list[Graph] = []

    for rbw_graph, faa_str in zip(rbw_graphs, faa_strings):
        rbw_graph.metadata.pop("lemmas", "")
        rbw_graph.metadata.pop("alignments", "")

        try:
            faa_graph: Graph = decode(faa_str, model=noop)

            for triple, epidata in faa_graph.epidata.items():
                if triple not in rbw_graph.epidata:
                    continue

                new_epidata: list[Epidatum] = [x for x in epidata if isinstance(x, AlignmentMarker)]

                for re in rbw_graph.epidata[triple]:
                    if isinstance(re, Pop):
                        new_epidata.insert(0, re)
                    elif re not in new_epidata:
                        new_epidata.append(re)

                rbw_graph.epidata[triple] = new_epidata
        except Exception:
            pass

        good_graphs.append(to_acyclic(rbw_graph))

    print(f"\n{source} PROCESSED.\n")
    return good_graphs


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-b", "--use-boa", action="store_true")
    parser.add_argument("-s", "--split-size", type=int)
    args: AlignArguments = parser.parse_args(namespace=AlignArguments())

    data_dir = Path("data")
    in_dir: Path = data_dir / "parsed"
    out_dir: Path = data_dir / "aligned"
    cleanup(out_dir)

    files: list[Path] = sorted(in_dir.iterdir(), key=lambda x: int(x.stem))
    buffer: list[Graph] = []
    fid: int = 0
    aid: int = -args.use_boa
    sid: int = 0

    with Pool() as pool:
        results: list[ApplyResult] = [pool.apply_async(work, (source,)) for source in files]

        for i, result in enumerate(results):
            graphs: list[Graph] = result.get()
            print(f"\nRESULT {i:05d} RETRIEVED.\n")

            for graph in graphs:
                if args.use_boa and is_starter(graph.metadata["snt"]):
                    aid += 1
                    sid = 0

                graph.metadata["id"] = f"{aid:07d}-{sid:04d}"
                buffer.append(graph)
                sid += 1

                if len(buffer) >= args.split_size:
                    with (out_dir / f"{fid:04d}.pkl").open("wb") as f:
                        dump(buffer, f)

                    buffer = []
                    fid += 1

                    if not args.use_boa:
                        aid += 1
                        sid = 0

    if len(buffer) > 0:
        with (out_dir / f"{fid:04d}.pkl").open("wb") as f:
            dump(buffer, f)
