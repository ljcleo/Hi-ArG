from pathlib import Path
from shutil import copy

import pandas as pd


def fix_surrogate(raw: str) -> str:
    return raw.encode("utf8", "replace").decode("utf8", "ignore")


def get_filename(name: str, component: str) -> str:
    stem: str = name

    if component != "":
        if name == "":
            stem = component
        else:
            stem = f"{name}_{component}"

    return f"{stem}.bin"


def load_component(in_dir: Path, name: str, component: str) -> pd.DataFrame:
    return pd.read_parquet(in_dir / get_filename(name, component))


def dump_component(
    data: pd.DataFrame, out_dir: Path, name: str, component: str
) -> None:
    data.to_parquet(out_dir / get_filename(name, component))


def copy_component(
    in_dir: Path, out_dir: Path, name: str, component: str
) -> None:
    filename: str = get_filename(name, component)
    copy(in_dir / filename, out_dir / filename)


def load_graphs(
    in_dir: Path, name: str
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return tuple(
        load_component(in_dir, name, component)
        for component in ("nodes", "edges", "aligns", "tops")
    )


def dump_graphs(
    nodes: pd.DataFrame, edges: pd.DataFrame, aligns: pd.DataFrame,
    tops: pd.DataFrame, out_dir: Path, name: str
) -> None:
    try:
        dump_component(nodes, out_dir, name, "nodes")
    except UnicodeEncodeError:
        nodes["text"] = nodes["text"].apply(fix_surrogate)
        dump_component(nodes, out_dir, name, "nodes")

    try:
        dump_component(edges, out_dir, name, "edges")
    except UnicodeEncodeError:
        edges["type"] = edges["type"].apply(fix_surrogate)
        dump_component(edges, out_dir, name, "edges")

    dump_component(aligns, out_dir, name, "aligns")

    try:
        dump_component(tops, out_dir, name, "tops")
    except UnicodeEncodeError:
        tops["snt"] = tops["snt"].apply(fix_surrogate)
        if "tokens" in tops:
            tops["tokens"] = tops["tokens"].apply(fix_surrogate)
        dump_component(tops, out_dir, name, "tops")
