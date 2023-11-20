from multiprocessing.managers import SharedMemoryManager, SyncManager
from multiprocessing.pool import ApplyResult, Pool
from multiprocessing.shared_memory import SharedMemory
from os import cpu_count
from queue import Empty, Queue
from time import sleep

import numpy as np
import pandas as pd
from pandarallel import pandarallel

shm_info = tuple[str, tuple[int, ...], np.dtype]


def convert_to_shm(data: np.ndarray, smm: SharedMemoryManager) -> shm_info:
    shm: SharedMemory = smm.SharedMemory(data.nbytes)
    converted = np.ndarray(data.shape, dtype=data.dtype, buffer=shm.buf)
    np.copyto(converted, data)
    return shm.name, converted.shape, converted.dtype


def fill_jobs(num_nodes: int, job_queue: Queue, result_queue: Queue, outer_pid: int) -> int:
    try:
        print(f"[{outer_pid:05d}] job filling started.", flush=True)
        step: int = 16384

        for i in range(0, num_nodes, step):
            job_queue.put((i, min(i + step, num_nodes)))

        print(f"[{outer_pid:05d}] job filling completed.", flush=True)
        return -1
    except Exception:
        result_queue.put(-1)
        raise


def gen_node_repr_batch(
    node_text_info: shm_info,
    edge_to_info: shm_info,
    edge_type_info: shm_info,
    edge_split_pos_info,
    job_queue: Queue,
    result_queue: Queue,
    outer_pid: int,
    inner_pid: int,
) -> int:
    sleep(1)
    print(f"[{outer_pid:05d}] process {inner_pid} started.", flush=True)
    count = 0

    def recover(info: shm_info) -> tuple[SharedMemory, np.ndarray]:
        shm = SharedMemory(name=info[0])
        return shm, np.ndarray(info[1], dtype=info[2], buffer=shm.buf)

    node_text_shm: SharedMemory | None = None
    edge_to_shm: SharedMemory | None = None
    edge_type_shm: SharedMemory | None = None
    edge_split_pos_shm: SharedMemory | None = None
    node_text: np.ndarray | None = None
    edge_to: np.ndarray | None = None
    edge_type: np.ndarray | None = None
    edge_split_pos: np.ndarray | None = None

    try:
        node_text_shm, node_text = recover(node_text_info)
        edge_to_shm, edge_to = recover(edge_to_info)
        edge_type_shm, edge_type = recover(edge_type_info)
        edge_split_pos_shm, edge_split_pos = recover(edge_split_pos_info)

        print(f"[{outer_pid:05d}] process {inner_pid} array recovered.", flush=True)

        while True:
            node_id: list[int] = []
            edge_repr: list[str] = []

            try:
                interval: tuple[int, int] = job_queue.get(timeout=3)
            except Empty:
                break

            for i in range(*interval):
                node_id.append(i)
                parts: list[str] = [f"{node_text[i]}"]

                parts.extend(
                    [
                        f"{edge_to[j]}|{edge_type[j]}"
                        for j in range(edge_split_pos[i], edge_split_pos[i + 1])
                    ]
                )

                edge_repr.append(":".join(parts))

            result_queue.put(pd.Series(edge_repr, index=node_id, name="edge_repr"))

            count += 1

        print(f"[{outer_pid:05d}] process {inner_pid} {count} jobs finished.", flush=True)
        result_queue.put(inner_pid)
        return inner_pid
    except Exception:
        result_queue.put(inner_pid)
        raise
    finally:
        for shm in (node_text_shm, edge_to_shm, edge_type_shm, edge_split_pos_shm):
            if shm is not None:
                shm.close()


def gen_node_repr(nodes: pd.DataFrame, edges: pd.DataFrame, parallel: bool, pid: int) -> pd.Series:
    edge_pivot: pd.Series = edges["from"] != edges["from"].shift(1)
    edge_count = np.diff(edge_pivot.to_numpy().nonzero()[0], append=edges.shape[0])
    num_nodes: int = nodes.shape[0]
    node_edge_count: np.ndarray = np.zeros(num_nodes, dtype=int)
    node_edge_count[edges["from"][edge_pivot]] = edge_count
    edge_split_pos: np.ndarray = np.append([0], np.cumsum(node_edge_count))
    print(f"[{pid:05d}] edge split pos generated.", flush=True)

    if parallel:
        batch_results: list[pd.Series] = []
        num_workers: int | None = cpu_count()

        if num_workers is None:
            num_workers = 1

        with SharedMemoryManager() as smm:
            node_text_info: shm_info = convert_to_shm(nodes["text"].to_numpy(), smm)
            edge_to_info: shm_info = convert_to_shm(edges["to"].to_numpy(), smm)
            edge_type_info: shm_info = convert_to_shm(edges["type"].to_numpy(), smm)
            edge_split_pos_info: shm_info = convert_to_shm(edge_split_pos, smm)
            print(f"[{pid:05d}] shared lists prepared.", flush=True)

            with SyncManager() as sym:
                job_queue: Queue = sym.Queue()
                result_queue: Queue = sym.Queue()

                with Pool(processes=num_workers) as p:
                    responses: list[ApplyResult] = [
                        p.apply_async(fill_jobs, (num_nodes, job_queue, result_queue, pid))
                    ]

                    responses.extend(
                        [
                            p.apply_async(
                                gen_node_repr_batch,
                                (
                                    node_text_info,
                                    edge_to_info,
                                    edge_type_info,
                                    edge_split_pos_info,
                                    job_queue,
                                    result_queue,
                                    pid,
                                    inner_pid,
                                ),
                            )
                            for inner_pid in range(num_workers)
                        ]
                    )

                    while True:
                        try:
                            item: pd.Series | int = result_queue.get(timeout=30)
                        except Empty:
                            raise RuntimeError("too slow!")

                        if isinstance(item, pd.Series):
                            batch_results.append(item)
                        else:
                            responses[item + 1].get()
                            print(f"[{pid:05d}] process {item} finished.", flush=True)

                            if item >= 0:
                                num_workers -= 1

                            print(f"[{pid:05d}] {num_workers} workers left.", flush=True)
                            if num_workers <= 0:
                                break

        result: pd.Series = pd.concat(batch_results)
        result.sort_index(ignore_index=True, inplace=True)
        return result
    else:
        results: list[str] = []
        edge_repr: pd.Series = edges.apply(lambda x: f'{x["to"].item()}|{x["type"].item()}', axis=1)
        print(f"[{pid:05d}] edge representations generated.", flush=True)

        for i, text in enumerate(nodes["text"]):
            cur: list[str] = [f"{text}"]
            cur.extend(edge_repr[edge_split_pos[i] : edge_split_pos[i + 1]])
            results.append(":".join(cur))

        return pd.Series(results)


def merge(
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
    aligns: pd.DataFrame,
    tops: pd.DataFrame,
    parallel: bool = True,
    pid: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if parallel:
        pandarallel.initialize(progress_bar=False, verbose=1)

    cat_node_ids: pd.Series = nodes["id"].astype("category")
    node_id_cats: pd.Index = cat_node_ids.cat.categories
    cat_node_texts: pd.Series = nodes["text"].astype("category")
    node_text_cats: pd.Index = cat_node_texts.cat.categories
    cat_edge_types: pd.Series = edges["type"].astype("category")
    edge_type_cats: pd.Index = cat_edge_types.cat.categories
    print(f"[{pid:05d}] node and edge category generated.", flush=True)

    int_nodes = pd.DataFrame({"id": cat_node_ids.cat.codes, "text": cat_node_texts.cat.codes})
    int_nodes.sort_values("id", ignore_index=True, inplace=True)
    int_nodes.drop_duplicates("id", inplace=True, ignore_index=True)
    int_nodes.drop(columns="id", inplace=True)

    int_edges = pd.DataFrame(
        {
            "from": pd.Categorical(edges["from"], categories=node_id_cats).codes,
            "to": pd.Categorical(edges["to"], categories=node_id_cats).codes,
            "type": cat_edge_types.cat.codes,
        }
    )

    int_edges.sort_values(["from", "to", "type"], ignore_index=True, inplace=True)
    int_edges.drop_duplicates(inplace=True, ignore_index=True)
    int_top_tops = pd.Series(pd.Categorical(tops["top"], categories=node_id_cats).codes)
    int_align_nodes = pd.Series(pd.Categorical(aligns["node"], categories=node_id_cats).codes)
    print(f"[{pid:05d}] node and edge categorized.", flush=True)

    node_repr: pd.Series = gen_node_repr(int_nodes, int_edges, parallel, pid)
    print(f"[{pid:05d}] node representations generated.", flush=True)
    node_repr.name = "repr"
    main_dup_link: pd.Series | None = None

    while not node_repr.is_unique:
        dup_mark: pd.Series = node_repr.duplicated()
        keep_mark: pd.Series = node_repr.duplicated(keep=False) ^ dup_mark
        new_node_repr: pd.Series = node_repr[~dup_mark]
        dup_node_repr: pd.Series = node_repr[dup_mark]

        dup_link: pd.Series = (
            dup_node_repr.reset_index()
            .merge(node_repr[keep_mark].reset_index(), how="left", on="repr")
            .set_index("index_x")["index_y"]
        )

        def map_dup(prev_id: int) -> int:
            return dup_link.get(prev_id, prev_id)

        def update_node_repr(r: str) -> str:
            node_text: str
            edge_repr: list[str]
            node_text, *edge_repr = r.split(":")
            new_edge_repr: list[tuple[int, int]] = []

            for e in edge_repr:
                edge_to: str
                edge_type: str
                edge_to, edge_type = e.split("|")
                new_edge_repr.append((map_dup(int(edge_to)), int(edge_type)))

            parts: list[str] = [node_text]
            parts.extend([f"{x}|{y}" for x, y in sorted(set(new_edge_repr))])
            return ":".join(parts)

        if parallel:
            new_node_repr = new_node_repr.parallel_map(update_node_repr)
        else:
            new_node_repr = new_node_repr.map(update_node_repr)

        node_repr = new_node_repr

        if main_dup_link is None:
            main_dup_link = dup_link
        else:
            if parallel:
                main_dup_link = main_dup_link.parallel_map(map_dup)
            else:
                main_dup_link = main_dup_link.map(map_dup)

            main_dup_link = pd.concat([main_dup_link, dup_link])

        print(f"[{pid:05d}] # duplicate node: {dup_link.shape[0]}", flush=True)

    if main_dup_link is None:
        return nodes, edges, aligns, tops

    def main_map_dup(prev_id: int) -> int:
        return main_dup_link.get(prev_id, prev_id)

    int_nodes = int_nodes.loc[node_repr.index]
    int_edges = int_edges[~int_edges["from"].isin(main_dup_link.index)].copy()

    if parallel:
        int_edges["to"] = int_edges["to"].parallel_map(main_map_dup)
        int_top_tops = int_top_tops.parallel_map(main_map_dup)
        int_align_nodes = int_align_nodes.parallel_map(main_map_dup)
    else:
        int_edges["to"] = int_edges["to"].map(main_map_dup)
        int_top_tops = int_top_tops.map(main_map_dup)
        int_align_nodes = int_align_nodes.map(main_map_dup)

    int_edges.drop_duplicates(inplace=True, ignore_index=True)

    output_nodes: pd.DataFrame = int_nodes.reset_index(names="id")
    output_nodes["id"] = node_id_cats[output_nodes["id"]]
    output_nodes["text"] = node_text_cats[output_nodes["text"]]

    output_edges: pd.DataFrame = int_edges.copy()
    output_edges["from"] = node_id_cats[output_edges["from"]]
    output_edges["to"] = node_id_cats[output_edges["to"]]
    output_edges["type"] = edge_type_cats[output_edges["type"]]

    output_tops: pd.DataFrame = tops.copy()
    output_tops["top"] = node_id_cats[int_top_tops]

    output_aligns: pd.DataFrame = aligns.copy()
    output_aligns["node"] = node_id_cats[int_align_nodes]

    print(f"[{pid:05d}] deduplication applied.", flush=True)
    return output_nodes, output_edges, output_aligns, output_tops
