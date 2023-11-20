from collections import defaultdict
from multiprocessing.pool import ApplyResult, Pool
from os import cpu_count, getpid, system
from pathlib import Path

import numpy as np
from utils.args import get_target_dataset
from utils.data import AKGDataset


def prepare(
    pid: int,
    dataset_tag: str,
    flag: bool,
    indices: np.ndarray,
    out_dir: Path,
    meta: tuple[str, ...],
) -> str:
    system(f"taskset -p 0xffffffffffffffff {getpid()}")
    dataset: AKGDataset = AKGDataset.from_shm(dataset_tag)
    dataset.max_comment_length = 128 if flag else 0

    all_info: defaultdict[str, list[np.ndarray]] = defaultdict(list)
    all_len: defaultdict[str, list[int]] = defaultdict(list)
    checkpoints: set[int] = {round(x) for x in np.linspace(0, indices.shape[0], 21)}
    print(f"[{pid:02d}] start working ...", flush=True)

    for count, index in enumerate(indices):
        info: dict[str, np.ndarray] = dataset[index]

        for k, v in info.items():
            all_info[k].append(v)
            all_len[k].append(v.shape[0])

        if count + 1 in checkpoints:
            print(f"[{pid:02d}] {count + 1} data prepared.", flush=True)

    print(f"[{pid:02d}] all data prepared.", flush=True)

    for k, v in all_info.items():
        sub_dir: Path = out_dir / k
        sub_dir.mkdir(exist_ok=True)

        np.savez_compressed(
            sub_dir / f"{pid:02d}.npz", data=np.concatenate(v, axis=0), len=np.array(all_len[k])
        )

    print(f"[{pid:02d}] all data dumped.", flush=True)
    return " ".join(meta + (f"{pid:02d}",))


if __name__ == "__main__":
    data_dir: Path = Path("data")
    source_dir: Path = data_dir / "simplified"
    target_dir: Path = data_dir / "calculated"
    target_dir.mkdir(exist_ok=True)
    target_dataset: str | None = get_target_dataset()

    for dataset_dir in source_dir.iterdir():
        dataset_name: str = dataset_dir.stem

        if not dataset_dir.is_dir():
            continue
        if target_dataset is not None and dataset_name != target_dataset:
            continue

        train_set: AKGDataset
        eval_set: AKGDataset
        train_set, eval_set = AKGDataset.load_train_eval(dataset_name)
        print(f"{dataset_name} datasets loaded.", flush=True)

        train_set_len: int = len(train_set)
        eval_set_len: int = len(eval_set)
        train_set_tag: str = train_set.to_shm()
        eval_set_tag: str = eval_set.to_shm()
        print(f"{dataset_name} datasets cached.", flush=True)

        target_sub_dir: Path = target_dir / dataset_name
        target_sub_dir.mkdir(exist_ok=True)

        with Pool(processes=round(cpu_count() * 0.95)) as pool:
            results: list[ApplyResult] = []

            for flag in range(2):
                prompt: str = "joint" if flag else "text"

                for split, dataset_len, dataset_tag in (
                    ("eval", eval_set_len, eval_set_tag),
                    ("train", train_set_len, train_set_tag),
                ):
                    chunk_size: int = 65536
                    pivot: np.ndarray = np.arange(0, dataset_len, chunk_size)
                    print(f"{dataset_name} {prompt} {split} indices prepared.", flush=True)

                    target_split_dir: Path = target_sub_dir / f"{split}_{flag}"
                    target_split_dir.mkdir(exist_ok=True)

                    results.extend(
                        [
                            pool.apply_async(
                                prepare,
                                (
                                    i,
                                    dataset_tag,
                                    bool(flag),
                                    np.arange(p, min(p + chunk_size, dataset_len)),
                                    target_split_dir,
                                    (dataset_name, prompt, split),
                                ),
                            )
                            for i, p in enumerate(pivot)
                        ]
                    )

            for result in results:
                print(f"{result.get()} finished.", flush=True)

        for tag in (eval_set_tag, train_set_tag):
            AKGDataset.clean_shm(tag)
