from multiprocessing.pool import ApplyResult, Pool
from os import cpu_count, getpid, system
from pathlib import Path

import numpy as np
from utils.args import get_target_dataset


def merge(source_dir: Path, target_dir: Path, meta: tuple[str, ...]) -> str:
    system(f"taskset -p 0xffffffffffffffff {getpid()}")
    key: str = source_dir.stem
    all_data: list[np.ndarray] = []
    all_len: list[np.ndarray] = []

    for file in sorted(source_dir.iterdir()):
        with np.load(file) as data:
            all_data.append(data["data"])
            all_len.append(data["len"])

        print(f"{key} {file.name} loaded.", flush=True)

    data: np.ndarray = np.concatenate(all_data)
    print(f"{key} merged.", flush=True)

    data_len: np.ndarray = np.concatenate(all_len)
    data_end: np.ndarray = np.cumsum(data_len)
    data_start: np.ndarray = np.concatenate([[0], data_end[:-1]])
    print(f"{key} bound calculated.", flush=True)

    np.savez_compressed(
        target_dir / f"{key}.npz", data=data, data_start=data_start, data_end=data_end
    )

    print(f"{key} dumped.", flush=True)
    return " ".join(meta + (key,))


if __name__ == "__main__":
    data_dir: Path = Path("data")
    source_dir: Path = data_dir / "calculated"
    target_dir: Path = data_dir / "prepared"
    target_dir.mkdir(exist_ok=True)
    target_dataset: str | None = get_target_dataset()

    with Pool(processes=round(cpu_count() * 0.95)) as pool:
        results: list[ApplyResult] = []

        for dataset_dir in source_dir.iterdir():
            dataset_name: str = dataset_dir.stem
            if target_dataset is not None and dataset_name != target_dataset:
                continue

            target_sub_dir: Path = target_dir / dataset_name
            target_sub_dir.mkdir(exist_ok=True)

            for flag in range(2):
                prompt: str = "joint" if flag else "text"

                for split in ("eval", "train"):
                    sub_name: str = f"{split}_{flag}"
                    sub_dir: Path = dataset_dir / sub_name
                    target_split_dir: Path = target_sub_dir / sub_name
                    target_split_dir.mkdir(exist_ok=True)

                    results.extend(
                        [
                            pool.apply_async(
                                merge, (dir, target_split_dir, (dataset_name, prompt, split))
                            )
                            for dir in sub_dir.iterdir()
                        ]
                    )

        for result in results:
            print(f"{result.get()} prepared.", flush=True)
