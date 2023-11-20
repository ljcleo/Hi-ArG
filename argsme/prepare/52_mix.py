from multiprocessing.pool import ApplyResult, Pool
from os import cpu_count, getpid, system
from pathlib import Path

import numpy as np
from utils.args import get_target_dataset


def mix(source_dirs: list[Path], target_dir: Path, key: str, meta: tuple[str, ...]) -> str:
    system(f"taskset -p 0xffffffffffffffff {getpid()}")
    all_data: list[np.ndarray] = []
    all_data_start: list[np.ndarray] = []
    all_data_end: list[np.ndarray] = []

    for dir in source_dirs:
        with np.load(dir / f"{key}.npz") as data:
            all_data.append(data["data"])
            all_data_start.append(data["data_start"])
            all_data_end.append(data["data_end"])
            assert all_data[-1].shape[0] == all_data_end[-1][-1]

        print(f"{key} from {dir.name} loaded.", flush=True)

    samples: list[np.ndarray] = []
    sample_len: list[int] = []

    for i in range(all_data_start[0].shape[0]):
        j: int = i % len(all_data)
        cur_start: int = all_data_start[j][i].item()
        cur_end: int = all_data_end[j][i].item()
        samples.append(all_data[j][cur_start:cur_end])
        sample_len.append(cur_end - cur_start)

    data: np.ndarray = np.concatenate(samples)
    print(f"{key} mixed.", flush=True)

    data_end: np.ndarray = np.cumsum(sample_len)
    data_start: np.ndarray = np.concatenate([[0], data_end[:-1]])
    print(f"{key} mixed bound calculated.", flush=True)

    np.savez_compressed(
        target_dir / f"{key}.npz", data=data, data_start=data_start, data_end=data_end
    )

    print(f"{key} dumped.", flush=True)
    return " ".join(meta + (key,))


if __name__ == "__main__":
    data_dir: Path = Path("data") / "prepared"
    target_dataset: str | None = get_target_dataset()

    with Pool(processes=round(cpu_count() * 0.95)) as pool:
        results: list[ApplyResult] = []

        for dataset_dir in data_dir.iterdir():
            dataset_name: str = dataset_dir.stem
            if target_dataset is not None and dataset_name != target_dataset:
                continue

            for split in ("eval", "train"):
                source_dir: list[Path] = [dataset_dir / f"{split}_{flag}" for flag in range(2)]
                keys: list[str] = sorted([x.stem for x in source_dir[0].iterdir()])
                target_dir: Path = dataset_dir / f"{split}_2"
                target_dir.mkdir(exist_ok=True)

                results.extend(
                    [
                        pool.apply_async(mix, (source_dir, target_dir, key, (dataset_name, split)))
                        for key in keys
                    ]
                )

        for result in results:
            print(f"{result.get()} mixed.", flush=True)
