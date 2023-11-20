from hashlib import sha256
from multiprocessing.pool import ApplyResult, Pool
from os import cpu_count, getpid, system
from pathlib import Path

import numpy as np


def cache_array(source_file: Path, target_dir: Path) -> str:
    system(f"taskset -p 0xffffffffffffffff {getpid()}")
    path: str = source_file.absolute().as_posix()
    key: str = sha256(path.encode("utf8")).hexdigest()
    print(f"caching {path} ...")

    with np.load(source_file) as data:
        arrays: dict[str, np.ndarray] = {k: data[k] for k in ("data", "data_start", "data_end")}

    for name, array in arrays.items():
        sub_key: str = f"{key}-{name}"

        with (target_dir / f"{sub_key}-data").open("wb") as f:
            f.write(np.ascontiguousarray(array).data)
        with (target_dir / f"{sub_key}-shape").open("wb") as f:
            f.write(np.array(array.shape).data)
        with (target_dir / f"{sub_key}-dtype").open("wb") as f:
            f.write(str(array.dtype).encode("utf8"))

    return path


if __name__ == "__main__":
    data_dir = Path("data")
    target_dir = Path("cache")
    target_dir.mkdir(exist_ok=True)

    source_files: list[Path] = [
        source_file
        for dataset_dir in data_dir.iterdir()
        for split_dir in dataset_dir.iterdir()
        for source_file in split_dir.iterdir()
    ]

    with Pool(processes=round(cpu_count() * 0.95)) as p:
        results: list[ApplyResult] = [
            p.apply_async(cache_array, (source_file, target_dir)) for source_file in source_files
        ]

        for result in results:
            print(f"{result.get()} cached.")
