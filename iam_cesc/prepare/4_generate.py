from multiprocessing.pool import ApplyResult, Pool
from pathlib import Path
from pickle import dump

import numpy as np
from utils.args import get_target_dataset
from utils.data import IAMDataset

if __name__ == "__main__":
    data_dir: Path = Path("data")
    source_dir: Path = data_dir / "simplified"
    target_dir: Path = data_dir / "generated"
    target_dir.mkdir(exist_ok=True)
    target_dataset: str | None = get_target_dataset()

    for dataset_dir in source_dir.iterdir():
        dataset_name: str = dataset_dir.stem

        if not dataset_dir.is_dir():
            continue
        if target_dataset is not None and dataset_name != target_dataset:
            continue

        train_set: IAMDataset
        dev_set: IAMDataset
        test_set: IAMDataset
        train_set, dev_set, test_set = IAMDataset.load_train_dev_test(dataset_name)
        print(f"{dataset_name} datasets loaded.")

        target_sub_dir: Path = target_dir / dataset_name
        target_sub_dir.mkdir(exist_ok=True)

        for mode, dataset in (("train", train_set), ("dev", dev_set), ("test", test_set)):

            def get(index: int) -> dict[str, np.ndarray]:
                return dataset[index]

            with Pool() as pool:
                results: list[ApplyResult] = [
                    pool.apply_async(get, (i,)) for i in range(len(dataset))
                ]

                samples: list[dict[str, np.ndarray]] = [result.get() for result in results]

            print(f"{dataset_name} {mode} samples generated.")

            with (target_sub_dir / f"{mode}.pkl").open("wb") as f:
                dump(samples, f)

            print(f"{dataset_name} {mode} samples dumped.")
