from hashlib import sha256
from multiprocessing import resource_tracker
from multiprocessing.shared_memory import SharedMemory
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


def get_external_shm(name: str) -> SharedMemory:
    shm = SharedMemory(name=name, create=False)
    resource_tracker.unregister(shm._name, "shared_memory")
    return shm


class StackedArray:
    def __init__(self, source_file: Path, cache_dir: Path, is_float: bool) -> None:
        self.source_file = source_file
        self.cache_dir = cache_dir
        self.is_float = is_float

        self._absolute_src: str = source_file.absolute().as_posix()
        self._key: str = sha256(self._absolute_src.encode("utf8")).hexdigest()

        self._data: np.ndarray | None = None
        self._start: np.ndarray | None = None
        self._end: np.ndarray | None = None

        self._shm_bank: dict[str, SharedMemory] = {}

    def __len__(self) -> int:
        if self._start is None:
            self._lazy_load()

        return self._start.shape[0]

    def __getitem__(self, index: int) -> np.ndarray:
        if self._start is None or self._end is None or self._data is None:
            self._lazy_load()

        start: int = self._start[index]
        end: int = self._end[index]
        return self._data[start:end].astype(np.float32 if self.is_float else int)

    def _lazy_load(self) -> None:
        print(f"loading {self.source_file.name} ...")
        self._data = self._load_array("data")
        self._start = self._load_array("data_start")
        self._end = self._load_array("data_end")

    def _load_array(self, name) -> np.ndarray:
        sub_key: str = f"{self._key}-{name}"
        arr_dtype: str = self._get_bytes(f"{sub_key}-dtype", True).decode("utf8")
        shape_bytes: bytes = self._get_bytes(f"{sub_key}-shape", True)
        arr_shape = np.ndarray(len(shape_bytes) >> 3, dtype=int, buffer=shape_bytes)

        return np.ndarray(
            arr_shape, dtype=arr_dtype, buffer=self._get_bytes(f"{sub_key}-data", False)
        )

    def _get_bytes(self, sec_key: str, return_bytes: bool) -> memoryview | bytes:
        try:
            data_shm: SharedMemory = get_external_shm(sec_key)
            self._shm_bank[sec_key] = data_shm

            if return_bytes:
                return data_shm.buf.tobytes()

            return data_shm.buf
        except Exception:
            try:
                with (self.cache_dir / sec_key).open("rb") as f:
                    return f.read()
            except Exception:
                raise RuntimeError(self.source_file, self._absolute_src, self._key)


def make_bool_array(data: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    if len(data.shape) == 1:
        data = data.reshape(-1, 1)

    array: np.ndarray = np.zeros(shape, dtype=int)
    array[tuple(data[:, i] for i in range(data.shape[1]))] = 1
    return array


class AKGDataset(Dataset):
    def __init__(self, dataset: str, split: str, comment: int = 0) -> None:
        super().__init__()
        self.comment: comment

        data_dir: Path = Path("data") / dataset
        split_dir: Path = data_dir / f"{split}_{comment}"
        cache_dir: Path = Path("cache")

        self._data: dict[str, StackedArray] = {
            file.stem: StackedArray(file, cache_dir, file.stem == "pos")
            for file in split_dir.iterdir()
        }

    def __len__(self) -> int:
        return len(self._data["input_ids"])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        data: dict[str, np.ndarray] = {k: v[index] for k, v in self._data.items()}
        len_dict: dict[str, int] = {}

        for key in ("input_ids", "node_attr", "edge_attr"):
            len_dict[key] = data[key].shape[0]
            data[f"{key}_labels"] -= 100

        for len_key, key in (("node_attr", "main_top_mask"), ("edge_attr", "edge_dir")):
            data[key] = make_bool_array(data[key], (len_dict[len_key],))

        comment_pairs: np.ndarray = data.pop("comment_pairs")
        data["comment_pair_map"] = np.full_like(data["input_ids"], -100)
        data["comment_pair_map"][comment_pairs[:, 1]] = comment_pairs[:, 0]

        data["top_mask"] = data.pop("main_top_mask")
        data["top_mask"][data.pop("comment_top_mask")] = 2

        data["edge_index"] = data["edge_index"].T
        return {k: torch.tensor(v) for k, v in data.items()}
