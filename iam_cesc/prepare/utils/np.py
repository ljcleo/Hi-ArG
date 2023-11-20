import numpy as np


def cat_ragged(data: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    start: np.ndarray = np.concatenate(
        [[0], np.cumsum([x.shape[0] for x in data])[:-1]]
    )

    result: np.ndarray = np.concatenate(data, axis=0)
    return result, start


def map_numpy(array: np.ndarray, map: dict[int, int]) -> np.ndarray:
    unique: np.ndarray
    inv: np.ndarray
    unique, inv = np.unique(array, return_inverse=True)
    return np.array([map[x] for x in unique])[inv].reshape(array.shape)
