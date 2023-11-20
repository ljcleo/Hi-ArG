from collections import defaultdict
from json import dump, load
from pathlib import Path
from typing import Callable

import numpy as np

metrics: tuple[str, ...] = ("macro_F1", "micro_F1", "support_F1", "contest_F1")
gold: int = metrics.index("macro_F1")

stats: dict[str, Callable[[np.ndarray], np.floating]] = {
    "mean": np.mean,
    "std": np.std,
    "min": np.min,
    "max": np.max,
}


def extract(data: dict[str, float], prefix: str) -> np.ndarray:
    return np.array([data[f"{prefix}_{metric}"] for metric in metrics])


def process(input_file: Path) -> list[tuple[str, np.ndarray]]:
    with input_file.open() as f:
        raw: dict[str, dict[str, float]] = load(f)

    keys: list[str] = sorted(raw.keys())
    seq: list[tuple[str, np.ndarray, np.ndarray]] = []

    for key in keys:
        data: dict[str, float] = raw[key]
        if "test_loss" in raw[key].keys():
            seq.append((key, np.vstack([extract(data, prefix) for prefix in ("eval", "test")])))

    return seq


def summary(data: np.ndarray) -> dict[str, float]:
    return {key: func(data).item() for key, func in stats.items()}


def best_per_seed(seq: np.ndarray, key: np.ndarray) -> dict[str, float]:
    return summary(seq[np.argmax(key, axis=0), np.arange(seq.shape[1])])


def best_overall(seq: np.ndarray, key: np.ndarray, k: int) -> dict[str, float]:
    return summary(seq.ravel()[np.argsort(key.ravel())[-k:]])


def rename(old: str) -> str:
    components: list[str] = old.strip().split("_")
    if len(components) > 2:
        components[-1] = f"{int(components[-1]) // 10000:02d}"

    return "_".join(components)


if __name__ == "__main__":
    source_dir = Path("evaluate")
    target_dir: Path = Path("summary") / "per_pretrain"
    target_dir.mkdir(parents=True, exist_ok=True)

    targets: list[Path] = [x for x in source_dir.iterdir() if x.is_dir()]
    targets.sort(key=lambda x: x.name)
    result: dict[str, dict[str, np.ndarray]] = {}

    for target in targets:
        name: str = rename(target.stem)
        temp: defaultdict[str, list[np.ndarray]] = defaultdict(list)

        for sub in target.iterdir():
            for s, x in process(sub):
                temp[s].append(x)

        if len(temp) > 0:
            result[name] = {k: np.stack(v, axis=-1) for k, v in temp.items()}

    for i, m in enumerate(metrics):
        dev_summary: dict[str, dict[str, dict[str, float]]] = {}
        test_summary: dict[str, dict[str, dict[str, float]]] = {}

        for name, seq in result.items():
            np_seq: np.ndarray = np.stack([x[1, i] for x in seq.values()])
            dev_key: np.ndarray = np.stack([x[0, gold] for x in seq.values()])
            test_key: np.ndarray = np.stack([x[1, gold] for x in seq.values()])
            dev_summary[name] = {s: summary(x[0, i]) for s, x in seq.items()}

            test_summary[name] = {
                "per_seed_by_dev": best_per_seed(np_seq, dev_key),
                "per_seed_by_test": best_per_seed(np_seq, test_key),
                "overall_by_dev": best_overall(np_seq, dev_key, 6),
                "overall_by_test": best_overall(np_seq, test_key, 6),
            }

        with (target_dir / f"dev_{m}.json").open("w", encoding="utf8") as f:
            dump(dev_summary, f, indent=4, sort_keys=True)
        with (target_dir / f"test_{m}.json").open("w", encoding="utf8") as f:
            dump(test_summary, f, indent=4, sort_keys=True)
