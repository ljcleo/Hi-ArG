from hashlib import sha256
from json import dumps
from pathlib import Path

if __name__ == "__main__":
    data_dir = Path("data")
    cache_dir = Path("cache")
    target_dir = Path("cache_info")
    target_dir.mkdir(exist_ok=True)

    for dataset_dir in data_dir.iterdir():
        for split_dir in dataset_dir.iterdir():
            target_file: str = f"{dataset_dir.stem}_{split_dir.stem}.jsonl"
            info: list[tuple[str, str, int]] = []

            for source_file in split_dir.iterdir():
                path: str = source_file.absolute().as_posix()
                key: str = sha256(path.encode("utf8")).hexdigest()
                size: int = (cache_dir / f"{key}-data-data").stat().st_size
                info.append((source_file.stem, key, size))

            with (target_dir / target_file).open("w", encoding="utf8") as f:
                f.writelines(
                    [
                        dumps({"name": name, "key": key, "size": size}) + "\n"
                        for name, key, size in sorted(info, key=lambda x: -x[2])
                    ]
                )
