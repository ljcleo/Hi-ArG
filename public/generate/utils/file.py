from pathlib import Path


def cleanup(dir: Path) -> None:
    if dir.is_file():
        dir.unlink()
    if not dir.exists():
        dir.mkdir(parents=True)

    for f in dir.iterdir():
        if f.is_dir():
            cleanup(f)
            f.rmdir()
        else:
            f.unlink()
