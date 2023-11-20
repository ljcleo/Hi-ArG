from pathlib import Path

from utils.file import cleanup

if __name__ == "__main__":
    data_dir = Path("data")
    in_dir: Path = data_dir / "extracted"
    out_dir: Path = data_dir / "arranged"
    cleanup(out_dir)

    files: list[Path] = list(in_dir.iterdir())
    files.sort(key=lambda x: x.stem)

    buffer: list[str] = []
    fid: int = 0

    for in_file in files:
        with in_file.open("r", encoding="utf8") as f:
            for line in f:
                line = line.strip()
                if len(line.split()) > 256:
                    continue

                buffer.append(f"{line}\n")

                if len(buffer) >= 512:
                    with (out_dir / f"{fid:05d}.txt").open("w", encoding="utf8") as g:
                        g.writelines(buffer)

                    buffer = []
                    print(f"WRITE TO {fid:05d}.txt")
                    fid += 1

    if len(buffer) > 0:
        with (out_dir / f"{fid:05d}.txt").open("w", encoding="utf8") as g:
            g.writelines(buffer)

        print(f"WRITE TO {fid:05d}.txt")
