from collections import deque
from json import load
from multiprocessing import Manager
from multiprocessing.pool import ApplyResult, Pool
from multiprocessing.queues import Queue
from os import cpu_count
from pathlib import Path
from queue import Empty
from typing import Any

from spacy import load as load_spacy
from spacy.language import Language
from utils.file import cleanup


def work(pid: int, queue: Queue, out_dir: Path, max_buffer: int) -> int:
    nlp: Language = load_spacy("en_core_web_sm")
    nlp.disable_pipes("parser", "ner")
    nlp.enable_pipe("senter")

    local_buffer: deque[str] = deque()
    fid: int = (pid + 1) * 10

    name: str = f"WORKER {pid:02d}"
    count: int = 0
    good: int = 0

    while True:
        premise: str
        stance: str
        conclusion: str

        try:
            premise, stance, conclusion = queue.get(timeout=3)
            premise_snt: tuple[str, ...] = tuple(x.text for x in nlp(premise).sents if len(x) >= 5)

            if len(premise_snt) >= 1:
                local_buffer.append(
                    f'"{conclusion}" is ' f"{'right' if stance == 'PRO' else 'wrong'}."
                )

                local_buffer.extend(premise_snt)
                good += 1

                while len(local_buffer) >= max_buffer:
                    with (out_dir / f"{fid:03d}.txt").open("w", encoding="utf8") as f:
                        for _ in range(max_buffer):
                            print(local_buffer.popleft(), file=f)

                    fid += 1

            count += 1
            if count % 500 == 0:
                print(f"[{name}] COUNT {count} GOOD {good}")
        except Empty:
            break

    print(f"[{name}] COUNT {count} GOOD {good}")

    while len(local_buffer) >= max_buffer:
        with (out_dir / f"{fid:03d}.txt").open("w", encoding="utf8") as f:
            for _ in range(max_buffer):
                print(local_buffer.popleft(), file=f)

        fid += 1

    if len(local_buffer) > 0:
        with (out_dir / f"{fid:03d}.txt").open("w", encoding="utf8") as f:
            print(*local_buffer, file=f)

        fid += 1

    return fid


if __name__ == "__main__":
    data_dir = Path("data")
    in_file: Path = data_dir / "raw" / "args-me-1.0-cleaned.json"
    out_dir: Path = data_dir / "extracted"
    cleanup(out_dir)

    raw_triples: list[tuple[str, str, str]] = []

    with in_file.open() as f:
        arguments: list[dict[str, Any]] = load(f)["arguments"]

        for argument in arguments:
            conclusion: str = argument["conclusion"]
            premises: list[dict[str, str]] = argument["premises"]

            for record in premises:
                stance: str = record["stance"]
                premise: list[str] = record["text"]
                raw_triples.append((premise, stance, conclusion))

    print("QUEUE SIZE", len(raw_triples))
    max_buffer: int = 65536

    with Manager() as manager:
        queue: Queue = manager.Queue(maxsize=len(raw_triples))
        pool_size: int = cpu_count()

        for triple in raw_triples:
            queue.put(triple)

        with Pool(processes=pool_size) as pool:
            results: list[ApplyResult] = [
                pool.apply_async(work, args=(pid, queue, out_dir, max_buffer))
                for pid in range(pool_size)
            ]

            for result in results:
                print(f"OUTPUT {result.get()} FILES")
