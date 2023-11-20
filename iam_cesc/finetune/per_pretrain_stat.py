from csv import DictWriter
from json import load
from pathlib import Path

metrics: tuple[str, ...] = ("macro_F1", "micro_F1", "support_F1", "contest_F1")
stats: tuple[str, ...] = ("mean", "std", "min", "max")
name_components: tuple[str, ...] = ("mode", "comment", "pretrain")


def extract_name(name: str) -> dict[str, str | int]:
    p: list[str] = name.strip().split("_")
    mode: int = int(p[0])
    comment: int = -1
    pretrain: int = 0

    if len(p) > 1:
        comment = int(p[-2])
        pretrain = int(p[-1]) * 10000

    return {"mode": mode, "comment": comment, "pretrain": pretrain}


if __name__ == "__main__":
    summary_dir: Path = Path("summary") / "per_pretrain"

    data: dict[str, dict[str, dict[str, float]]] = {
        stat: {metric: {} for metric in metrics} for stat in stats
    }

    for metric in metrics:
        in_file: Path = summary_dir / f"test_{metric}.json"

        with in_file.open(encoding="utf8") as f:
            raw: dict[str, dict[str, dict[str, float]]] = load(f)
            for stat in stats:
                data[stat][metric] = {k: v["per_seed_by_dev"][stat] for k, v in raw.items()}

    for stat in stats:
        cur_data: dict[str, dict[str, float]] = data[stat]
        out_file: Path = summary_dir / f"{stat}.csv"
        output: list[dict[str, str | int | float]] = []

        for name in sorted(cur_data[metrics[0]].keys()):
            output.append(extract_name(name))
            output[-1].update({m: cur_data[m][name] for m in metrics})

        with out_file.open("w", encoding="utf8", newline="") as f:
            writer = DictWriter(f, name_components + metrics)
            writer.writeheader()
            writer.writerows(output)
