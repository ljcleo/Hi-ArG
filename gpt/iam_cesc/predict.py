import re
from argparse import ArgumentParser
from dataclasses import dataclass
from io import TextIOWrapper
from pathlib import Path
from time import sleep

from openai import OpenAI


@dataclass
class CommandArguments:
    prompt_method: str = ""
    api_key: str = ""
    base_url: str = "https://api.openai.com/v1"
    model: str = "gpt-3.5-turbo"
    max_retries: int = 10


def get_nonempty(f: TextIOWrapper) -> list[str]:
    return [x.strip() for x in f if x.strip() != ""]


def parse_output(response: str, error_result: str) -> str:
    try:
        result: re.Match | None = re.search(r"\{\"label\":\s*[012]\}", response)
        return error_result if result is None else re.sub(r"\s", "", result.group(0))
    except Exception:
        return error_result


def main(args: CommandArguments) -> None:
    data_dir = Path("converted")
    prompt_dir = Path("prompt")
    pred_dir = Path("predictions")
    pred_dir.mkdir(exist_ok=True)

    with open(data_dir / "input.jsonl", "r", encoding="utf8") as f:
        inputs: list[str] = get_nonempty(f)

    pred_file: Path = pred_dir / f"{args.prompt_method}.jsonl"
    skip: int = 0

    if pred_file.exists():
        with pred_file.open("r", encoding="utf8") as f:
            skip = len(get_nonempty(f))

    client = OpenAI(api_key=args.api_key, base_url=args.base_url, max_retries=args.max_retries)
    prompt: str = (prompt_dir / f"{args.prompt_method}.txt").read_text(encoding="utf8").strip()
    error_result: str = '{"label":-1}'

    with pred_file.open("a", encoding="utf8") as f:
        for i, task in enumerate(inputs):
            if i < skip:
                print(f"SKIP {i:04d}")
                continue

            result: str = error_result

            try:
                response = str(
                    client.chat.completions.create(
                        messages=[
                            {"role": "system", "content": prompt},
                            {"role": "user", "content": task},
                        ],
                        model=args.model,
                        temperature=0.2,
                    )
                    .choices[0]
                    .message.content
                )

                result = parse_output(response, error_result)

                if result == error_result:
                    sleep(1)

                    result = parse_output(
                        str(
                            client.chat.completions.create(
                                messages=[
                                    {"role": "system", "content": prompt},
                                    {"role": "user", "content": task},
                                    {"role": "assistant", "content": response},
                                    {"role": "user", "content": "Verdict?"},
                                ],
                                model=args.model,
                                temperature=0.2,
                            )
                            .choices[0]
                            .message.content
                        ),
                        error_result,
                    )
            except Exception as e:
                print(e)

            print(f"PREDICT {i:04d} RESULT {result}")
            print(result, file=f, flush=True)
            sleep(1)


def parse_args() -> CommandArguments:
    parser = ArgumentParser()
    parser.add_argument("prompt-method", choices=("direct", "explain"))
    parser.add_argument("api-key")
    parser.add_argument("-u", "--base-url", default=CommandArguments.base_url)
    parser.add_argument("-m", "--model", default=CommandArguments.model)
    parser.add_argument("-r", "--max-retries", default=CommandArguments.max_retries, type=int)
    return parser.parse_args(namespace=CommandArguments())


if __name__ == "__main__":
    main(parse_args())
