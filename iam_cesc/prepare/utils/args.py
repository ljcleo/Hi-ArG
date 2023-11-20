from argparse import ArgumentParser
from dataclasses import dataclass


@dataclass
class PrepareArguments:
    dataset: str = None


def get_target_dataset() -> str | None:
    parser = ArgumentParser(description='Prepare dataset.')
    parser.add_argument('-d', '--dataset', help='Specify dataset.')
    args: PrepareArguments = parser.parse_args(namespace=PrepareArguments())
    return args.dataset
