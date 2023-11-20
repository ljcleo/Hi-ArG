from pathlib import Path

import numpy as np
import pandas as pd
from pandarallel import pandarallel
from transformers import RobertaTokenizerFast
from utils.args import get_target_dataset
from utils.io import dump_component, load_component


def edge_inv(x: str) -> str:
    return x[:-3] if x.endswith("-of") else f"{x}-of"


if __name__ == "__main__":
    data_dir: Path = Path("data")
    source_dir: Path = data_dir / "paired"
    target_dir: Path = data_dir / "tokenized"
    target_dir.mkdir(exist_ok=True)
    tokenizer: RobertaTokenizerFast = RobertaTokenizerFast.from_pretrained("./tokenizer")

    def translate(token: str) -> int:
        result = tokenizer.convert_tokens_to_ids(token)
        if result != tokenizer.unk_token_id:
            return result

        result = tokenizer.convert_tokens_to_ids(f"Ġ{token}")
        if result != tokenizer.unk_token_id:
            return result

        result = tokenizer.convert_tokens_to_ids(token.lower())
        if result != tokenizer.unk_token_id:
            return result

        result = tokenizer.convert_tokens_to_ids(f"Ġ{token.lower()}")
        if result != tokenizer.unk_token_id:
            return result

        result = tokenizer.convert_tokens_to_ids(token.capitalize())
        if result != tokenizer.unk_token_id:
            return result

        result = tokenizer.convert_tokens_to_ids(f"Ġ{token.capitalize()}")
        if result != tokenizer.unk_token_id:
            return result

        for i in (2, 3, 4):
            if len(token) > i and token[-i] == "$" and token[-i + 1 :].isdigit():
                return translate(token[:-2])

        return tokenizer.unk_token_id

    target_dataset: str | None = get_target_dataset()
    pandarallel.initialize(progress_bar=True)

    for dataset_dir in source_dir.iterdir():
        dataset_name: str = dataset_dir.stem
        if target_dataset is not None and dataset_name != target_dataset:
            continue

        nodes: pd.DataFrame = load_component(dataset_dir, "", "nodes")
        edges: pd.DataFrame = load_component(dataset_dir, "", "edges")
        print(f"{dataset_name} graph loaded.")

        nodes["text"] = nodes["text"].parallel_map(translate).astype(np.uint16)
        edges["from"] = edges["from"].astype(np.uint32)
        edges["to"] = edges["to"].astype(np.uint32)

        edges["inv_type"] = (
            edges["type"].parallel_map(lambda x: translate(f":{edge_inv(x)}")).astype(np.uint16)
        )

        edges["type"] = edges["type"].parallel_map(lambda x: translate(f":{x}")).astype(np.uint16)
        print(f"{dataset_name} graph tokenized.")

        num_nodes: int = nodes.shape[0]
        from_node: np.ndarray
        from_count: np.ndarray
        from_node, from_count = np.unique(edges["from"], return_counts=True)
        out_degree: np.ndarray = np.zeros(num_nodes, dtype=np.uint32)
        out_degree[from_node] = from_count.astype(np.uint32)
        nodes["start"] = np.cumsum(out_degree, dtype=np.uint32) - out_degree
        print(f"{dataset_name} edge list prepared.")

        target_sub_dir: Path = target_dir / dataset_name
        target_sub_dir.mkdir(exist_ok=True)

        dump_component(nodes, target_sub_dir, "", "nodes")
        dump_component(edges, target_sub_dir, "", "edges")
        print(f"{dataset_name} graph dumped.")
