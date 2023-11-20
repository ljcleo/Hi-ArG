from pathlib import Path

import numpy as np
import pandas as pd
from transformers import BatchEncoding, RobertaTokenizerFast
from utils.args import get_target_dataset
from utils.io import dump_component, load_component
from utils.np import cat_ragged

if __name__ == "__main__":
    data_dir: Path = Path("data")
    source_dir: Path = data_dir / "paired"
    target_dir: Path = data_dir / "tokenized"
    target_dir.mkdir(exist_ok=True)

    tokenizer: RobertaTokenizerFast = RobertaTokenizerFast.from_pretrained("./tokenizer")
    target_dataset: str | None = get_target_dataset()

    for dataset_dir in source_dir.iterdir():
        dataset_name: str = dataset_dir.stem
        if target_dataset is not None and dataset_name != target_dataset:
            continue

        tops: pd.DataFrame = load_component(dataset_dir, "", "tops")
        print(f"{dataset_name} tops loaded.")

        top_snt: list[str] = tops["snt"].to_list()
        tops.drop(columns="snt", inplace=True)

        tokenized_result: BatchEncoding = tokenizer(
            top_snt, add_special_tokens=False, truncation=True, return_offsets_mapping=True
        )

        print(f"{dataset_name} text tokenized.")

        top_snt_cat: np.ndarray
        top_snt_start: np.ndarray

        top_snt_cat, top_snt_start = cat_ragged(
            [np.array(x, dtype=np.uint16) for x in tokenized_result["input_ids"]]
        )

        tops["top"] = tops["top"].astype(np.uint32)
        tops["start"] = pd.Series(top_snt_start, dtype=np.uint32)
        print(f"{dataset_name} text concatenated.")

        text_align_cat: np.ndarray = cat_ragged(
            [
                np.array(x, dtype=np.uint16).reshape(-1, 2)
                for x in tokenized_result["offset_mapping"]
            ]
        )[0]

        print(f"{dataset_name} text align concatenated.")

        target_sub_dir: Path = target_dir / dataset_name
        target_sub_dir.mkdir(exist_ok=True)

        dump_component(tops, target_sub_dir, "", "tops")
        print(f"{dataset_name} tops dumped.")

        np.savez_compressed(target_sub_dir / "top_snt.npz", data=top_snt_cat)
        print(f"{dataset_name} text dumped.")

        np.savez_compressed(target_sub_dir / "text_align.npz", data=text_align_cat)
        print(f"{dataset_name} text align dumped.")
