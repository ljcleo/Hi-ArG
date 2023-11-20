from dataclasses import dataclass

import torch
from torch import Tensor
from torch_geometric.data import Data
from transformers import DataCollatorWithPadding, PreTrainedTokenizerBase


@dataclass
class DataCollatorForJointInput:
    tokenizer: PreTrainedTokenizerBase
    pad_to_multiple_of: int = 16
    input_text: bool = True
    input_graph: bool = True

    def __post_init__(self):
        if self.tokenizer.mask_token is None or self.tokenizer.eos_token is None:
            raise ValueError
        if not self.input_text and not self.input_graph:
            raise ValueError

        self._collator = DataCollatorWithPadding(
            self.tokenizer, pad_to_multiple_of=self.pad_to_multiple_of
        )

    def __call__(self, examples: list[dict[str, Tensor]]) -> dict[str, Tensor | list[Data]]:
        batch: dict[str, list[Tensor]] = {"labels": torch.cat([x.pop("labels") for x in examples])}

        if self.input_text:
            batch.update(
                self._collator(
                    [{k: example.pop(k) for k in ("input_ids",)} for example in examples]
                )
            )

        if self.input_graph:
            batch["graphs"] = [
                Data(
                    x=example["node_attr"],
                    edge_index=example["edge_index"],
                    edge_attr=example["edge_attr"],
                    y=example["node_attr"],
                    pos=example["pos"],
                    edge_y=example["edge_attr"],
                )
                for example in examples
            ]

            batch["mark"] = torch.arange(len(examples))

        return batch

    def _pad(self, data: list[Tensor], pad_length: int) -> Tensor:
        result = torch.full((len(data), pad_length), 0).to(data[0])
        for i, x in enumerate(data):
            result[i, : x.shape[0]] = x

        return result
