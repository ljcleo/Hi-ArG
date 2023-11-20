from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor
from torch_geometric.data import Data
from transformers import DataCollatorWithPadding, PreTrainedTokenizerBase


class PseudoTensor(list):
    def numel(self) -> int:
        return sum(x.numel() for x in self)


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

    def __call__(
        self, examples: list[tuple[dict[str, Tensor], ...]]
    ) -> dict[str, list[Tensor] | list[list[Data]]]:
        group_size: np.ndarray = np.array([len(group) for group in examples])
        group_end: np.ndarray = np.cumsum(group_size)
        group_start: np.ndarray = np.concatenate([[0], group_end[:-1]])
        intervals: list[tuple[int, int]] = list(zip(group_start, group_end))

        batch: dict[str, list[Tensor]] = {
            "labels": [torch.cat([x.pop("labels") for x in group], dim=1) for group in examples],
        }

        if self.input_text:
            flat_text_features: dict[str, Tensor] = self._collator(
                [{k: example.pop(k) for k in ("input_ids",)} for example in sum(examples, ())]
            )

            batch.update({k: [v[i:j] for i, j in intervals] for k, v in flat_text_features.items()})
            batch["input_ids"] = PseudoTensor(batch["input_ids"])
            pad_length: int = batch["input_ids"][0].shape[1]

            batch["choice_mask"] = [
                self._pad([example.pop("choice_mask") for example in group], pad_length)
                for group in examples
            ]

        if self.input_graph:
            batch["graphs"] = [
                [
                    Data(
                        x=example["node_attr"],
                        edge_index=example["edge_index"],
                        edge_attr=example["edge_attr"],
                        y=example["node_attr"],
                        pos=example["pos"],
                        edge_y=example["edge_attr"],
                        top_mask=example["top_mask"],
                    )
                    for example in group
                ]
                for group in examples
            ]

        return batch

    def _pad(self, data: list[Tensor], pad_length: int) -> Tensor:
        result = torch.full((len(data), pad_length), 0).to(data[0])
        for i, x in enumerate(data):
            result[i, : x.shape[0]] = x

        return result
