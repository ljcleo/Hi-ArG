from dataclasses import dataclass

import torch
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
from torch_geometric.data import Batch, Data


@dataclass
class DataCollatorForJointInput:
    input_text: bool = True
    input_graph: bool = True

    def __post_init__(self):
        if not self.input_text and not self.input_graph:
            raise ValueError

    def __call__(self, examples: list[dict[str, torch.Tensor]]) -> dict[str, Tensor | Batch]:
        text_features: dict[str, Tensor] = {
            k: torch.stack([example.pop(k, torch.tensor(0)) for example in examples])
            for k in ("input_ids", "comment_pair_map", "input_ids_labels")
        }

        text_features["attention_mask"] = torch.ones_like(text_features["input_ids"])

        batch: dict[str, Tensor | Batch] = {
            "match_labels": pad_sequence(
                [example.pop("match_labels") for example in examples],
                batch_first=True,
                padding_value=-100,
            ),
            "mark": torch.arange(len(examples)),
        }

        if self.input_text:
            batch.update(text_features)

        if self.input_graph:
            batch["good_graphs"], batch["bad_graphs"] = tuple(
                Batch.from_data_list(x)
                for x in zip(*[self._corrupt_graph(example) for example in examples])
            )

        return batch

    def _corrupt_graph(self, example: dict[str, torch.Tensor]) -> Data:
        good_graph = Data(
            x=example["node_attr"],
            edge_index=example["edge_index"],
            edge_attr=example["edge_attr"],
            pos=example["pos"],
        )

        bad_graph = good_graph.clone()
        bad_graph.x = bad_graph.x[torch.randperm(bad_graph.x.shape[0])]

        good_graph.y = example["node_attr_labels"]
        good_graph.top_mask = example["top_mask"]
        good_graph.edge_y = example["edge_attr_labels"]
        good_graph.edge_dir = example["edge_dir"]

        return good_graph, bad_graph
