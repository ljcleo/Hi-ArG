import torch
import torch.nn as nn
from model import GreaseArgConfig, GreaseArgModel, GreaseArgPreTrainedModel
from torch import Tensor
from torch_geometric.data import Batch, Data
from transformers.modeling_outputs import SequenceClassifierOutput


class GreaseArgClassificationTransform(nn.Module):
    def __init__(self, config: GreaseArgConfig) -> None:
        super().__init__()
        self.config = config
        input_size: int = 0

        if config.input_text:
            input_size += config.hidden_size
        if config.input_graph:
            input_size += config.gnn_hidden_size << 1

        self.dense = nn.Linear(input_size, config.hidden_size)
        self.tanh = nn.Tanh()
        self.output = nn.Linear(config.hidden_size, config.num_labels)

        self.dropout = nn.Dropout(
            config.classifier_dropout if config.classifier_dropout else config.hidden_dropout_prob
        )

    def forward(
        self, lm_x: Tensor | None = None, gnn_x: Tensor | None = None, ptr: Tensor | None = None
    ) -> Tensor:
        inputs = []
        if self.config.input_text:
            inputs.append(lm_x[:, 0])

        if self.config.input_graph:
            gin: list[Tensor] = []

            for i, j in zip(ptr[:-1], ptr[1:]):
                cur_gnn_x: Tensor = gnn_x[i:j, :]
                gin.append(cur_gnn_x[j - i - 2 :, :].view(1, -1))

            inputs.append(torch.cat(gin))

        x = torch.cat(inputs, dim=1)
        x = self.dense(x)
        x = self.tanh(x)
        x = self.dropout(x)
        x = self.output(x)
        return x


class GreaseArgModelForClassification(GreaseArgPreTrainedModel):
    def __init__(self, config: GreaseArgConfig):
        super().__init__(config)
        self.config = config

        self.grease_arg = GreaseArgModel(config)
        self.transform = GreaseArgClassificationTransform(config)
        self.loss = nn.CrossEntropyLoss()

        self.post_init()

    def forward(
        self,
        input_ids: Tensor | None = None,
        attention_mask: Tensor | None = None,
        graphs: list[Data] | None = None,
        mark: Tensor | None = None,
        labels: Tensor | None = None,
    ):
        if self.config.input_graph:
            graphs = Batch.from_data_list([graphs[i] for i in mark])

        lm_x, gnn_x = self.grease_arg(input_ids, attention_mask, graphs)

        logits = self.transform(lm_x, gnn_x, graphs.ptr if self.config.input_graph else None).view(
            -1, self.config.num_labels
        )

        loss = None if labels is None else self.loss(logits, labels)
        return SequenceClassifierOutput(loss=loss, logits=logits)
