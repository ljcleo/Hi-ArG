import numpy as np
import torch
import torch.nn as nn
from model import GreaseArgConfig, GreaseArgModel, GreaseArgPreTrainedModel
from torch import Tensor
from torch_geometric.data import Batch, Data
from transformers.modeling_outputs import SequenceClassifierOutput


class GreaseArgMultipleChoiceTransform(nn.Module):
    def __init__(self, config: GreaseArgConfig) -> None:
        super().__init__()
        self.config = config
        input_size: int = 0

        if config.input_text:
            input_size += config.hidden_size
        if config.input_graph:
            input_size += config.gnn_hidden_size

        self.dense = nn.Linear(input_size, config.hidden_size)
        self.tanh = nn.Tanh()
        self.output = nn.Linear(config.hidden_size, config.hidden_size)

        self.dropout = nn.Dropout(
            config.classifier_dropout if config.classifier_dropout else config.hidden_dropout_prob
        )

    def forward(
        self,
        lm_x: Tensor | None = None,
        choice_mask: Tensor | None = None,
        gnn_x: Tensor | None = None,
        top_mask: Tensor | None = None,
        ptr: Tensor | None = None,
    ) -> Tensor:
        inputs = []

        if self.config.input_text:
            inputs.append(lm_x[choice_mask.to(torch.bool)])

        if self.config.input_graph:
            gin: list[Tensor] = []

            for i, j in zip(ptr[:-1], ptr[1:]):
                cur_gnn_x: Tensor = gnn_x[i:j, :]
                cur_top_mask: Tensor = top_mask[i:j]

                top_pos: Tensor = cur_top_mask[
                    cur_top_mask.argsort(descending=True)[: (cur_top_mask != -100).sum()]
                ].flip(0)

                gin.append(cur_gnn_x[top_pos, :])

            inputs.append(torch.cat(gin))

        x = torch.cat(inputs, dim=1)
        x = self.dense(x)
        x = self.tanh(x)
        x = self.dropout(x)
        x = self.output(x)
        return x


class GreaseArgModelForMultipleChoice(GreaseArgPreTrainedModel):
    def __init__(self, config: GreaseArgConfig, use_cs: bool):
        super().__init__(config)
        self.config = config
        self.use_cs = use_cs

        self.grease_arg = GreaseArgModel(config)
        self.transform = GreaseArgMultipleChoiceTransform(config)

        if not use_cs:
            self.linear = nn.Linear(config.hidden_size << 1, 1)

        self.post_init()

    def forward(
        self,
        input_ids: list[Tensor] | None = None,
        attention_mask: list[Tensor] | None = None,
        choice_mask: list[Tensor] | None = None,
        graphs: list[list[Data]] | None = None,
        labels: list[Tensor] | None = None,
    ):
        group_size = None
        if labels is not None:
            labels = torch.cat([x[0, 1:] for x in labels])

        if self.config.input_text:
            group_size = [x.size(0) for x in input_ids]
            input_ids = torch.cat(input_ids)
            attention_mask = torch.cat(attention_mask)
            choice_mask = torch.cat(choice_mask)

        if self.config.input_graph:
            g_group_size = [len(x) for x in graphs]

            if group_size is None:
                group_size = g_group_size
            else:
                assert g_group_size == group_size

            graphs = Batch.from_data_list(sum(graphs, []))

        lm_x, gnn_x = self.grease_arg(input_ids, attention_mask, graphs)

        repr_x = self.transform(
            lm_x,
            choice_mask,
            gnn_x,
            graphs.top_mask if self.config.input_graph else None,
            graphs.ptr if self.config.input_graph else None,
        ).view(-1, self.config.hidden_size)

        argument_logits = []
        key_point_logits = []
        pivot = 0

        group_end = np.cumsum(group_size)
        group_start = np.concatenate([[0], group_end[:-1]])

        for i, j in zip(group_start, group_end):
            if self.config.input_text:
                count = choice_mask[i:j].count_nonzero().item()
            else:
                count = (graphs.top_mask[graphs.ptr[i] : graphs.ptr[j]] != -100).sum().item()

            group = repr_x[pivot : pivot + count]
            pivot += count
            argument_logits.append(group[1:])
            key_point_logits.append(group[:1].expand(group.size(0) - 1, -1))

        logits_count = [x.size(0) for x in key_point_logits]
        argument_logits = torch.cat(argument_logits)
        key_point_logits = torch.cat(key_point_logits)

        if self.use_cs:
            logits = torch.cosine_similarity(argument_logits, key_point_logits)
        else:
            logits = self.linear(torch.cat([argument_logits, key_point_logits], dim=1)).squeeze(-1)

        if labels is None:
            loss = None
        else:
            mask = labels != -100
            labels = labels[mask]

            if self.use_cs:
                loss = 0.5 * torch.cosine_embedding_loss(
                    argument_logits[mask],
                    key_point_logits[mask],
                    labels * 2 - 1,
                    margin=0.5,
                    reduction=1,
                ).pow(2)
            else:
                loss = torch.binary_cross_entropy_with_logits(
                    logits[mask], labels.float(), reduction=1
                )

        logits_end = np.cumsum(logits_count)
        logits_start = np.concatenate([[0], logits_end[:-1]])
        logits = [logits[i:j] for i, j in zip(logits_start, logits_end)]
        return SequenceClassifierOutput(loss=loss, logits=logits)
