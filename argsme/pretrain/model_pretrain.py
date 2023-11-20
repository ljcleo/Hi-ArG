from dataclasses import dataclass

import torch
import torch.nn as nn
from model import GreaseArgConfig, GreaseArgModel, GreaseArgPreTrainedModel
from torch import Tensor
from torch_geometric.data import Batch
from transformers.activations import ACT2FN
from transformers.utils import ModelOutput


class GreaseArgPredictionTransform(nn.Module):
    def __init__(self, config: GreaseArgConfig, input_type: str) -> None:
        super().__init__()
        input_shape: int = config.hidden_size

        if input_type == "node":
            input_shape = config.gnn_hidden_size
        elif input_type == "edge":
            input_shape = config.gnn_hidden_size << 1

        self.dense = nn.Linear(input_shape, config.hidden_size)

        if isinstance(config.hidden_act, str):
            self.activation = ACT2FN[config.hidden_act]
        else:
            self.activation = config.hidden_act

        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, x: Tensor) -> Tensor:
        x = self.dense(x)
        x = self.activation(x)
        x = self.layer_norm(x)
        return x


class GreaseArgPreTrainingHead(nn.Module):
    def __init__(
        self,
        config: GreaseArgConfig,
        with_text_x: bool = True,
        with_match_x: float = True,
        with_node_x: float = True,
        with_top_x: float = True,
        with_cont_x: float = True,
        with_edge_x: float = True,
        with_dir_x: float = True,
    ) -> None:
        super().__init__()
        self.config = config
        self.with_text_x = with_text_x
        self.with_match_x = with_match_x
        self.with_node_x = with_node_x
        self.with_top_x = with_top_x
        self.with_edge_x = with_edge_x
        self.with_dir_x = with_dir_x
        self.with_cont_x = with_cont_x

        if config.input_text:
            self.text_tf = GreaseArgPredictionTransform(config, "text")
        else:
            self.with_text_x = False
            self.with_match_x = False

        if config.input_graph:
            self.node_tf = GreaseArgPredictionTransform(config, "node")
            self.edge_tf = GreaseArgPredictionTransform(config, "edge")
        else:
            self.with_node_x = False
            self.with_top_x = False
            self.with_edge_x = False
            self.with_dir_x = False
            self.with_cont_x = False

        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))
        self.decoder.bias = self.bias

        if config.input_text:
            self.match_decoder = nn.Linear(config.hidden_size * (2 + config.input_graph), 1)

        if config.input_graph:
            self.top_decoder = nn.Linear(config.hidden_size, config.hidden_size)
            self.dir_decoder = nn.Linear(config.hidden_size, 1)
            self.cont_decoder = nn.Linear(config.hidden_size, 1)

    def forward(
        self,
        lm_x: Tensor | None = None,
        good_gnn_x: Tensor | None = None,
        bad_gnn_x: Tensor | None = None,
        comment_pair_map: Tensor | None = None,
        edge_index: Tensor | None = None,
        top_mask: Tensor | None = None,
        ptr: Tensor | None = None,
    ) -> dict[str, Tensor]:
        result: dict[str, Tensor] = {}
        match_x: list[Tensor] = []

        if self.with_text_x or self.with_match_x:
            text_x: Tensor = self.text_tf(lm_x)

            if self.with_match_x:
                pair_mask: Tensor = comment_pair_map >= 0

                real_pair_map: Tensor = (
                    comment_pair_map
                    + torch.arange(comment_pair_map.shape[0], device=comment_pair_map.device).view(
                        -1, 1
                    )
                    * comment_pair_map.shape[1]
                )[pair_mask]

                match_x.extend(
                    [text_x.view(-1, text_x.shape[-1])[real_pair_map], text_x[pair_mask]]
                )

            if self.with_text_x:
                result["text_x"] = self.decoder(text_x)

        if self.with_match_x or self.with_node_x or self.with_top_x or self.with_cont_x:
            node_x: Tensor = self.node_tf(good_gnn_x)

            if self.with_match_x:
                comment_top_mask: Tensor = top_mask == 2
                match_x.append(node_x[comment_top_mask])

            if self.with_node_x:
                result["node_x"] = self.decoder(node_x)

            if self.with_top_x:
                main_top_mask: Tensor = top_mask == 1

                if torch.any(main_top_mask):
                    top_x: Tensor = node_x[main_top_mask]
                    top_x = self.top_decoder(top_x)
                    all_top_x: list[Tensor] = []
                    pivot = 0

                    for i, j in zip(ptr[:-1], ptr[1:]):
                        bound = pivot + main_top_mask[i:j].sum()
                        cur_top_x = top_x[pivot:bound, :]
                        all_top_x.append((cur_top_x @ cur_top_x.mT).ravel())
                        pivot = bound

                    result["top_x"] = torch.cat(all_top_x)

            if self.with_cont_x:
                result["cont_x"] = self.cont_decoder(
                    torch.stack([node_x, self.node_tf(bad_gnn_x)])
                ).squeeze(-1)

        if self.with_edge_x or self.with_dir_x:
            edge_x: Tensor = self.edge_tf(
                torch.cat([good_gnn_x[edge_index[0, :], :], good_gnn_x[edge_index[1, :], :]], dim=1)
            )

            if self.with_edge_x:
                result["edge_x"] = self.decoder(edge_x)

            if self.with_dir_x:
                result["dir_x"] = self.dir_decoder(edge_x).squeeze(-1)

        if self.with_match_x and len(match_x) > 0:
            result["match_x"] = self.match_decoder(torch.cat(match_x, dim=-1)).squeeze(-1)

        return result


@dataclass
class GreaseArgPreTrainingOutput(ModelOutput):
    loss: torch.FloatTensor | None = None

    text_loss: torch.FloatTensor | None = None
    match_loss: torch.FloatTensor | None = None
    node_loss: torch.FloatTensor | None = None
    top_loss: torch.FloatTensor | None = None
    edge_loss: torch.FloatTensor | None = None
    dir_loss: torch.FloatTensor | None = None
    cont_loss: torch.FloatTensor | None = None

    text_logits: torch.FloatTensor | None = None
    match_logits: torch.FloatTensor | None = None
    node_logits: torch.FloatTensor | None = None
    top_logits: torch.FloatTensor | None = None
    edge_logits: torch.FloatTensor | None = None
    dir_logits: torch.FloatTensor | None = None
    cont_logits: torch.FloatTensor | None = None


class GreaseArgModelForPreTraining(GreaseArgPreTrainedModel):
    def __init__(
        self,
        config: GreaseArgConfig,
        text_loss_weight: float = 1.0,
        match_loss_weight: float = 1.0,
        node_loss_weight: float = 1.0,
        top_loss_weight: float = 1.0,
        edge_loss_weight: float = 1.0,
        dir_loss_weight: float = 1.0,
        cont_loss_weight: float = 1.0,
    ):
        super().__init__(config)
        self.config = config
        self.text_loss_weight = text_loss_weight
        self.match_loss_weight = match_loss_weight
        self.node_loss_weight = node_loss_weight
        self.top_loss_weight = top_loss_weight
        self.edge_loss_weight = edge_loss_weight
        self.dir_loss_weight = dir_loss_weight
        self.cont_loss_weight = cont_loss_weight

        self.grease_arg = GreaseArgModel(config)

        self.pretrain_head = GreaseArgPreTrainingHead(
            config,
            with_text_x=text_loss_weight > 0,
            with_match_x=match_loss_weight > 0,
            with_node_x=node_loss_weight > 0,
            with_top_x=top_loss_weight > 0,
            with_edge_x=edge_loss_weight > 0,
            with_dir_x=dir_loss_weight > 0,
            with_cont_x=cont_loss_weight > 0,
        )

        self.text_loss = nn.CrossEntropyLoss(reduction="none")
        self.match_loss = nn.BCEWithLogitsLoss(reduction="none")
        self.node_loss = nn.CrossEntropyLoss(reduction="none")
        self.top_loss = nn.BCEWithLogitsLoss(reduction="none")
        self.edge_loss = nn.CrossEntropyLoss(reduction="none")
        self.dir_loss = nn.BCEWithLogitsLoss(reduction="none")
        self.cont_loss = nn.BCEWithLogitsLoss(reduction="none")

        self.post_init()

    def get_output_embeddings(self):
        return self.pretrain_head.decoder

    def set_output_embeddings(self, new_embeddings):
        self.pretrain_head.decoder = new_embeddings

    def forward(
        self,
        input_ids: Tensor | None = None,
        attention_mask: Tensor | None = None,
        comment_pair_map: Tensor | None = None,
        good_graphs: Batch | None = None,
        bad_graphs: Batch | None = None,
        input_ids_labels: Tensor | None = None,
        match_labels: Tensor | None = None,
        mark: Tensor | None = None,
    ):
        if mark is not None and self.config.input_graph:
            good_graphs = Batch.from_data_list([good_graphs.get_example(i) for i in mark])
            bad_graphs = Batch.from_data_list([bad_graphs.get_example(i) for i in mark])

        lm_x, good_gnn_x = self.grease_arg(input_ids, attention_mask, good_graphs)

        if self.config.input_graph and self.cont_loss_weight > 0:
            _, bad_gnn_x = self.grease_arg(input_ids, attention_mask, bad_graphs)
        else:
            bad_gnn_x = None

        def get_graph_feature(graphs: Batch, key: str) -> Tensor | None:
            return graphs[key] if self.config.input_graph else None

        pred_x = self.pretrain_head(
            lm_x,
            good_gnn_x,
            bad_gnn_x,
            comment_pair_map,
            *[get_graph_feature(good_graphs, x) for x in ("edge_index", "top_mask", "ptr")]
        )

        text_x = pred_x.get("text_x")
        match_x = pred_x.get("match_x")
        node_x = pred_x.get("node_x")
        top_x = pred_x.get("top_x")
        edge_x = pred_x.get("edge_x")
        dir_x = pred_x.get("dir_x")
        cont_x = pred_x.get("cont_x")

        loss = torch.tensor(0.0)
        text_loss = None
        match_loss = None
        node_loss = None
        top_loss = None
        edge_loss = None
        dir_loss = None
        cont_loss = None

        if text_x is not None:
            loss = loss.to(text_x)
            text_mask = input_ids_labels >= 0

            if torch.any(text_mask):
                text_loss = self.text_loss(text_x[text_mask], input_ids_labels[text_mask])
                assert not torch.any(torch.isnan(text_loss)), "text loss nan"
                assert not torch.any(torch.isinf(text_loss)), "text loss inf"
                text_loss = text_loss.mean()
                loss = loss + text_loss * self.text_loss_weight

        if match_x is not None:
            loss = loss.to(match_x)
            match_mask = match_labels >= 0

            if torch.any(match_mask):
                match_loss = self.match_loss(match_x.ravel(), match_labels[match_mask].float())
                assert not torch.any(torch.isnan(match_loss)), "match loss nan"
                assert not torch.any(torch.isinf(match_loss)), "match loss inf"
                match_loss = match_loss.mean()
                loss = loss + match_loss * self.match_loss_weight

        if node_x is not None:
            loss = loss.to(node_x)
            node_labels = good_graphs.y.ravel()
            node_mask = node_labels >= 0

            if torch.any(node_mask):
                node_loss = self.node_loss(node_x[node_mask], node_labels[node_mask])
                assert not torch.any(torch.isnan(node_loss)), "node loss nan"
                assert not torch.any(torch.isinf(node_loss)), "node loss inf"
                node_loss = node_loss.mean()
                loss = loss + node_loss * self.node_loss_weight

        if top_x is not None:
            loss = loss.to(top_x)
            all_top_label = []
            main_top_mask = good_graphs.top_mask == 1

            for i, j in zip(good_graphs.ptr[:-1], good_graphs.ptr[1:]):
                num_top = main_top_mask[i:j].sum()
                flag = torch.ones(num_top - 1)
                all_top_label.append((torch.diag(flag, 1) + torch.diag(flag, -1)).ravel())

            top_label = torch.cat(all_top_label).to(top_x)
            top_loss = self.top_loss(top_x, top_label)
            assert not torch.any(torch.isnan(top_loss)), "top loss nan"
            assert not torch.any(torch.isinf(top_loss)), "top loss inf"
            top_loss = top_loss.mean()
            loss = loss + top_loss * self.top_loss_weight

        if edge_x is not None:
            loss = loss.to(edge_x)
            edge_labels = good_graphs.edge_y.ravel()
            edge_mask = edge_labels >= 0

            if torch.any(edge_mask):
                edge_loss = self.edge_loss(edge_x[edge_mask], edge_labels[edge_mask])
                assert not torch.any(torch.isnan(edge_loss)), "edge loss nan"
                assert not torch.any(torch.isinf(edge_loss)), "edge loss inf"
                edge_loss = edge_loss.mean()
                loss = loss + edge_loss * self.edge_loss_weight

        if dir_x is not None:
            loss = loss.to(dir_x)
            dir_loss = self.dir_loss(dir_x.ravel(), good_graphs.edge_dir.ravel().float())
            assert not torch.any(torch.isnan(dir_loss)), "dir loss nan"
            assert not torch.any(torch.isinf(dir_loss)), "dir loss inf"
            dir_loss = dir_loss.mean()
            loss = loss + dir_loss * self.dir_loss_weight

        if cont_x is not None:
            loss = loss.to(cont_x)
            cont_labels = torch.zeros_like(cont_x)
            cont_labels[0, :] = 1
            cont_loss = self.cont_loss(cont_x, cont_labels)
            assert not torch.any(torch.isnan(cont_loss)), "cont loss nan"
            assert not torch.any(torch.isinf(cont_loss)), "cont loss inf"
            cont_loss = cont_loss.mean()
            loss = loss + cont_loss * self.cont_loss_weight

        return GreaseArgPreTrainingOutput(
            loss=loss,
            text_loss=text_loss,
            match_loss=match_loss,
            node_loss=node_loss,
            top_loss=top_loss,
            edge_loss=edge_loss,
            dir_loss=dir_loss,
            cont_loss=cont_loss,
            text_logits=text_x,
            match_logits=match_x,
            node_logits=node_x,
            top_logits=top_x,
            edge_logits=edge_x,
            dir_logits=dir_x,
            cont_logits=cont_x,
        )
