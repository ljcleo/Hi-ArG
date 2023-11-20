import math

import torch
import torch.nn as nn
import torch_geometric.nn as gnn
from torch import Tensor
from torch_geometric.data import Batch
from transformers import PreTrainedModel, RobertaConfig, RobertaModel
from transformers.activations import ACT2FN


def get_activation(act: str | nn.Module) -> nn.Module:
    return ACT2FN[act] if isinstance(act, str) else act


class GreaseArgConfig(RobertaConfig):
    model_type: str = "GreaseArg"

    def __init__(
        self,
        input_text: bool = True,
        input_graph: bool = True,
        gnn_pos_embed_size: int = 4,
        num_joint_layers: int = 9,
        num_gnn_attn_heads: int = 2,
        gnn_hidden_size: int = 256,
        gnn_hidden_act: str | nn.Module = "relu",
        num_mix_layers: int = 1,
        mix_hidden_size: int = 512,
        mix_hidden_act: str | nn.Module = "relu",
        num_joint_attn_heads: int = 8,
        **kwargs
    ) -> None:
        super().__init__(**kwargs)
        if not input_text and not input_graph:
            raise ValueError

        self.input_text = input_text
        self.input_graph = input_graph
        self.gnn_pos_embed_size = gnn_pos_embed_size
        self.num_joint_layers = num_joint_layers
        self.num_gnn_attention_heads = num_gnn_attn_heads
        self.gnn_hidden_size = gnn_hidden_size
        self.gnn_hidden_act = gnn_hidden_act
        self.num_mix_layers = num_mix_layers
        self.mix_hidden_size = mix_hidden_size
        self.mix_hidden_act = mix_hidden_act
        self.num_joint_attn_heads = num_joint_attn_heads

    @staticmethod
    def from_roberta_config(config: RobertaConfig, **kwargs) -> "GreaseArgConfig":
        kwargs.update(config.to_dict())
        return GreaseArgConfig(**kwargs)

    def get_roberta_config(self) -> RobertaConfig:
        return RobertaConfig(**self.to_dict())


class GreaseArgGraphEmbeddings(nn.Module):
    def __init__(self, config: GreaseArgConfig) -> None:
        super().__init__()

        self.node_embedding = nn.Linear(config.hidden_size, config.gnn_hidden_size)
        self.edge_embedding = nn.Linear(config.hidden_size, config.gnn_hidden_size)
        self.position_embedding = nn.Linear(config.gnn_pos_embed_size, config.gnn_hidden_size)

        self.node_layer_norm = nn.LayerNorm(config.gnn_hidden_size, eps=config.layer_norm_eps)
        self.edge_layer_norm = nn.LayerNorm(config.gnn_hidden_size, eps=config.layer_norm_eps)

        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, x: Tensor, edge_attr: Tensor, pos: Tensor) -> tuple[Tensor, Tensor]:
        x = self.node_embedding(x)
        pos = self.position_embedding(pos)
        x = x + pos
        x = self.node_layer_norm(x)
        x = self.dropout(x)

        edge_attr = self.edge_embedding(edge_attr)
        edge_attr = self.edge_layer_norm(edge_attr)
        edge_attr = self.dropout(edge_attr)
        return x, edge_attr


class GreaseArgGraphTransformerLayer(nn.Module):
    def __init__(self, config: GreaseArgConfig) -> None:
        super().__init__()
        output_size: int = config.gnn_hidden_size // config.num_gnn_attention_heads

        self.transformer_conv = gnn.TransformerConv(
            in_channels=config.gnn_hidden_size,
            out_channels=output_size,
            heads=config.num_gnn_attention_heads,
            dropout=config.hidden_dropout_prob,
            edge_dim=config.gnn_hidden_size,
        )

        self.activation: nn.Module = get_activation(config.gnn_hidden_act)
        self.layer_norm = nn.LayerNorm(config.gnn_hidden_size, eps=config.layer_norm_eps)

    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor):
        x = self.transformer_conv(x, edge_index, edge_attr=edge_attr)
        x = self.layer_norm(x)
        x = self.activation(x)
        return x


class GreaseArgMixLayer(nn.Module):
    def __init__(self, config: GreaseArgConfig) -> None:
        super().__init__()

        self.hidden_size: int = config.hidden_size
        merge_hidden_size: int = config.hidden_size + config.gnn_hidden_size

        self.dense_1 = nn.Linear(merge_hidden_size, config.mix_hidden_size)
        self.dense_2 = nn.Linear(config.mix_hidden_size, merge_hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.activation: nn.Module = get_activation(config.mix_hidden_act)
        self.layer_norm = nn.LayerNorm(config.mix_hidden_size, eps=config.layer_norm_eps)

    def forward(self, lm_x: Tensor, gnn_x: Tensor, ptr: Tensor) -> tuple[Tensor, Tensor]:
        head_x: Tensor = lm_x[:, 0, :]
        top_x: Tensor = gnn_x[ptr[1:] - 1, :]

        x: Tensor = torch.cat([head_x, top_x], dim=1)
        x = self.dense_1(x)
        x = self.dropout(x)
        x = self.activation(x)
        x = self.layer_norm(x)
        x = self.dense_2(x)
        head_x = x[:, : self.hidden_size]
        top_x = x[:, self.hidden_size :]

        lm_x = torch.cat([head_x.unsqueeze(1), lm_x[:, 1:, :]], dim=1)
        graphs_x: list[Tensor] = []

        for i, (j, k) in enumerate(zip(ptr[:-1], ptr[1:])):
            graphs_x.extend([top_x[i : i + 1], gnn_x[j + 1 : k]])

        gnn_x = torch.cat(graphs_x)
        return lm_x, gnn_x


class GreaseArgMutualAttention(nn.Module):
    def __init__(self, config: GreaseArgConfig):
        super().__init__()

        self.num_attn_heads: int = config.num_attention_heads
        self.head_size = int(config.hidden_size / self.num_attn_heads)
        self.all_head_size: int = self.num_attn_heads * self.head_size

        self.lm_q = nn.Linear(config.hidden_size, self.all_head_size)
        self.lm_k = nn.Linear(config.hidden_size, self.all_head_size)
        self.lm_v = nn.Linear(config.hidden_size, self.all_head_size)
        self.gnn_q = nn.Linear(config.gnn_hidden_size, self.all_head_size)
        self.gnn_k = nn.Linear(config.gnn_hidden_size, self.all_head_size)
        self.gnn_v = nn.Linear(config.gnn_hidden_size, self.all_head_size)

        self.lm_dense = nn.Linear(self.all_head_size, config.hidden_size)
        self.gnn_dense = nn.Linear(self.all_head_size, config.gnn_hidden_size)

        self.lm_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.gnn_layer_norm = nn.LayerNorm(config.gnn_hidden_size, eps=config.layer_norm_eps)

        self.attn_dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.softmax = nn.Softmax(-1)

    def transpose_for_scores(self, x: Tensor) -> Tensor:
        new_shape: tuple[int, ...] = x.size()[:-1] + (self.num_attn_heads, self.head_size)

        x = x.view(new_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self, lm_x: Tensor, lm_mask: Tensor, gnn_x: Tensor, gnn_mask: Tensor
    ) -> tuple[Tensor, Tensor]:
        lm_q: Tensor = self.transpose_for_scores(self.lm_q(lm_x))
        lm_k: Tensor = self.transpose_for_scores(self.lm_k(lm_x))
        lm_v: Tensor = self.transpose_for_scores(self.lm_v(lm_x))
        gnn_q: Tensor = self.transpose_for_scores(self.gnn_q(gnn_x))
        gnn_k: Tensor = self.transpose_for_scores(self.gnn_k(gnn_x))
        gnn_v: Tensor = self.transpose_for_scores(self.gnn_v(gnn_x))

        lm_gnn_scores: Tensor = lm_q @ gnn_k.transpose(-1, -2)
        lm_gnn_scores = lm_gnn_scores / math.sqrt(self.head_size)
        lm_gnn_scores = lm_gnn_scores + gnn_mask

        gnn_lm_scores: Tensor = gnn_q @ lm_k.transpose(-1, -2)
        gnn_lm_scores = gnn_lm_scores / math.sqrt(self.head_size)
        gnn_lm_scores = gnn_lm_scores + lm_mask

        lm_gnn_probs: Tensor = self.softmax(lm_gnn_scores)
        gnn_lm_probs: Tensor = self.softmax(gnn_lm_scores)

        lm_gnn_probs = self.attn_dropout(lm_gnn_probs)
        gnn_lm_probs = self.attn_dropout(gnn_lm_probs)

        lm_gnn_ctx: Tensor = lm_gnn_probs @ gnn_v
        lm_gnn_ctx = lm_gnn_ctx.permute(0, 2, 1, 3).contiguous()
        s: tuple[int, ...] = lm_gnn_ctx.size()[:-2] + (self.all_head_size,)
        lm_gnn_ctx = lm_gnn_ctx.view(s)

        gnn_lm_ctx: Tensor = gnn_lm_probs @ lm_v
        gnn_lm_ctx = gnn_lm_ctx.permute(0, 2, 1, 3).contiguous()
        s = gnn_lm_ctx.size()[:-2] + (self.all_head_size,)
        gnn_lm_ctx = gnn_lm_ctx.view(s)

        lm_gnn_ctx = self.lm_dense(lm_gnn_ctx)
        lm_gnn_ctx = self.dropout(lm_gnn_ctx)
        lm_gnn_ctx = self.lm_layer_norm(lm_x + lm_gnn_ctx)

        gnn_lm_ctx = self.gnn_dense(gnn_lm_ctx)
        gnn_lm_ctx = self.dropout(gnn_lm_ctx)
        gnn_lm_ctx = self.gnn_layer_norm(gnn_x + gnn_lm_ctx)

        return (lm_gnn_ctx, gnn_lm_ctx)


class GreaseArgTextGraphAttention(nn.Module):
    def __init__(self, config: GreaseArgConfig) -> None:
        super().__init__()
        self.attention = GreaseArgMutualAttention(config)

    def forward(
        self, lm_x: Tensor, lm_mask: Tensor, gnn_x: Tensor, ptr: Tensor
    ) -> tuple[Tensor, Tensor]:
        gnn_mask: Tensor = torch.ones(gnn_x.shape[0]).to(lm_mask)

        gnn_x = nn.utils.rnn.pad_sequence(
            [gnn_x[i:j] for i, j in zip(ptr[:-1], ptr[1:])], batch_first=True
        )

        gnn_mask = nn.utils.rnn.pad_sequence(
            [gnn_mask[i:j] for i, j in zip(ptr[:-1], ptr[1:])], batch_first=True
        )

        gnn_mask = gnn_mask[:, None, None, :]
        gnn_mask = (1.0 - gnn_mask) * torch.finfo(lm_mask.dtype).min
        lm_x, gnn_x = self.attention(lm_x, lm_mask, gnn_x, gnn_mask)

        gnn_x = torch.cat([gnn_x[i, : k - j, :] for i, (j, k) in enumerate(zip(ptr[:-1], ptr[1:]))])

        return lm_x, gnn_x


class GreaseArgPreTrainedModel(PreTrainedModel):
    config_class = GreaseArgConfig
    base_model_prefix: str = "grease_arg"
    supports_gradient_checkpointing: bool = False

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def update_keys_to_ignore(self, config: GreaseArgConfig, del_keys_to_ignore: list[str]) -> None:
        if not config.tie_word_embeddings:
            self._keys_to_ignore_on_save = [
                k for k in self._keys_to_ignore_on_save if k not in del_keys_to_ignore
            ]

            self._keys_to_ignore_on_load_missing = [
                k for k in self._keys_to_ignore_on_load_missing if k not in del_keys_to_ignore
            ]


class GreaseArgModel(GreaseArgPreTrainedModel):
    def __init__(self, config: GreaseArgConfig):
        super().__init__(config)
        self.config = config

        self.word_embeddings: nn.Embedding

        if config.input_text:
            self.roberta = RobertaModel(config.get_roberta_config(), add_pooling_layer=False)

            self.word_embeddings = self.roberta.embeddings.word_embeddings
        else:
            self.word_embeddings = nn.Embedding(
                config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id
            )

        if config.input_graph:
            self.graph_embeddings = GreaseArgGraphEmbeddings(config)

            self.gnn_layers = nn.ModuleList(
                [GreaseArgGraphTransformerLayer(config) for _ in range(config.num_joint_layers)]
            )

            if config.input_text:
                self.mix_layers = nn.ModuleList(
                    [GreaseArgMixLayer(config) for _ in range(config.num_joint_layers)]
                )

                self.attention = GreaseArgTextGraphAttention(config)

        self.post_init()

    def load_roberta_pretrained(self, name):
        model: RobertaModel = RobertaModel.from_pretrained(name, add_pooling_layer=False)

        model.resize_token_embeddings(self.config.vocab_size)
        self.word_embeddings = model.embeddings.word_embeddings

        if self.config.input_text:
            self.roberta = model

    def get_input_embeddings(self):
        return self.word_embeddings

    def set_input_embeddings(self, value):
        self.word_embeddings = value
        if self.config.input_text:
            self.roberta.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def forward(
        self,
        input_ids: Tensor | None = None,
        attention_mask: Tensor | None = None,
        graphs: Batch | None = None,
    ) -> tuple[Tensor | None, Tensor | None]:
        if self.config.input_text:
            lm_mask: Tensor = self.get_extended_attention_mask(attention_mask, input_ids.shape)

            lm_x: Tensor = self.roberta.embeddings(input_ids)
            layer_offset: int = self.config.num_hidden_layers - self.config.num_joint_layers

            for layer in self.roberta.encoder.layer[:layer_offset]:
                lm_x = layer(lm_x, lm_mask)[0]

        if self.config.input_graph:
            graphs = graphs.to(self.device)
            gnn_x: Tensor = self.word_embeddings(graphs.x)
            edge_x: Tensor = self.word_embeddings(graphs.edge_attr)

            gnn_x = gnn_x.squeeze(1)
            edge_x = edge_x.squeeze(1)
            gnn_x, edge_x = self.graph_embeddings(gnn_x, edge_x, graphs.pos)

        if not self.config.input_graph:
            for roberta_layer in self.roberta.encoder.layer[layer_offset:]:
                lm_x = roberta_layer(lm_x, lm_mask)[0]
        elif not self.config.input_text:
            for gnn_layer in self.gnn_layers:
                gnn_x = gnn_layer(gnn_x, graphs.edge_index, edge_x)
        else:
            for roberta_layer, gnn_layer, mix_layer in zip(
                self.roberta.encoder.layer[layer_offset:], self.gnn_layers, self.mix_layers
            ):
                lm_x = roberta_layer(lm_x, lm_mask)[0]
                gnn_x = gnn_layer(gnn_x, graphs.edge_index, edge_x)
                lm_x, gnn_x = mix_layer(lm_x, gnn_x, graphs.ptr)

        if not self.config.input_graph:
            return lm_x, None
        elif not self.config.input_text:
            return None, gnn_x
        else:
            lm_x, gnn_x = self.attention(lm_x, lm_mask, gnn_x, graphs.ptr)
            return lm_x, gnn_x
