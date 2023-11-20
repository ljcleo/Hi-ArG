from dataclasses import dataclass
from json import load
from multiprocessing.resource_tracker import unregister
from multiprocessing.shared_memory import SharedMemory
from pathlib import Path
from random import choices
from typing import ClassVar

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import ArpackNoConvergence, eigs
from torch.utils.data import Dataset


def map_numpy(array: np.ndarray, map: dict[int, int]) -> np.ndarray:
    unique: np.ndarray
    inv: np.ndarray
    unique, inv = np.unique(array, return_inverse=True)
    return np.array([map[x] for x in unique])[inv].reshape(array.shape)


def assign_shm(data: bytes, tag: str) -> None:
    print(f"assign {tag} ...", flush=True)
    with open(f"/dev/shm/{tag}", "wb") as f:
        f.write(data)


def read_shm(tag: str) -> SharedMemory:
    shm = SharedMemory(name=tag, create=False)
    unregister(shm._name, "shared_memory")
    return shm


def clean_shm(tag: str) -> None:
    shm = SharedMemory(name=tag, create=False)
    shm.close()
    shm.unlink()


def assign_np_shm(array: np.ndarray, tag: str) -> None:
    assign_shm(np.ascontiguousarray(array).data, f"{tag}-data")
    assign_shm(np.array(array.shape).data, f"{tag}-shape")
    assign_shm(str(array.dtype).encode("utf8"), f"{tag}-dtype")


def read_np_shm(tag: str, bank: list[SharedMemory]) -> np.ndarray:
    data_shm: SharedMemory = read_shm(f"{tag}-data")
    shape_shm: SharedMemory = read_shm(f"{tag}-shape")
    dtype_shm: SharedMemory = read_shm(f"{tag}-dtype")

    shape_bytes: bytes = shape_shm.buf.tobytes()
    array_dtype: str = dtype_shm.buf.tobytes().decode("utf8")
    array_shape = np.ndarray(len(shape_bytes) >> 3, dtype=int, buffer=shape_bytes)

    bank.extend([data_shm, shape_shm, dtype_shm])
    return np.ndarray(array_shape, dtype=array_dtype, buffer=data_shm.buf)


def clean_np_shm(tag: str) -> None:
    for info in ("data", "shape", "dtype"):
        clean_shm(f"{tag}-{info}")


@dataclass
class AKGDataset(Dataset):
    vocab_size: int
    bos_id: int
    eos_id: int
    mask_id: int
    top_id: int
    top_down_id: int
    top_up_id: int

    nodes_text: np.ndarray
    nodes_start: np.ndarray
    nodes_end: np.ndarray
    edges_data: np.ndarray
    top_snt: np.ndarray

    aligns_node: np.ndarray
    tops_top: np.ndarray
    tops_boa: np.ndarray
    tops_pair: np.ndarray
    tops_against: np.ndarray
    tops_start: np.ndarray
    tops_end: np.ndarray
    tops_align_start: np.ndarray
    tops_align_end: np.ndarray
    tops_cross_start: np.ndarray
    tops_cross_end: np.ndarray
    cross_align: np.ndarray

    random_seed: int = 19260817
    max_length: int = 512
    pos_embed_size: int = 4
    max_comment_length: int = 0
    mask_ratio: float = 0.3

    shm_bank: ClassVar[list[SharedMemory]] = None

    @classmethod
    def load_train_eval(
        cls,
        dataset: str,
        random_seed: int | None = None,
        max_length: int | None = None,
        pos_embed_size: int | None = None,
        max_comment_length: int | None = None,
        mask_ratio: float | None = None,
    ) -> tuple["AKGDataset", "AKGDataset"]:
        main_dir: Path = Path("data") / "simplified"
        data_dir: Path = main_dir / dataset

        if random_seed is None:
            random_seed = cls.random_seed
        if max_length is None:
            max_length = cls.max_length
        if pos_embed_size is None:
            pos_embed_size = cls.pos_embed_size
        if max_comment_length is None:
            max_comment_length = cls.max_comment_length
        if mask_ratio is None:
            mask_ratio = cls.mask_ratio

        with (main_dir / "tokens.json").open("r", encoding="utf8") as f:
            tokens: dict[str, int] = load(f)
            vocab_size: int = tokens["vocab"]
            bos_id: int = tokens["bos"]
            eos_id: int = tokens["eos"]
            mask_id: int = tokens["mask"]
            top_id: int = tokens["top"]
            top_down_id: int = tokens["top_down"]
            top_up_id: int = tokens["top_up"]

        with np.load(data_dir / "main.npz") as data:
            nodes_text: np.ndarray = data["nodes_text"]
            nodes_start: np.ndarray = data["nodes_start"]
            nodes_end: np.ndarray = data["nodes_end"]
            edges_data: np.ndarray = data["edges_data"]
            top_snt: np.ndarray = data["top_snt"]

        train_eval: list["AKGDataset"] = []

        for split in ("train", "eval"):
            with np.load(data_dir / f"{split}.npz") as data:
                aligns_node: np.ndarray = data["aligns_node"]
                tops_top: np.ndarray = data["tops_top"]
                tops_boa: np.ndarray = data["tops_boa"]
                tops_pair: np.ndarray = data["tops_pair"]
                tops_against: np.ndarray = data["tops_against"]
                tops_start: np.ndarray = data["tops_start"]
                tops_end: np.ndarray = data["tops_end"]
                tops_align_start: np.ndarray = data["tops_align_start"]
                tops_align_end: np.ndarray = data["tops_align_end"]
                tops_cross_start: np.ndarray = data["tops_cross_start"]
                tops_cross_end: np.ndarray = data["tops_cross_end"]
                cross_align: np.ndarray = data["cross_align"]

            train_eval.append(
                AKGDataset(
                    vocab_size,
                    bos_id,
                    eos_id,
                    mask_id,
                    top_id,
                    top_down_id,
                    top_up_id,
                    nodes_text,
                    nodes_start,
                    nodes_end,
                    edges_data,
                    top_snt,
                    aligns_node,
                    tops_top,
                    tops_boa,
                    tops_pair,
                    tops_against,
                    tops_start,
                    tops_end,
                    tops_align_start,
                    tops_align_end,
                    tops_cross_start,
                    tops_cross_end,
                    cross_align,
                    random_seed=random_seed,
                    max_length=max_length,
                    pos_embed_size=pos_embed_size,
                    max_comment_length=max_comment_length,
                    mask_ratio=mask_ratio,
                )
            )

        return tuple(train_eval)

    @classmethod
    def from_shm(cls, tag: str) -> "AKGDataset":
        bank: list[SharedMemory] = []

        nodes_text: np.ndarray = read_np_shm(f"{tag}-nodes_text", bank)
        nodes_start: np.ndarray = read_np_shm(f"{tag}-nodes_start", bank)
        nodes_end: np.ndarray = read_np_shm(f"{tag}-nodes_end", bank)
        edges_data: np.ndarray = read_np_shm(f"{tag}-edges_data", bank)
        top_snt: np.ndarray = read_np_shm(f"{tag}-top_snt", bank)

        aligns_node: np.ndarray = read_np_shm(f"{tag}-aligns_node", bank)
        tops_top: np.ndarray = read_np_shm(f"{tag}-tops_top", bank)
        tops_boa: np.ndarray = read_np_shm(f"{tag}-tops_boa", bank)
        tops_pair: np.ndarray = read_np_shm(f"{tag}-tops_pair", bank)
        tops_against: np.ndarray = read_np_shm(f"{tag}-tops_against", bank)
        tops_start: np.ndarray = read_np_shm(f"{tag}-tops_start", bank)
        tops_end: np.ndarray = read_np_shm(f"{tag}-tops_end", bank)

        tops_align_start: np.ndarray = read_np_shm(f"{tag}-tops_align_start", bank)
        tops_align_end: np.ndarray = read_np_shm(f"{tag}-tops_align_end", bank)
        tops_cross_start: np.ndarray = read_np_shm(f"{tag}-tops_cross_start", bank)

        tops_cross_end: np.ndarray = read_np_shm(f"{tag}-tops_cross_end", bank)
        cross_align: np.ndarray = read_np_shm(f"{tag}-cross_align", bank)

        vocab_size: int
        bos_id: int
        eos_id: int
        mask_id: int
        top_id: int
        top_down_id: int
        top_up_id: int
        random_seed: int
        max_length: int
        pos_embed_size: int
        max_comment_length: int
        mask_ratio_percent: int

        (
            vocab_size,
            bos_id,
            eos_id,
            mask_id,
            top_id,
            top_down_id,
            top_up_id,
            random_seed,
            max_length,
            pos_embed_size,
            max_comment_length,
            mask_ratio_percent,
        ) = read_np_shm(f"{tag}-special", bank).tolist()

        mask_ratio: float = mask_ratio_percent / 100

        dataset = AKGDataset(
            vocab_size,
            bos_id,
            eos_id,
            mask_id,
            top_id,
            top_down_id,
            top_up_id,
            nodes_text,
            nodes_start,
            nodes_end,
            edges_data,
            top_snt,
            aligns_node,
            tops_top,
            tops_boa,
            tops_pair,
            tops_against,
            tops_start,
            tops_end,
            tops_align_start,
            tops_align_end,
            tops_cross_start,
            tops_cross_end,
            cross_align,
            random_seed=random_seed,
            max_length=max_length,
            pos_embed_size=pos_embed_size,
            max_comment_length=max_comment_length,
            mask_ratio=mask_ratio,
        )

        dataset.shm_bank = bank
        return dataset

    def to_shm(self) -> str:
        tag = "".join(choices("0123456789abcdef", k=16))

        assign_np_shm(self.nodes_text, f"{tag}-nodes_text")
        assign_np_shm(self.nodes_start, f"{tag}-nodes_start")
        assign_np_shm(self.nodes_end, f"{tag}-nodes_end")
        assign_np_shm(self.edges_data, f"{tag}-edges_data")
        assign_np_shm(self.top_snt, f"{tag}-top_snt")

        assign_np_shm(self.aligns_node, f"{tag}-aligns_node")
        assign_np_shm(self.tops_top, f"{tag}-tops_top")
        assign_np_shm(self.tops_boa, f"{tag}-tops_boa")
        assign_np_shm(self.tops_pair, f"{tag}-tops_pair")
        assign_np_shm(self.tops_against, f"{tag}-tops_against")
        assign_np_shm(self.tops_start, f"{tag}-tops_start")
        assign_np_shm(self.tops_end, f"{tag}-tops_end")
        assign_np_shm(self.tops_align_start, f"{tag}-tops_align_start")
        assign_np_shm(self.tops_align_end, f"{tag}-tops_align_end")
        assign_np_shm(self.tops_cross_start, f"{tag}-tops_cross_start")
        assign_np_shm(self.tops_cross_end, f"{tag}-tops_cross_end")
        assign_np_shm(self.cross_align, f"{tag}-cross_align")

        special: np.ndarray = np.array(
            [
                self.vocab_size,
                self.bos_id,
                self.eos_id,
                self.mask_id,
                self.top_id,
                self.top_down_id,
                self.top_up_id,
                self.random_seed,
                self.max_length,
                self.pos_embed_size,
                self.max_comment_length,
                round(self.mask_ratio * 100),
            ],
            dtype=int,
        )

        assign_np_shm(special, f"{tag}-special")
        return tag

    @staticmethod
    def clean_shm(tag: str) -> None:
        for name in (
            "nodes_text",
            "nodes_start",
            "nodes_end",
            "edges_data",
            "top_snt",
            "aligns_node",
            "tops_top",
            "tops_boa",
            "tops_pair",
            "tops_against",
            "tops_start",
            "tops_end",
            "tops_align_start",
            "tops_align_end",
            "tops_cross_start",
            "tops_cross_end",
            "cross_align",
            "special",
        ):
            clean_np_shm(f"{tag}-{name}")

    def __len__(self) -> int:
        return self.tops_top.shape[0]

    def __getitem__(self, index: int) -> dict[str, np.ndarray]:
        max_length: int = self.max_length

        top_indices: list[int] = []
        top_text_range: list[tuple[int, int]] = []
        input_ids: np.ndarray = np.array([self.bos_id], dtype=np.uint16)
        pivot: int = index

        candidate_comments: list[tuple[int, int]] = []
        comment_top_indices: list[int] = []
        comment_pairs: np.ndarray = np.zeros((0, 2), dtype=int)
        comment_labels: np.ndarray = np.array([], dtype=int)

        while input_ids.shape[0] < max_length:
            top_indices.append(pivot)
            top_input_ids: np.ndarray = self._get_top_snt(pivot)
            available_length: int = min(top_input_ids.shape[0], max_length - input_ids.shape[0])
            top_text_range.append((input_ids.shape[0], input_ids.shape[0] + available_length))
            input_ids = np.append(input_ids, top_input_ids[:available_length])

            if self.max_comment_length > 0:
                x: int = self.tops_pair[pivot].item()
                y = int(self.tops_against[pivot].item())

                if x != -1 and x not in [x[0] for x in candidate_comments]:
                    candidate_comments.append((x, y))

            pivot += 1
            if pivot >= self.tops_top.shape[0]:
                pivot = 0

            if self.tops_boa[pivot] and input_ids.shape[0] < max_length:
                input_ids = np.append(input_ids, self.eos_id)
                if input_ids.shape[0] >= max_length:
                    break

                if len(candidate_comments) > 0:
                    source_pos: int = input_ids.shape[0] - 1
                    target_pos: int = source_pos
                    comment_length: int = 0
                    np.random.default_rng(self.random_seed + pivot).shuffle(candidate_comments)

                    for comment_index, label in candidate_comments:
                        comment_input_ids: np.ndarray = self._get_top_snt(comment_index)

                        comment_length += comment_input_ids.shape[0] + 1
                        target_pos += comment_input_ids.shape[0] + 1

                        if (
                            comment_length > self.max_comment_length
                            or target_pos >= self.max_length
                        ):
                            break

                        comment_pairs = np.vstack([comment_pairs, [source_pos, target_pos]])
                        comment_labels = np.append(comment_labels, label)
                        comment_top_indices.append(comment_index)
                        top_text_range.append((input_ids.shape[0], target_pos))

                        input_ids = np.concatenate([input_ids, comment_input_ids, [self.eos_id]])
                        assert input_ids[source_pos] == self.eos_id, input_ids

                        assert input_ids[target_pos] == self.eos_id, (
                            input_ids,
                            comment_input_ids,
                            comment_length,
                            target_pos,
                        )

                candidate_comments = []

        special_tokens_pos: np.ndarray = np.nonzero(
            np.isin(input_ids, (self.bos_id, self.eos_id, 0))
        )[0]

        all_nodes: np.ndarray
        node_top_count: np.ndarray

        all_nodes, node_top_count = np.unique(
            np.concatenate([self._get_top_nodes(top_index) for top_index in top_indices]),
            return_counts=True,
        )

        num_nodes: int = all_nodes.shape[0]
        node_map: dict[int, int] = {x: i for i, x in enumerate(all_nodes)}
        top_link_nodes: np.ndarray = map_numpy(self.tops_top[top_indices], node_map)

        num_tops: int = len(top_indices)
        num_main_tops: int = num_tops - len(comment_top_indices)
        top_node_indices: np.ndarray = np.arange(num_tops) + num_nodes
        num_nodes += num_tops + 1

        node_attr: np.ndarray = np.concatenate(
            [self.nodes_text[all_nodes], np.full(num_tops + 1, self.top_id)]
        )

        node_mask_weight: np.ndarray = np.concatenate([node_top_count, np.zeros(num_tops + 1)])
        mask_pos_list: list[np.ndarray] = []

        for top_index, (start, stop) in zip(top_indices, top_text_range):
            top_cross_align: np.ndarray = self._get_top_cross_aligns(top_index).copy()
            top_cross_align[:, 1] += start
            top_cross_align = top_cross_align[top_cross_align[:, 1] < stop]
            top_cross_align[:, 0] = map_numpy(top_cross_align[:, 0], node_map)
            mask_pos_list.append(top_cross_align)

        node_text_mask_pos: np.ndarray = np.concatenate(mask_pos_list, axis=0)

        raw_edges: np.ndarray = np.concatenate(
            [self._get_out_edges(node_index) for node_index in all_nodes]
        )

        raw_edges = raw_edges[np.isin(raw_edges[:, 1], all_nodes)]
        raw_edges[:, :2] = map_numpy(raw_edges[:, :2], node_map)

        real_from: np.ndarray = np.concatenate(
            [raw_edges[:, 0], top_node_indices, np.full(num_tops, num_nodes - 1)]
        )

        real_to: np.ndarray = np.concatenate([raw_edges[:, 1], top_link_nodes, top_node_indices])

        edge_index: np.ndarray = np.vstack(
            [np.concatenate([real_from, real_to]), np.concatenate([real_to, real_from])]
        )

        edge_attr: np.ndarray = np.concatenate(
            [
                raw_edges[:, 2],
                np.full(num_tops << 1, self.top_down_id),
                raw_edges[:, 3],
                np.full(num_tops << 1, self.top_up_id),
            ]
        )

        edge_dir: np.ndarray = np.arange(real_from.shape[0])

        edge_mask_weight: np.ndarray = (
            node_mask_weight[edge_index[0]] * node_mask_weight[edge_index[1]]
        )

        adj_matrix = sp.csr_array(
            (np.ones(edge_index.shape[1]), edge_index), shape=(num_nodes, num_nodes)
        )

        degree: np.ndarray = np.array(adj_matrix.sum(axis=0)).clip(1) ** -0.5

        laplacian: sp.csr_array = sp.csr_array(
            sp.eye(adj_matrix.shape[0])
        ) - adj_matrix * degree * degree.reshape(-1, 1)

        eigenvalues: np.ndarray
        eigenvectors: np.ndarray
        pos_embed: np.ndarray

        def dense_way() -> np.ndarray:
            eigenvalues, eigenvectors = np.linalg.eig(laplacian.toarray())
            eigenvectors = eigenvectors[:, eigenvalues.real.argsort()]
            k: int = min(laplacian.shape[0], self.pos_embed_size + 1)
            return eigenvectors[:, 1:k].real

        if laplacian.shape[0] - 2 <= self.pos_embed_size:
            pos_embed = dense_way()
        else:
            try:
                eigenvalues, eigenvectors = eigs(
                    laplacian,
                    which="SR",
                    tol=1e-2,
                    k=min(laplacian.shape[0] - 2, self.pos_embed_size + 1),
                )

                eigenvectors = eigenvectors[:, eigenvalues.real.argsort()]
                pos_embed = eigenvectors[:, 1 : self.pos_embed_size + 1].real
            except ArpackNoConvergence as e:
                eigenvalues = e.eigenvalues
                eigenvectors = e.eigenvectors
            except Exception:
                pos_embed = dense_way()

        if pos_embed.shape[1] < self.pos_embed_size:
            pos_embed = np.concatenate(
                [
                    pos_embed,
                    np.zeros((pos_embed.shape[0], self.pos_embed_size - pos_embed.shape[1])),
                ],
                axis=1,
            )

        if pos_embed.shape[1] < self.pos_embed_size:
            pos_embed = np.concatenate(
                [
                    pos_embed,
                    np.zeros((pos_embed.shape[0], self.pos_embed_size - pos_embed.shape[1])),
                ],
                axis=1,
            )

        mask_seed: int = (self.random_seed << 1) + index
        input_ids_labels: np.ndarray
        node_attr_labels: np.ndarray
        edge_attr_labels: np.ndarray

        node_attr, node_attr_labels = self._mask_tokens(
            node_attr,
            self.mask_ratio / 2,
            node_mask_weight,
            np.array([], dtype=int),
            np.array([], dtype=int),
            mask_seed,
        )

        edge_attr, edge_attr_labels = self._mask_tokens(
            edge_attr,
            self.mask_ratio / 2,
            edge_mask_weight,
            np.array([], dtype=int),
            np.array([], dtype=int),
            mask_seed,
        )

        node_mask_flag: np.ndarray = node_attr_labels != 0
        node_mask_flag[edge_index[:, edge_attr_labels != 0].ravel()] = True

        text_premask_pos: np.ndarray = np.unique(
            node_text_mask_pos[node_mask_flag[node_text_mask_pos[:, 0]], 1]
        )

        input_ids, input_ids_labels = self._mask_tokens(
            input_ids,
            self.mask_ratio,
            np.ones_like(input_ids),
            special_tokens_pos,
            text_premask_pos,
            mask_seed,
        )

        main_tops: np.ndarray = top_node_indices[:num_main_tops]
        comment_tops: np.ndarray = top_node_indices[num_main_tops:]

        return {
            "input_ids": input_ids.astype(np.uint16),
            "comment_pairs": comment_pairs.astype(np.uint16),
            "main_top_mask": main_tops.astype(np.uint16),
            "comment_top_mask": comment_tops.astype(np.uint16),
            "node_attr": node_attr.astype(np.uint16),
            "edge_index": edge_index.T.astype(np.uint16),
            "edge_attr": edge_attr.astype(np.uint16),
            "edge_dir": edge_dir.astype(np.uint16),
            "pos": pos_embed.astype(np.float16),
            "input_ids_labels": input_ids_labels.astype(np.uint16),
            "node_attr_labels": node_attr_labels.astype(np.uint16),
            "edge_attr_labels": edge_attr_labels.astype(np.uint16),
            "match_labels": comment_labels.astype(np.uint8),
        }

    def _get_top_snt(self, top_index: int) -> np.ndarray:
        snt_start: int = self.tops_start[top_index]
        snt_end: int = self.tops_end[top_index]
        return self.top_snt[snt_start:snt_end]

    def _get_top_nodes(self, top_index: int) -> np.ndarray:
        align_start: int = self.tops_align_start[top_index]
        align_end: int = self.tops_align_end[top_index]
        return self.aligns_node[align_start:align_end]

    def _get_top_cross_aligns(self, top_index: int) -> np.ndarray:
        cross_start: int = self.tops_cross_start[top_index]
        cross_end: int = self.tops_cross_end[top_index]
        return self.cross_align[cross_start:cross_end, :]

    def _get_out_edges(self, node_index: int) -> np.ndarray:
        edge_start: int = self.nodes_start[node_index]
        edge_end: int = self.nodes_end[node_index]
        return self.edges_data[edge_start:edge_end, :]

    def _mask_tokens(
        self,
        inputs: np.ndarray,
        mask_ratio: float,
        mask_weight: np.ndarray,
        special_tokens_pos: np.ndarray,
        premask_pos: np.ndarray,
        seed: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        mask: np.ndarray = np.zeros_like(inputs, dtype=bool)
        mask[premask_pos] = True
        mask_weight = mask_weight.copy()
        mask_weight[premask_pos] = 0

        num_tokens: int = inputs.shape[0]
        mask_ratio = max(0, mask_ratio - premask_pos.shape[0] / num_tokens)
        rng: np.random.Generator = np.random.default_rng(seed)

        if mask_ratio > 0:
            mask_prob: np.ndarray = np.minimum(mask_weight / mask_weight.mean() * mask_ratio, 1.0)
            mask_prob[special_tokens_pos] = 0
            mask |= rng.random(inputs.shape) < mask_prob

        labels: np.ndarray = inputs + 100
        labels[~mask] = 0

        rep_mask: np.ndarray = rng.random(num_tokens) < mask * 0.8
        inputs[rep_mask] = self.mask_id
        mask &= ~rep_mask

        rnd_mask: np.ndarray = rng.random(num_tokens) < mask * 0.5
        inputs[rnd_mask] = rng.integers(self.vocab_size, size=rnd_mask.sum())
        return inputs, labels
