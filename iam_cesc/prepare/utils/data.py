from dataclasses import dataclass
from json import load
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import ArpackNoConvergence, eigs
from torch.utils.data import Dataset


def map_numpy(array: np.ndarray, map: dict[int, int]) -> np.ndarray:
    unique: np.ndarray
    inv: np.ndarray
    unique, inv = np.unique(array, return_inverse=True)
    return np.array([map[x] for x in unique])[inv].reshape(array.shape)


@dataclass
class IAMDataset(Dataset):
    bos_id: int
    eos_id: int
    top_id: int
    top_down_id: int
    top_up_id: int

    nodes_text: np.ndarray
    nodes_start: np.ndarray
    nodes_end: np.ndarray
    edges_data: np.ndarray
    aligns_node: np.ndarray
    tops_top: np.ndarray
    tops_start: np.ndarray
    tops_end: np.ndarray
    tops_align_start: np.ndarray
    tops_align_end: np.ndarray
    top_snt: np.ndarray

    tasks_topic: np.ndarray
    tasks_candidate: np.ndarray
    tasks_label: np.ndarray

    max_length: int = 512
    pos_embed_size: int = 4

    @classmethod
    def load_train_dev_test(
        cls, dataset: str, max_length: int | None = None,
        pos_embed_size: int | None = None
    ) -> tuple['IAMDataset', 'IAMDataset', 'IAMDataset']:
        main_dir: Path = Path('data') / 'simplified'
        data_dir: Path = main_dir / dataset

        if max_length is None:
            max_length = cls.max_length
        if pos_embed_size is None:
            pos_embed_size = cls.pos_embed_size

        with (main_dir / 'tokens.json').open('r', encoding='utf8') as f:
            tokens: dict[str, int] = load(f)
            bos_id: int = tokens['bos']
            eos_id: int = tokens['eos']
            top_id: int = tokens['top']
            top_down_id: int = tokens['top_down']
            top_up_id: int = tokens['top_up']

        with np.load(data_dir / 'main.npz') as data:
            nodes_text: np.ndarray = data['nodes_text']
            nodes_start: np.ndarray = data['nodes_start']
            nodes_end: np.ndarray = data['nodes_end']
            edges_data: np.ndarray = data['edges_data']
            aligns_node: np.ndarray = data['aligns_node']
            tops_top: np.ndarray = data['tops_top']
            tops_start: np.ndarray = data['tops_start']
            tops_end: np.ndarray = data['tops_end']
            tops_align_start: np.ndarray = data['tops_align_start']
            tops_align_end: np.ndarray = data['tops_align_end']
            top_snt: np.ndarray = data['top_snt']

        datasets: list[IAMDataset] = []

        for mode in ('train', 'dev', 'test'):
            with np.load(data_dir / f'{mode}.npz') as data:
                tasks_topic: np.ndarray = data['tasks_topic']
                tasks_candidate: np.ndarray = data['tasks_candidate']
                tasks_label: np.ndarray = data['tasks_label']

            datasets.append(IAMDataset(
                bos_id, eos_id, top_id, top_down_id, top_up_id, nodes_text,
                nodes_start, nodes_end, edges_data, aligns_node, tops_top,
                tops_start, tops_end, tops_align_start, tops_align_end,
                top_snt, tasks_topic, tasks_candidate, tasks_label,
                max_length=max_length, pos_embed_size=pos_embed_size
            ))

        return tuple(datasets)

    def __len__(self) -> int:
        return self.tasks_topic.shape[0]

    def __getitem__(self, index: int) -> dict[str, np.ndarray]:
        return self._gen_data([
            self.tasks_topic[index].item(), self.tasks_candidate[index].item()
        ], self.tasks_label[index].item())

    def _gen_data(
        self, top_indices: list[int], label: int
    ) -> dict[str, np.ndarray]:
        top_text_range: list[tuple[int, int]] = []
        input_ids: np.ndarray = np.array([self.bos_id], dtype=np.uint16)

        for index in top_indices:
            top_input_ids: np.ndarray = self._get_top_snt(index)

            top_text_range.append((
                input_ids.shape[0], input_ids.shape[0] + top_input_ids.shape[0]
            ))

            input_ids = np.append(input_ids, top_input_ids)
            input_ids = np.append(input_ids, self.eos_id)

        all_nodes: np.ndarray = np.unique(np.concatenate(
            [self._get_top_nodes(top_index) for top_index in top_indices]
        ))

        num_nodes: int = all_nodes.shape[0]
        node_map: dict[int, int] = {x: i for i, x in enumerate(all_nodes)}

        top_link_nodes: np.ndarray = \
            map_numpy(self.tops_top[top_indices], node_map)

        num_tops: int = top_link_nodes.shape[0]
        top_node_indices: np.ndarray = np.arange(num_tops) + num_nodes
        num_nodes += num_tops + 1

        node_attr: np.ndarray = np.concatenate(
            [self.nodes_text[all_nodes], np.full(num_tops + 1, self.top_id)]
        )

        sub_edges: np.ndarray = np.concatenate(
            [self._get_out_edges(node_index) for node_index in all_nodes]
        )

        sub_edges = sub_edges[np.isin(sub_edges[:, 1], all_nodes)]
        sub_edges[:, :2] = map_numpy(sub_edges[:, :2], node_map)

        real_from: np.ndarray = np.concatenate([
            sub_edges[:, 0], top_node_indices, np.full(num_tops, num_nodes - 1)
        ])

        real_to: np.ndarray = \
            np.concatenate([sub_edges[:, 1], top_link_nodes, top_node_indices])

        edge_index: np.ndarray = np.vstack([
            np.concatenate([real_from, real_to]),
            np.concatenate([real_to, real_from])
        ])

        edge_attr: np.ndarray = np.concatenate([
            sub_edges[:, 2], np.full(num_tops << 1, self.top_down_id),
            sub_edges[:, 3], np.full(num_tops << 1, self.top_up_id)
        ])

        adj_matrix = sp.csr_array(
            (np.ones(edge_index.shape[1]), edge_index),
            shape=(num_nodes, num_nodes)
        )

        degree: np.ndarray = np.array(adj_matrix.sum(axis=0)).clip(1) ** -0.5
        laplacian: sp.csr_array = sp.csr_array(sp.eye(adj_matrix.shape[0])) \
            - adj_matrix * degree * degree.reshape(-1, 1)

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
                    laplacian, which='SR', tol=1e-2,
                    k=min(laplacian.shape[0] - 2, self.pos_embed_size + 1)
                )

                eigenvectors = eigenvectors[:, eigenvalues.real.argsort()]
                pos_embed = eigenvectors[:, 1:self.pos_embed_size + 1].real
            except ArpackNoConvergence as e:
                eigenvalues = e.eigenvalues
                eigenvectors = e.eigenvectors
            except Exception:
                pos_embed = dense_way()

        if pos_embed.shape[1] < self.pos_embed_size:
            pos_embed = np.concatenate([pos_embed, np.zeros(
                (pos_embed.shape[0], self.pos_embed_size - pos_embed.shape[1])
            )], axis=1)

        return {
            'input_ids': input_ids.astype(np.uint16),
            'node_attr': node_attr.astype(np.uint16),
            'edge_index': edge_index.astype(np.uint16),
            'edge_attr': edge_attr.astype(np.uint16),
            'pos': pos_embed.astype(np.float32),
            'labels': np.array([label], dtype=np.int8),
        }

    def _get_top_snt(self, top_index: int) -> np.ndarray:
        snt_start: int = self.tops_start[top_index]
        snt_end: int = self.tops_end[top_index]
        return self.top_snt[snt_start:snt_end]

    def _get_top_nodes(self, top_index: int) -> np.ndarray:
        align_start: int = self.tops_align_start[top_index]
        align_end: int = self.tops_align_end[top_index]
        return self.aligns_node[align_start:align_end]

    def _get_out_edges(self, node_index: int) -> np.ndarray:
        edge_start: int = self.nodes_start[node_index]
        edge_end: int = self.nodes_end[node_index]
        return self.edges_data[edge_start:edge_end, :]
