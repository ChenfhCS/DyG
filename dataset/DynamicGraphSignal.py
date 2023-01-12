import torch
import numpy as np
from typing import List, Union
from torch_geometric.data import Data
from .util import train_test_split

# Edge_Indices = List[Union[np.ndarray, None]]
# Edge_Weights = List[Union[np.ndarray, None]]
# Node_Features = List[Union[np.ndarray, None]]
# Targets = List[Union[np.ndarray, None]]
# Additional_Features = List[np.ndarray]


class DynamicGraphTemporalSignal(object):
    r"""A data iterator object to contain a dynamic graph with a
    changing edge set and weights . The feature set and node labels
    (target) are also dynamic. The iterator returns a single discrete temporal
    snapshot for a time period (e.g. day or week). This single snapshot is a
    Pytorch Geometric Data object. Between two temporal snapshots the edges,
    edge weights, target matrices and optionally passed attributes might change.
    Args:
        edge_indices (List of Numpy arrays): List of edge index tensors.
        edge_weights (List of Numpy arrays): List of edge weight tensors.
        features (List of Numpy arrays): List of node feature tensors.
        targets (List of Numpy arrays): List of node label (target) tensors.
        **kwargs (optional List of Numpy arrays): List of additional attributes.
    """

    def __init__(
        self,
        raw_graphs,
        edge_indices,
        edge_weights,
        features,
        sample_pos,
        sample_neg,
        **kwargs
    ):
        self.raw_graphs = raw_graphs
        self.edge_indices = edge_indices
        self.edge_weights = edge_weights
        self.features = features
        self.sample_pos = sample_pos
        self.sample_neg = sample_neg
        self.additional_feature_keys = []
        for key, value in kwargs.items():
            setattr(self, key, value)
            self.additional_feature_keys.append(key)
        self._check_temporal_consistency()
        self._set_snapshot_count()

    def _check_temporal_consistency(self):
        assert len(self.edge_indices) == len(
            self.edge_weights
        ), "Temporal dimension inconsistency."
        assert len(self.features) == len(
            self.edge_weights
        ), "Temporal dimension inconsistency."
        for key in self.additional_feature_keys:
            assert len(self.features) == len(
                getattr(self, key)
            ), "Temporal dimension inconsistency."

    def _set_snapshot_count(self):
        self.snapshot_count = len(self.features)

    def _get_raw_graph(self, time_index: int):
        if self.edge_indices[time_index] is None:
            return self.raw_graphs[time_index]
        else:
            return self.raw_graphs[time_index]

    def _get_edge_index(self, time_index: int):
        if self.edge_indices[time_index] is None:
            return self.edge_indices[time_index]
        else:
            return self.edge_indices[time_index]

    def _get_edge_weight(self, time_index: int):
        if self.edge_weights[time_index] is None:
            return self.edge_weights[time_index]
        else:
            return self.edge_weights[time_index]

    def _get_features(self, time_index: int):
        if self.features[time_index] is None:
            return self.features[time_index]
        else:
            return torch.FloatTensor(self.features[time_index])

    def _get_samples(self, time_index: int):
        '''
        samples
        '''
        pos_item = self.sample_pos[time_index]
        neg_item = self.sample_neg[time_index]

        train_samples, train_labels, val_samples, val_labels,  test_samples, test_labels = train_test_split(pos_item, neg_item)
        return train_samples, train_labels, val_samples, val_labels,  test_samples, test_labels

    # def num_vertices(self) -> int:
    #     return self._dgraph.num_vertices()

    # def num_source_vertices(self) -> int:
    #     return self._dgraph.num_source_vertices()

    # def max_vertex_id(self) -> int:
    #     return self._dgraph.max_vertex_id()

    # def num_edges(self) -> int:
    #     return self._dgraph.num_edges()

    # def out_degree(self, vertexs: np.ndarray) -> np.ndarray:
    #     return self._dgraph.out_degree(vertexs)

    # def nodes(self) -> np.ndarray:
    #     """
    #     Return the nodes of the graph.
    #     """
    #     return self._dgraph.nodes()

    # def src_nodes(self) -> np.ndarray:
    #     """
    #     Return the source nodes of the graph.
    #     """
    #     return self._dgraph.src_nodes()

    # def edges(self) -> np.ndarray:
    #     """
    #     Return the edges of the graph.
    #     """
    #     return self._dgraph.edges()

    # def get_temporal_neighbors(self, vertex: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    #     """
    #     Return the neighbors of the specified vertex. The neighbors are sorted
    #     by timestamps in decending order.

    #     Note that this function is inefficient and should be used sparingly.

    #     Args:
    #         vertex: the vertex to get neighbors for.

    #     Returns: A tuple of (target_vertices, timestamps, edge_ids)
    #     """
    #     return self._dgraph.get_temporal_neighbors(vertex)

    # def avg_linked_list_length(self) -> float:
    #     """
    #     Return the average linked list length.
    #     """
    #     return self._dgraph.avg_linked_list_length()

    # def get_graph_memory_usage(self) -> int:
    #     """
    #     Return the graph memory usage of the graph in bytes.
    #     """
    #     return self._dgraph.get_graph_memory_usage()

    # def get_metadata_memory_usage(self) -> int:
        """
        Return the metadata memory usage of the graph in bytes.
        """
        return self._dgraph.get_metadata_memory_usage()

    def _get_additional_feature(self, time_index: int, feature_key: str):
        feature = getattr(self, feature_key)[time_index]
        if feature.dtype.kind == "i":
            return torch.LongTensor(feature)
        elif feature.dtype.kind == "f":
            return torch.FloatTensor(feature)

    def _get_additional_features(self, time_index: int):
        additional_features = {
            key: self._get_additional_feature(time_index, key)
            for key in self.additional_feature_keys
        }
        return additional_features

    def __getitem__(self, time_index: Union[int, slice]):
        if isinstance(time_index, slice):
            snapshot = DynamicGraphTemporalSignal(
                self.edge_indices[time_index],
                self.edge_weights[time_index],
                self.features[time_index],
                # self.targets[time_index],
                **{key: getattr(self, key)[time_index] for key in self.additional_feature_keys}
            )
        else:
            raw_graph = self._get_raw_graph(time_index)
            x = self._get_features(time_index)
            edge_index = self._get_edge_index(time_index)
            edge_weight = self._get_edge_weight(time_index)
            train_samples, train_labels, val_samples, val_labels, test_samples, test_labels = self._get_samples(time_index)
            # y = self._get_target(time_index)
            # additional_features = self._get_additional_features(time_index)
            additional_features = {'raw_graph': raw_graph,
                                    'train_samples': train_samples, 
                                    'train_labels': train_labels,
                                    'val_samples': val_samples,
                                    'val_labels': val_labels,
                                    'test_samples': test_samples,
                                    'test_labels': test_labels
            }
            snapshot = Data(x=x, edge_index=edge_index, edge_attr=edge_weight, **additional_features)
        return snapshot

    def __next__(self):
        if self.t < len(self.features):
            snapshot = self[self.t]
            self.t = self.t + 1
            return snapshot
        else:
            self.t = 0
            raise StopIteration

    def __iter__(self):
        self.t = 0
        return self