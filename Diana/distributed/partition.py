import torch
import torch_geometric
import os, sys
import time
sys.path.append("..") 

from method.graph_coarsener.STA_LabelPropagation import propagation
from .utils import (graph_concat, get_edges, chunk_generation, assignment,
                    get_pyg_full_graph, sequence_generation, snapshots_generation,
                    get_neighbors, assignment_advance)

class partitioner:
    """
    Partition module in Diana
    """
    def __init__(self, args, dynamic_graph, num_gpus, type):
        self._args = args
        self._dynamic_graph = dynamic_graph
        self._num_gpus = num_gpus
        self._type = type

        self.full_graph = graph_concat(self._dynamic_graph)
        self.num_nodes = len(self.full_graph.nodes())
        self.spatial_edge_index, self.temporal_edge_index = get_edges(self.full_graph)
        self.all_neighbors = get_neighbors(self.full_graph)
        self.pyg_full_graph = get_pyg_full_graph(self.full_graph)
        self.train_samples_list = [graph.train_samples for graph in self._dynamic_graph]
        self.test_samples_list = [graph.test_samples for graph in self._dynamic_graph]
        self.train_labels_list = [graph.train_labels for graph in self._dynamic_graph]
        self.test_labels_list = [graph.test_labels for graph in self._dynamic_graph]
    
    def partition(self):
        if self._type == 'PGC':
            return self._PGC_partition()
        elif self._type == 'PSS':
            return self._PSS_partition()
        elif self._type == 'PTS':
            return self._PTS_partition()
        else:
            return 0
    
    def _PGC_partition(self):
        start_time = time.time()
        new_graph = propagation(self._args, self.full_graph)
        print(f'label propagation time {time.time() - start_time}')

        start_time = time.time()
        graph_chunks = chunk_generation(new_graph, [graph.x for graph in self._dynamic_graph], self.spatial_edge_index, self.temporal_edge_index)    
        print(f'chunk generation time {time.time() - start_time}')

        start_time = time.time()
        # chunk_gpu_map = assignment_advance(graph_chunks, self._num_gpus)
        chunk_gpu_map = assignment(graph_chunks, self._num_gpus)
        print(f'chunk assignment time {time.time() - start_time}')
        return graph_chunks, chunk_gpu_map

    def _PSS_partition(self):
        start_time = time.time()
        spatial_snapshots = snapshots_generation(self.full_graph, [graph.x for graph in self._dynamic_graph], self.spatial_edge_index, self.temporal_edge_index)    
        print(f'snapshot generation time {time.time() - start_time}')

        start_time = time.time()
        sequence_gpu_map = assignment(spatial_snapshots, self._num_gpus)
        print(f'snapshot assignment time {time.time() - start_time}')
        return spatial_snapshots, sequence_gpu_map
    
    def _PTS_partition(self):
        start_time = time.time()
        temporal_sequences = sequence_generation(self.full_graph, [graph.x for graph in self._dynamic_graph], self.spatial_edge_index, self.temporal_edge_index)    
        print(f'sequence generation time {time.time() - start_time}')

        start_time = time.time()
        sequence_gpu_map = assignment(temporal_sequences, self._num_gpus)
        print(f'sequence assignment time {time.time() - start_time}')
        return temporal_sequences, sequence_gpu_map
