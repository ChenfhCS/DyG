import numpy as np
import networkx as nx
import torch
import scipy.sparse as sp
import os, sys
sys.path.append("..") 
current_path = os.path.abspath(os.path.join(os.getcwd(), ".."))

from tqdm import tqdm
from ..DynamicGraphSignal import DynamicGraphTemporalSignal

# sys.path.append("..")
# import DynamicGraphTemporalSignal
from ..util import generate_degree_feats, create_edge_samples, remap

current_path = os.path.abspath(os.path.join(os.getcwd(), ".."))

class EpinionDatasetLoader(object):
    """A dataset of mobility and history of reported cases of Epinion
    in England NUTS3 regions, from 3 March to 12 of May. The dataset is
    segmented in days and the graph is directed and weighted. The graph
    indicates how many people moved from one region to the other each day,
    based on Facebook Data For Good disease prevention maps.
    The node features correspond to the number of Epinion cases
    in the region in the past **window** days. The task is to predict the
    number of cases in each node after 1 day. 
    """

    def __init__(self, timesteps: int = 10):
        self.timesteps = timesteps
        self._load_graph()

    def _load_graph(self):
        self._dataset = []
        pbar = tqdm(range(self.timesteps), desc='Loading snapshots', leave=False)
        for snapshot_id in pbar:
            graph_path = current_path + f'/dataset/Epinion/data/snapshots/snapshot_{snapshot_id}.gpickle'
            snapshot = nx.read_gpickle(graph_path)
            if snapshot_id > 0:
                previous_snapshot = self._dataset[snapshot_id-1]
                snapshot.add_nodes_from(previous_snapshot.nodes(data=True))
                self._dataset[snapshot_id-1] = remap(previous_snapshot)
            self._dataset.append(snapshot)
        self._dataset[-1] = remap(self._dataset[-1])

    def _get_edges_and_weights(self):
        self._edges = []
        self._edge_weights = []
        for time in range(len(self._dataset)):
            adj_sp = nx.adjacency_matrix(self._dataset[time]).tocoo()
            adj = torch.sparse.LongTensor(torch.LongTensor([adj_sp.row.tolist(), adj_sp.col.tolist()]),
                              torch.LongTensor(adj_sp.data.astype(np.int32))).coalesce()
            self._edges.append(adj.indices())
            self._edge_weights.append(adj.values())
    
    def _get_nodes(self):
        self._nodes = []
        for t in range(len(self._dataset)):
            self._nodes.append(torch.tensor([node for node in self._dataset[t].nodes()]))

    def _get_edge_weights(self):
        return 0
    
    def _get_features(self):
        self.features = []
        feats_path = current_path + "/dataset/Epinion/data/eval_{}_feats/".format(str(len(self._dataset)))
        # print(feats_path)
        try:
            pbar = tqdm(self._dataset, desc='Loading features', leave=False)
            for time, _ in enumerate(pbar):
                path = feats_path+'no_{}.npz'.format(time)
                feat = sp.load_npz(path)
                if time == 0:
                    feat_array = feat.toarray()
                    num_feats = feat_array.shape[1]
                feat_coo = feat.tocoo()
                values = feat_coo.data
                indices = np.vstack((feat_coo.row, feat_coo.col))
                i = torch.LongTensor(indices)
                v = torch.FloatTensor(values)
                shape = feat_coo.shape
                feat_tensor_de = torch.sparse.FloatTensor(i, v, torch.Size(shape)).to_dense()
                self.features.append(feat_tensor_de)
        except IOError:
            # adj_matrices = list(map(lambda x: nx.adjacency_matrix(x), self._dataset))
            adj_matrices = [nx.adjacency_matrix(graph) for graph in self._dataset]
            self.features = generate_degree_feats(self._dataset, adj_matrices)
            folder_in = os.path.exists(feats_path)
            if not folder_in:
                os.makedirs(feats_path)
            pbar = tqdm(self.features, desc='Generating and saving features', leave=False)
            for id,feat in enumerate(pbar):
                feat_sp = sp.csr_matrix(feat)
                path = feats_path+'no_{}.npz'.format(id)
                sp.save_npz(path, feat_sp)
    
    def _get_samples(self):
        """
        Link prediction: generate positive and negative edges
        """
        self.sample_pos = []
        self.sample_neg = []
        pbar = tqdm(self._dataset, desc='Generating samples', leave=False)
        for time, _ in enumerate(pbar):
            postive, negative = create_edge_samples(self._dataset[time])
            self.sample_pos.append(postive)
            self.sample_neg.append(negative)
    
    def get_dataset(self, lags: int = 10) -> DynamicGraphTemporalSignal:
        """Returning the Epinion data iterator.

        Args types:
            * **lags** *(int)* - The number of time lags.
        Return types:
            * **dataset** *(DynamicGraphTemporalSignal)* - The Epinion dataset.
        """

        self.lags = lags
        self._get_nodes()
        self._get_edges_and_weights()
        self._get_features()
        self._get_samples()
        dataset = DynamicGraphTemporalSignal(
           self._dataset, self._nodes, self._edges, self._edge_weights, self.features, self.sample_pos, self.sample_neg
        )
        return dataset

if __name__ == '__main__':
    current_path = os.getcwd()
    dataset = EpinionDatasetLoader()
    dataset._get_edges_and_weights()
    print('_get_edges_and_weights() Pass!')
    dataset._get_features()
    print('_get_features() Pass!')