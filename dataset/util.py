import torch
import numpy as np
import networkx as nx
from sklearn.model_selection import train_test_split

# Define a function to remap node ids
def remap(snapshot):
    node_mapping = {n: i for i, n in enumerate(sorted(snapshot.nodes()))}
    snapshot_remap = nx.relabel_nodes(snapshot, node_mapping, copy=True)
    return snapshot_remap

def generate_degree_feats(graphs, adjs):
    feats = []
    for (graph, adj) in zip(graphs, adjs):
        # cur_out, cur_in = _get_degree_from_adj(adj,graph.number_of_nodes())
        in_degree = graph.in_degree()
        out_degree = graph.out_degree()
        in_degree_list = list(dict(in_degree).values())
        out_degree_list = list(dict(out_degree).values())
        out_tensor = torch.tensor(out_degree_list, dtype=torch.float32)
        in_tensor = torch.tensor(in_degree_list, dtype=torch.float32)
        # add a dimension
        out_tensor = torch.unsqueeze(out_tensor, dim=1)
        in_tensor = torch.unsqueeze(in_tensor, dim=1)
        # print(out_tensor, in_tensor)
        # out_tensor = torch.tensor(cur_out)
        # in_tensor = torch.tensor(cur_in)
        feat = torch.cat([in_tensor, out_tensor], dim = 1)
        feats.append(feat)
    return feats

def create_edge_samples(graph, val_mask_fraction=0.1, test_mask_fraction=0.1):
    r"""
    Generate postive and negative edges from next_graph (i.e., last graph)
    1. Postive edges: the edges in the next_graph while both the source and target nodes exist in 'graph' (i.e., previous graph)
    2. Negative edges: the edges in the next_graph but the source and target nodes do not exist in 'graph' (i.e., previous graph)
    """
    edges = np.array(list(nx.Graph(graph).edges()))
    edges_positive = []   # Constraint to restrict new links to existing nodes.
    Num_of_edges = 10000
    for idx, e in enumerate(edges):
        if idx <= Num_of_edges:
        # if next_graph.has_edge(e[0], e[1]) and idx <= Num_of_edges:
            edges_positive.append(e)
    edges_positive = np.array(edges_positive) # [E, 2]
    edges_negative = _negative_sample(edges_positive, graph.number_of_nodes(), graph)
    # train_edges_pos, test_pos, train_edges_neg, test_neg = train_test_split(edges_positive,
    #         edges_negative, test_size=val_mask_fraction+test_mask_fraction)
    # val_edges_pos, test_edges_pos, val_edges_neg, test_edges_neg = train_test_split(test_pos,
    #         test_neg, test_size=test_mask_fraction/(test_mask_fraction+val_mask_fraction))
    return edges_positive, edges_negative
    return train_edges_pos, train_edges_neg, val_edges_pos, val_edges_neg, test_edges_pos, test_edges_neg

def _negative_sample(edges_pos, nodes_num, graph):
    edges_neg = []
    while len(edges_neg) < len(edges_pos):
        idx_i = np.random.randint(0, nodes_num)
        idx_j = np.random.randint(0, nodes_num)
        if idx_i == idx_j:
            continue
        if graph.has_edge(idx_i, idx_j) or graph.has_edge(idx_j, idx_i):
            continue
        if edges_neg:
            if [idx_i, idx_j] in edges_neg or [idx_j, idx_i] in edges_neg:
                continue
        edges_neg.append([idx_i, idx_j])
    return edges_neg

def train_test_split(sample_pos, sample_neg, train_ratio: float = 0.6, val_ratio: float = 0.2):
    num_train_samples_pos = int(len(sample_pos) * train_ratio)
    num_train_samples_neg = int(len(sample_neg) * train_ratio)

    num_val_samples_pos = int(len(sample_pos) * val_ratio)
    num_val_samples_neg = int(len(sample_neg) * val_ratio)

    train_sample_pos = sample_pos[0:num_train_samples_pos]
    val_sample_pos = sample_pos[num_train_samples_pos:num_train_samples_pos+num_val_samples_pos]
    test_sample_pos = sample_pos[num_train_samples_pos+num_val_samples_pos:]

    train_sample_neg = sample_neg[0:num_train_samples_neg]
    val_sample_neg = sample_neg[num_train_samples_neg:num_train_samples_neg+num_val_samples_neg]
    test_sample_neg = sample_neg[num_train_samples_neg+num_val_samples_neg:]

    train_pos_labels = np.array([[1,0] for i in range(len(train_sample_pos))])
    train_neg_labels = np.array([[0,1] for i in range(len(train_sample_neg))])
    val_pos_labels = np.array([0] * len(val_sample_pos))
    val_neg_labels = np.array([1] * len(val_sample_neg))
    test_pos_labels = np.array([0] * len(test_sample_pos))
    test_neg_labels = np.array([1] * len(test_sample_neg))

    train_samples = torch.tensor(np.vstack((train_sample_pos, train_sample_neg)))  # train_pos_feats and train_neg_feats are 2-dim numpy matrix, stack them to a new numpy matrix via vstack()
    train_labels = torch.tensor(np.vstack((train_pos_labels, train_neg_labels)), dtype=torch.float32)  # train_pos_labels and train_neg_labels are 1-dim numpy array
    val_samples = torch.tensor(np.vstack((val_sample_pos, val_sample_neg)))
    val_labels = torch.tensor(np.append(val_pos_labels, val_neg_labels), dtype=torch.int32)
    test_samples = torch.tensor(np.vstack((test_sample_pos, test_sample_neg)))
    test_labels = torch.tensor(np.append(test_pos_labels, test_neg_labels), dtype=torch.int32)

    return train_samples, train_labels, val_samples, val_labels,  test_samples, test_labels