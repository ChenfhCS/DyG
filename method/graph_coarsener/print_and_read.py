"""Tools for data reading and writing."""

import os
import json
import pandas as pd
import networkx as nx
import pickle as pkl
import numpy as np
from texttable import Texttable

current_path = os.path.abspath(os.path.dirname(os.getcwd()))

def load_data(args):
    r"""
    Load graphs with given the dataset name
    param:
        dataset_name: dataset's name
        platform: converse graph to which platform. dgl or pyg
        timesteps: the num of graphs for experiments
        features (bool): whether generate features with one-hot encoding
        graph_id: which graphs should be loaded
    """
    # dataset_name = args.dataset
    # timesteps = args.timesteps
    dataset_name = args['dataset']
    timesteps = args['timesteps']

    new_graphs = []
    # load networkx graphs data
    graph_path = current_path + '/Data/{}/data/{}'.format(dataset_name, 'graphs.npz')
    if dataset_name == 'Enron':
        with open(graph_path, "rb") as f:
            graphs = pkl.load(f)
    else:
        graphs = np.load(graph_path, allow_pickle=True, encoding='latin1')['graph']
    
    graphs = graphs[3: 3+timesteps]

    return graphs

def argument_printer(args):
    """
    Function to print the arguments in a nice tabular format.
    :param args: Parameters used for the model.
    """
    args = vars(args)
    keys = sorted(args.keys())
    t = Texttable() 
    t.add_rows([["Parameter", "Value"]])
    t.add_rows([[k.replace("_", " ").capitalize(), args[k]] for k in keys])
    print(t.draw())

def graph_concat(args, graphs):
    """
    Function to concatenate snapshots along the temporal dimension
    :param graphs: list of snapshots
    """
    G = nx.Graph()
    current_number_nodes = []
    for i in range(len(graphs)):
        snap_id = [i]
        node_idx = {node: {'orig_id': node} for node in list(graphs[i].nodes())}
        nx.set_node_attributes(graphs[i], snap_id, "snap_id")
        nx.set_node_attributes(graphs[i], node_idx)
        attr = {edge: {"type": 'str'} for edge in graphs[i].edges()}
        nx.set_edge_attributes(graphs[i], attr)
        # print('Snapshot {} has nodes {} and edges {}'.format(i, graphs[i].number_of_nodes(), graphs[i].number_of_edges()))
        G = nx.disjoint_union(G, graphs[i])
    for node in list(G):
        snap_id = G.nodes[node]['snap_id'][0]
        tem_idx = 0
        for i in range(snap_id):
            tem_neighbor = node - graphs[snap_id - 1 - i].number_of_nodes() - tem_idx
            tem_idx += graphs[snap_id - 1 - i].number_of_nodes()
            if G.nodes[tem_neighbor]['snap_id'][0] == snap_id - 1 - i:
                G.add_edge(tem_neighbor, node)
                attr = {(tem_neighbor, node): {"type": 'tem'}}
                nx.set_edge_attributes(G, attr)
    
    # print(G.nodes[60]['orig_id'])
    # print(G.number_of_nodes(), G.number_of_edges())
    return G

def graph_reader(args, graphs):
    """
    Function to read graph from input path.
    :param input_path: Graph read into memory.
    :return graph: Networkx graph.
    """
    if not graphs:
        graphs = load_data(args)
    graph = graphs[1]
    full_graph = graph_concat(args, graphs)
    print('Load graph with total nodes: {} and edges: {}'.format(full_graph.number_of_nodes(), full_graph.number_of_edges()))
    return graphs, full_graph

    edges = pd.read_csv(input_path)
    graph = nx.from_edgelist(edges.values.tolist())
    return graph

def json_dumper(data, path):
    """
    Function to save a JSON to disk.
    :param data: Dictionary of cluster memberships.
    :param path: Path for dumping the JSON.
    """
    with open(path, 'w') as outfile:
        json.dump(data, outfile)
