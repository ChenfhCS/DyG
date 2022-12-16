import os
from statistics import mean
import scipy
import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
import torch
from tqdm import tqdm

current_path = os.getcwd()

def _generate_degree_feats(graphs, adjs):
    feats = []
    for (graph, adj) in zip(graphs, adjs):
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

def _normalize_graph_gcn(adj):
    r"""GCN-based normalization of adjacency matrix 
    (scipy sparse format). Output is in tuple format
    """
    adj = sp.coo_matrix(adj, dtype=np.float32)
    adj_ = adj + sp.eye(adj.shape[0], dtype=np.float32)
    rowsum = np.array(adj_.sum(1), dtype=np.float32)
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten(), dtype=np.float32)
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
    return adj_normalized

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
    dataset_name = args['dataset']
    timesteps = args['timesteps']

    new_graphs = []
    # load networkx graphs data
    current_path = os.path.abspath(os.path.join(os.getcwd(), ".."))
    graph_path = current_path + '/dataset/{}/data/{}'.format(dataset_name, 'graphs.npz')
    if dataset_name == 'Enron':
        with open(graph_path, "rb") as f:
            graphs = pkl.load(f)
    else:
        graphs = np.load(graph_path, allow_pickle=True, encoding='latin1')['graph']
    
    graphs = graphs[3: 3+timesteps]

    return graphs

def generate_graphs(args, graphs):
    # get num of nodes for each snapshot
    features = args['featureless']
    timesteps = args['timesteps']

    Nodes_info = []
    Edge_info = []

    # # execution time evaluation
    # start = 3
    # total_num = 20000
    # new_window = 10
    # average = int(total_num/new_window)
    # graphs_new = []
    # for i in range(new_window):
    #     nodes = list(graphs[start + i].nodes())
    #     nodes_new = nodes[: average]
    #     graph_new = graphs[start + i].subgraph(nodes_new)
    #     graphs_new.append(graph_new)

    # graphs=graphs_new
    # args['timesteps'] = len(graphs)
    # timesteps = args['timesteps']

    for i in range(args['timesteps']):
        Nodes_info.append(graphs[i].number_of_nodes())
        Edge_info.append(graphs[i].number_of_edges())
    args['nodes_info'] = Nodes_info
    args['edges_info'] = Edge_info

    # print(args['nodes_info'], args['edges_info'], args['timesteps'])

    adj_matrices = list(map(lambda x: nx.adjacency_matrix(x), graphs))
    # print("Loaded {} graphs ".format(len(graphs)))

    if features:
        # if args['method'] == 'dist':
        #     feats_path = current_path + "/Data/{}/data/eval_{}_dist_{}_feats.npy".format(args['dataset'], str(args['timesteps']), args['world_size'])
        # else:
        #     feats_path = current_path + "/Data/{}/data/eval_{}_feats.npy".format(args['dataset'], str(args['timesteps']))
        # feats_path = current_path + "/Data/{}/data/eval_{}_feats.npy".format(args['dataset'], str(args['timesteps']))

        # save as sparse matrix
        feats_path = current_path + "../dataset/{}/data/eval_{}_feats/".format(args['dataset'], str(args['timesteps']))
        print(feats_path)
        try:
            # feats = np.load(feats_path, allow_pickle=True)
            num_feats = 0
            feats = []
            for time in range(timesteps):
                path = feats_path+'no_{}.npz'.format(time)
                feat = sp.load_npz(path)
                # feat = torch.Tensor(feat)

                if time == 0:
                    feat_array = feat.toarray()
                    num_feats = feat_array.shape[1]
                feat_coo = feat.tocoo()

                values = feat_coo.data
                indices = np.vstack((feat_coo.row, feat_coo.col))

                i = torch.LongTensor(indices)
                v = torch.FloatTensor(values)
                shape = feat_coo.shape

                # feat_tensor_sp = torch.sparse.FloatTensor(i, v, torch.Size(shape))
                feat_tensor_de = torch.sparse.FloatTensor(i, v, torch.Size(shape)).to_dense()

                # feat_tensor_sp = torch.sparse.FloatTensor(torch.LongTensor([feat_coo.row.tolist(), feat_coo.col.tolist()]),
                #                  torch.LongTensor(feat_coo.data.astype(np.int32)))

                feats.append(feat_tensor_de)

            print("Loading node features!")
        except IOError:
            print("Generating and saving node features ....")
            # method 1: compute the max degree over all graphs
            # max_deg, _ = _count_max_deg(graphs, adj_matrices)
            # feats = _generate_one_hot_feats(graphs, adj_matrices, max_deg)
            # method 2:
            # feats = _generate_feats(adj_matrices, timesteps)
            # print('saved feats, ',feats)

            # method 3: generate features using in and out degree
            feats = _generate_degree_feats(graphs, adj_matrices)

            folder_in = os.path.exists(feats_path)
            if not folder_in:
                os.makedirs(feats_path)
            pbar = tqdm(feats)
            for id,feat in enumerate(pbar):
                # print('feature shape is ', feat.shape)
                # feats_sp.append(sp.csr_matrix(feat))
                feat_sp = sp.csr_matrix(feat)
                path = feats_path+'no_{}.npz'.format(id)
                sp.save_npz(path, feat_sp)
                pbar.set_description('Compress feature and save:')

            num_feats = feats[0].shape[1]
            # to tensor_sp
            feats_tensor_sp = []
            for feat in feats:
                feats_tensor_sp.append(torch.Tensor(feat).to_sparse())
            # np.save(feats_path, feats)
            feats = feats_tensor_sp
    #normlized adj
    adj_matrices = [_normalize_graph_gcn(adj) for adj in adj_matrices]

    # return (args, graphs, adj_matrices, feats, num_feats)
    print(feats[0].shape[1])
    return graphs, adj_matrices, feats, num_feats

def graph_concat(graphs):
    """
    Function to concatenate snapshots along the temporal dimension
    :param graphs: list of snapshots
    """
    G = nx.Graph()

    # add nodes and edges
    # for i in range(len(graphs)):
    for i in tqdm(range(len(graphs)), desc='Concatenating...', leave=False):
        snap_id = [i]
        node_idx = {node: {'orig_id': node} for node in list(graphs[i].nodes())}
        nx.set_node_attributes(graphs[i], snap_id, "snap_id")
        nx.set_node_attributes(graphs[i], node_idx)
        attr = {edge: {"type": 'str'} for edge in graphs[i].edges()}
        nx.set_edge_attributes(graphs[i], attr)
        # print('Snapshot {} has nodes {} and edges {}'.format(i, graphs[i].number_of_nodes(), graphs[i].number_of_edges()))
        G = nx.disjoint_union(G, graphs[i])
    # for node in list(G):
    for node in tqdm(list(G), desc='Write attributes...', leave=False):
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