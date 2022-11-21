from tkinter import E
import torch
import argparse
import time
import random
import numpy as np
import os
from tqdm import tqdm

from graph_loader import load_data_train, generate_graph, convert_graph
from layers_mb import GAT_Layer_train as StructuralAttentionLayer
from layers_mb import ATT_layer_train as TemporalAttentionLayer
from torch.utils.data import DataLoader,TensorDataset, random_split

current_path = os.getcwd()

# generate training data for predict structural computation costs
def generate_str_data(args, file):
    dataset_file = file + '{}_dataset.pt'.format(args['timesteps'])
    try:
        with open(dataset_file, 'rb') as f:
            dataset = torch.load(f)
            data_x_tensor = dataset['x']
            data_y_tensor = dataset['y']
        print('Load dataset!')
        # print(data_x_tensor, data_y_tensor)

    except IOError:
        # step 1: randomly generate subgraph for dataset
        graphs_list = load_data_train(args)
        graphs_data = []
        pbar = tqdm(graphs_list, leave=False)
        for id, graphs_raw in enumerate(pbar):
            pbar.set_description('Generate data points:')
            pbar_1 = tqdm(graphs_raw, leave=False)
            for i, graph in enumerate(pbar_1):
                pbar_1.set_description('Process snapshot:')
                # random subgraph
                total_num_nodes = graph.number_of_nodes()
                pbar_2 = tqdm(range(100), leave=False)
                for i, step in enumerate(pbar_2):
                    pbar_2.set_description('Randomly generate subgraphs:')
                    num_nodes = random.randint(int(total_num_nodes/3), total_num_nodes)
                    nodes_list = random.sample(range(total_num_nodes), num_nodes)
                    subgraph = graph.subgraph(nodes_list)
                    sub_adj, sub_feats = generate_graph(args, subgraph)
                    graph_data = convert_graph(subgraph, sub_adj, sub_feats, args['data_str'])
                    # print('convert graph!')
                    graphs_data.append(graph_data)
        data_x = [[float(graph.number_of_nodes()/10000), float(graph.number_of_edges()/10000)] for graph in graphs_data]
        data_x_tensor = torch.tensor(data_x, dtype=torch.float32)

        # step 2: process subgraph to generate labels
        labels = []
        input_dim = 2
        structural_encoder = StructuralAttentionLayer(args=args,
                                                    input_dim=input_dim,
                                                    output_dim=128,
                                                    n_heads=8,
                                                    attn_drop=0.1,
                                                    ffd_drop=0.1,
                                                    residual=args['residual'])
        for graph in graphs_data:
            costs = []
            for step in range(20):
                costs.append(structural_encoder(graph))
            computation_cost = np.mean(costs[5:])
            labels.append(computation_cost)
        data_y_tensor = torch.tensor(labels)

        torch.save({"x": data_x_tensor, "y": data_y_tensor}, dataset_file)

    # step 3: generate dataset and split it to different sets
    dataset = TensorDataset(data_x_tensor, data_y_tensor)
    dataset_size = data_y_tensor.size(0)
    train_size = int(0.8 * dataset_size)
    test_size = dataset_size - train_size
    train_dataset, test_dataset = random_split(
                                    dataset=dataset,
                                    lengths=[train_size, test_size],
                                    generator=torch.Generator().manual_seed(0)
    )

    # step 4: dataload
    train_loader = DataLoader(dataset=train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=1000, shuffle=False)

    return train_loader, test_loader

# generate training data for predict temporal computation costs
def generate_tem_data(args, file):
    # timesteps: maximum number of snapshots
    dataset_file = file + 'tem_{}_dataset.pt'.format(args['timesteps'])
    try:
        with open(dataset_file, 'rb') as f:
            dataset = torch.load(f)
            data_x_tensor = dataset['x']
            data_y_tensor = dataset['y']
        print('Load dataset!')
        # print(data_x_tensor, data_y_tensor)

    except IOError:
        # step 1: randomly generate subgraph for dataset
        graphs_list = load_data_train(args)
        timesteps_per_data_point = [] # predict input feat[1]
        vertices_per_data_point = []  # predict input feat[0]
        tempora_feat_input = []  # to measured temporal computation costs
        labels = []
        input_dim = 128
        temporal_encoder = TemporalAttentionLayer(args,
                                           method=0,
                                           input_dim=input_dim,
                                           n_heads=8,
                                           attn_drop= 0.5,
                                           residual= args['residual'],
                                           interval_ratio = 0)
        pbar = tqdm(graphs_list, leave=False)
        for id, graphs_raw in enumerate(pbar):
            pbar.set_description('Process dynamic graphs:')
            pbar_1 = tqdm(range(1000), leave=False)
            for k in pbar_1:
                pbar_1.set_description('Process snapshots:')
                # randomly choose timesteps
                random_timesteps = random.randint(1, len(graphs_raw))
                pbar_2 = tqdm(range(random_timesteps), leave=False)
                nodes_list_per_snapshot = []
                tensor_per_snapshot = []
                full_nodes = []
                for t in pbar_2:
                    total_num_nodes = graphs_raw[t].number_of_nodes()
                    pbar_2.set_description('Randomly generate temporal data:')
                    num_nodes = random.randint(int(total_num_nodes/3), total_num_nodes)
                    nodes_list = random.sample(range(total_num_nodes), num_nodes)
                    full_nodes.extend(nodes_list)
                    nodes_list_per_snapshot.append(nodes_list)
                    feat_tensor = torch.rand(num_nodes, 128)
                    tensor_per_snapshot.append(feat_tensor)
                # generate a data point
                full_nodes = list(set(full_nodes))
                timesteps_per_data_point.append(random_timesteps)
                vertices_per_data_point.append(len(full_nodes))
                tempora_feat_tensor = torch.zeros(len(full_nodes), random_timesteps, 128, dtype=torch.float32)  # [N,T,F]

                full_index = range(len(full_nodes))
                idx_mask_dict = dict(zip(full_nodes, full_index))
                for t in range(random_timesteps):
                    idx = [idx_mask_dict[node_idx] for node_idx in nodes_list_per_snapshot[t]]
                    tempora_feat_tensor[idx, t, :] = tensor_per_snapshot[t]
                # print(tempora_feat_tensor.size())
                costs = []
                for step in range(20):
                    costs.append(temporal_encoder(tempora_feat_tensor))
                computation_cost = np.mean(costs[5:])
                labels.append(computation_cost)
                # tempora_feat_input.append(tempora_feat_tensor)

        # print(timesteps_per_data_point)
        # print(tempora_feat_input[0].size())

        data_x = [[float(vertices_per_data_point[i]/10000), float(timesteps_per_data_point[i]/10)] for i in range(len(timesteps_per_data_point))]  # training data feature [num_nodes, num_timesteps]
        data_x_tensor = torch.tensor(data_x, dtype=torch.float32)
        data_y_tensor = torch.tensor(labels)

        # step 2: process subgraph to generate labels

        # for temporal_input in tempora_feat_input:
        #     costs = []
        #     for step in range(20):
        #         costs.append(temporal_encoder(temporal_input))
        #     computation_cost = np.mean(costs[5:])
        #     labels.append(computation_cost)
        # data_y_tensor = torch.tensor(labels)

        torch.save({"x": data_x_tensor, "y": data_y_tensor}, dataset_file)

    # step 3: generate dataset and split it to different sets
    dataset = TensorDataset(data_x_tensor, data_y_tensor)
    dataset_size = data_y_tensor.size(0)
    train_size = int(0.8 * dataset_size)
    test_size = dataset_size - train_size
    train_dataset, test_dataset = random_split(
                                    dataset=dataset,
                                    lengths=[train_size, test_size],
                                    generator=torch.Generator().manual_seed(0)
    )

    # step 4: dataload
    train_loader = DataLoader(dataset=train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=1000, shuffle=False)

    return train_loader, test_loader

def generate_all_data(args, train_flag):
    # timesteps: maximum number of snapshots
    file = current_path + "/Dataset/"
    dataset_str_file = file + 'str_{}_dataset.pt'.format(args['timesteps'])
    dataset_tem_file = file + 'tem_{}_dataset.pt'.format(args['timesteps'])
    if train_flag == 'str':
        dataset_file = dataset_str_file
    else:
        dataset_file = dataset_tem_file
    try:
        with open(dataset_file, 'rb') as f:
            dataset = torch.load(f)
            data_x_tensor = dataset['x']
            data_y_tensor = dataset['y']
        print('Load dataset!')
        # print(data_x_tensor, data_y_tensor)

    except IOError:
        dynamic_graphs_list = load_data_train(args)
        # generate subgraphs
        # define two encoders
        structural_encoder = StructuralAttentionLayer(args=args,
                                                        input_dim=2,
                                                        output_dim=128,
                                                        n_heads=8,
                                                        attn_drop=0.1,
                                                        ffd_drop=0.1,
                                                        residual=args['residual'])
        temporal_encoder = TemporalAttentionLayer(args,
                                            method=0,
                                            input_dim=128,
                                            n_heads=8,
                                            attn_drop= 0.5,
                                            residual= args['residual'],
                                            interval_ratio = 0)


        data_x_str = []
        data_y_str = []
        data_x_tem = []
        data_y_tem = []
        pbar = tqdm(dynamic_graphs_list, leave=False)
        for dynamic_graph in pbar:
            pbar.set_description('Processing dynamic graphs:')
            pbar_1 = tqdm(range(1000), leave=False)
            num_feat = 0
            for k in pbar_1:
                pbar_1.set_description('Generate data points:')
                # randomly choose timesteps
                random_timesteps = random.randint(1, len(dynamic_graph))
                pbar_2 = tqdm(range(random_timesteps), leave=False)
                nodes_list_per_snapshot = []
                tensor_per_snapshot = []
                full_nodes = []
                for t in pbar_2:
                    graph = dynamic_graph[t]
                    total_num_nodes = dynamic_graph[t].number_of_nodes()
                    pbar_2.set_description('Processing snapshots:')
                    num_nodes = random.randint(int(total_num_nodes/3), total_num_nodes)
                    nodes_list = random.sample(range(total_num_nodes), num_nodes)
                    subgraph = graph.subgraph(nodes_list)
                    sub_adj, sub_feats = generate_graph(args, subgraph)
                    graph_data = convert_graph(subgraph, sub_adj, sub_feats, args['data_str'])
                    # print('convert graph!')
                    # for str dataset feature x
                    data_x_str.append([float(graph_data.number_of_nodes()/10000), float(graph_data.number_of_edges()/10000)])
                    costs = []
                    for step in range(20):
                        costs.append(structural_encoder(graph_data))
                    computation_cost = np.mean(costs[5:])
                    data_y_str.append(computation_cost)

                    feat_tensor = torch.rand(num_nodes, 128)
                    tensor_per_snapshot.append(feat_tensor)
                    full_nodes.extend(nodes_list)
                    nodes_list_per_snapshot.append(nodes_list)

                # generate a data point
                full_nodes = list(set(full_nodes))
                # for tem dataset feature x
                data_x_tem.append([float(len(full_nodes)/10000), float(random_timesteps/10)])

                tempora_feat_tensor = torch.zeros(len(full_nodes), random_timesteps, 128, dtype=torch.float32)  # [N,T,F]
                full_index = range(len(full_nodes))
                idx_mask_dict = dict(zip(full_nodes, full_index))
                for t in range(random_timesteps):
                    idx = [idx_mask_dict[node_idx] for node_idx in nodes_list_per_snapshot[t]]
                    tempora_feat_tensor[idx, t, :] = tensor_per_snapshot[t]
                # print(tempora_feat_tensor.size())
                costs = []
                for step in range(20):
                    costs.append(temporal_encoder(tempora_feat_tensor))
                computation_cost = np.mean(costs[5:])
                data_y_tem.append(computation_cost)

        data_x_str_tensor = torch.tensor(data_x_str, dtype=torch.float32)
        data_y_str_tensor = torch.tensor(data_y_str)
        data_x_tem_tensor = torch.tensor(data_x_tem, dtype=torch.float32)
        data_y_tem_tensor = torch.tensor(data_y_tem)
        # save
        torch.save({"x": data_x_str_tensor, "y": data_y_str_tensor}, dataset_str_file)
        torch.save({"x": data_x_tem_tensor, "y": data_y_tem_tensor}, dataset_tem_file)
        if train_flag == 'str':
            data_x_tensor = data_x_str_tensor
            data_y_tensor = data_y_str_tensor
        else:
            data_x_tensor = data_x_tem_tensor
            data_y_tensor = data_y_tem_tensor
    
    # step 3: generate dataset and split it to different sets
    dataset = TensorDataset(data_x_tensor, data_y_tensor)
    dataset_size = data_y_tensor.size(0)
    train_size = int(0.8 * dataset_size)
    test_size = dataset_size - train_size
    train_dataset, test_dataset = random_split(
                                    dataset=dataset,
                                    lengths=[train_size, test_size],
                                    generator=torch.Generator().manual_seed(0)
    )

    # step 4: dataload
    train_loader = DataLoader(dataset=train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=1000, shuffle=False)

    return train_loader, test_loader


def _get_args():
    import json
    parser = argparse.ArgumentParser(description='Test parameters')
    parser.add_argument('--json-path', type=str, required=True,
                        help='the path of hyperparameter json file')
    # parser.add_argument('--test-type', type=str, required=True, choices=['local', 'dp', 'ddp'],
    #                     help='method for DGNN training')
    
    # for experimental configurations
    parser.add_argument('--timesteps', type=int, nargs='?', default=8,
                    help="total time steps used for train, eval and test")
    parser.add_argument('--epochs', type=int, nargs='?', default=100,
                    help="total number of epochs")
    parser.add_argument('--world_size', type=int, default=1,
                        help='method for DGNN training')
    parser.add_argument('--gate', type=bool, default=False,
                        help='method for DGNN training')
    parser.add_argument('--dataset', type=str, default='Epinion',
                        help='method for DGNN training')
    parser.add_argument('--partition', type=str, nargs='?', default="Time",
                    help='How to partition the graph data')
    parser.add_argument('--balance', type=bool, nargs='?', default=True,
                    help='balance workload')
    parser.add_argument('--N1', type=int, nargs='?', default=10000,
                    help='N1: total number of nodes')
    parser.add_argument('--N2', type=int, nargs='?', default=1000,
                    help='number of different nodes')

    args = vars(parser.parse_args())
    with open(args['json_path'],'r') as load_f:
        para = json.load(load_f)
    args.update(para)

    return args

def get_loader(args, flag):
    # graphs, load_adj, load_feats, num_feats = generate_graphs(args, graphs_raw)
    dataset_path = current_path + "/Dataset/"
    # if flag == 'str':
    #     train_loader, test_loader = generate_str_data(args, dataset_path)
    # else:
    #     train_loader, test_loader = generate_tem_data(args, dataset_path)

    train_loader, test_loader = generate_all_data(args, flag)
    return train_loader, test_loader

if __name__ == '__main__':
    args = _get_args()
    train_loader, test_loader = get_loader(args, 'tem')
    