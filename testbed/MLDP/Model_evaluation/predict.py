import torch
import numpy as np
import argparse
import torch.nn as nn
import os
import warnings
import pandas as pd
import random
import warnings

from data_loader import get_loader
from graph_loader import load_data_test, generate_graph, convert_graph
from mlp import MLP_Predictor
from tqdm import tqdm
from DySAT import test_dysat

warnings.simplefilter("ignore")

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
    parser.add_argument('--dataset', type=str, default='Epinion',
                        help='method for DGNN training')
    parser.add_argument('--encoder', type=str, nargs='?', default="str",
                    help='Which encoder needs to be predicted')


    args = vars(parser.parse_args())
    with open(args['json_path'],'r') as load_f:
        para = json.load(load_f)
    args.update(para)

    return args

def _save_log(x, name):
    df_loss=pd.DataFrame(data=x)
    df_loss.to_csv('./experiment_results/{}.csv'.format(name), header=False)

def _get_test_dynamic_graph(args):
    graphs_raw = load_data_test(args)
    # generate subgraphs
    pbar = tqdm(graphs_raw, leave=False)
    graph_test = []  # list of dynamic graphs
    pbar_1 = tqdm(range(20), leave=False)
    num_feat = 0
    for k in pbar_1:
        pbar_1.set_description('Process snapshots:')
        # randomly choose timesteps
        random_timesteps = random.randint(1, len(graphs_raw))
        pbar_2 = tqdm(range(random_timesteps), leave=False)
        # nodes_list_per_snapshot = []
        # tensor_per_snapshot = []
        full_nodes = []
        snapshots = []
        for t in pbar_2:
            graph = graphs_raw[t]
            total_num_nodes = graphs_raw[t].number_of_nodes()
            pbar_2.set_description('Randomly generate temporal data:')
            num_nodes = random.randint(int(total_num_nodes/3), total_num_nodes)
            nodes_list = random.sample(range(total_num_nodes), num_nodes)
            subgraph = graph.subgraph(nodes_list)
            sub_adj, sub_feats = generate_graph(args, subgraph)
            num_feat = sub_feats.size(1)
            graph_data = convert_graph(subgraph, sub_adj, sub_feats, args['data_str'])
            # print('convert graph!')
            snapshots.append(graph_data)
        graph_test.append(snapshots)
    return graph_test, num_feat

if __name__ == '__main__':
    args = _get_args()
    args['device'] = torch.device("cuda")
    device = args['device']

    model_str = MLP_Predictor(in_feature = 2)
    model_str.load_state_dict(torch.load('./model/str_{}.pt'.format(10)))
    model_str = model_str.to(device)

    model_tem = MLP_Predictor(in_feature = 2)
    model_tem.load_state_dict(torch.load('./model/tem_{}.pt'.format(10)))
    model_tem = model_tem.to(device)

    test_graphs, num_feat = _get_test_dynamic_graph(args)
    measured_model = test_dysat(args, num_feat)

    measured_time_costs = []
    predicted_time_costs = []
    measured_individual_costs = []
    predicted_individual_costs = []
    for test_graph in test_graphs:
        # test_graph = [graph.to(device) for graph in test_graph]
        # measurement
        total_list = []
        individual_list = [[] for i in range(len(test_graph) + 1)]
        for step in range(20):
            measured_time, individual_cost = measured_model(test_graph)
            total_list.append(measured_time)
            for i in range(len(individual_cost)):
                individual_list[i].append(individual_cost[i])
        measured_time_costs.append(np.mean(total_list[10:]))
        for i in range(len(individual_cost)):
            measured_individual_costs.append(np.mean(individual_list[i][10:]))

        model_str.eval()
        model_tem.eval()
        # prediction
        num_snapshots = len(test_graph)
        str_time = 0
        tem_time = 0
        full_nodes = []
        for snapshot in test_graph:
            num_vertices = snapshot.number_of_nodes()
            num_edges = snapshot.number_of_edges()
            input = torch.Tensor([float(num_vertices/10000), float(num_edges/10000)]).to(device)
            # print(input)
            cost = model_str(input)
            str_time += cost
            full_nodes.extend(snapshot.nodes().tolist())
            predicted_individual_costs.append(cost)
        full_nodes = list(set(full_nodes))
        tem_input = torch.Tensor([float(len(full_nodes)/10000), float(len(test_graph)/10)]).to(device)
        cost = model_tem(tem_input)
        tem_time += cost
        predicted_individual_costs.append(cost)
        predicted_time = str_time + tem_time
        predicted_time_costs.append(predicted_time.item())

    print('Measured time costs: {}; Predicted time costs: {}'.format(measured_time_costs, predicted_time_costs))
    _save_log(measured_time_costs, 'measured_test_{}_{}'.format(args['dataset'], args['timesteps']))
    _save_log(predicted_time_costs, 'predicted_test_{}_{}'.format(args['dataset'], args['timesteps']))

    # print(measured_individual_costs, predicted_individual_costs)


        

    

