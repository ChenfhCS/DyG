import numpy as np
import argparse
import torch
import networkx as nx
import math
import random
import time

import sys 
sys.path.append("..") 

from tqdm import tqdm
from data_process import load_data, generate_graphs, graph_concat
from method import MLP_Predictor, coarsener
from utils import RNN_comm_nodes_new, GCN_comm_nodes, Comm_time, Computation_time, generate_test_graph

bandwidth_1MB = float(1024*1024*8)
bandwidth_10MB = float(10*1024*1024*8)
bandwidth_100MB = float(100*1024*1024*8)
bandwidth_GB = float(1024*1024*1024*8)


class PTS():
    def __init__(self, args, graphs, nodes_list, adjs_list, num_devices):
        self.args = args
        self.nodes_list = nodes_list
        self.adjs_list = adjs_list
        self.num_devices = num_devices
        self.graphs = graphs
        self.timesteps = len(nodes_list)
        self.workloads_GCN = [[torch.full_like(self.nodes_list[time], False, dtype=torch.bool) for time in range(self.timesteps)] for i in range(num_devices)]
        self.workloads_ATT = [[torch.full_like(self.nodes_list[time], False, dtype=torch.bool) for time in range(self.timesteps)] for i in range(num_devices)]

        # runtime
        Q_id, Q_node_id, Q_workload = self.partition()

        self.schedule(Q_id, Q_node_id, Q_workload)
    
    def partition(self):
        '''
        Step 1: partition snapshot into P set; partition nodes into Q set
        '''
        Q_id = []
        Q_node_id = []
        Q_workload = []
        Degree = []
        for time in range(self.timesteps):
            if time == 0:
                    start = 0
            else:
                start = self.nodes_list[time - 1].size(0)
            end = self.nodes_list[time].size(0)
            workload = self.nodes_list[time][start:end]
            for node in workload.tolist():
                Q_id.append(time)
                # divided_nodes = self.nodes_list[time].size(0) - workload.size(0)
                Q_node_id.append(node)
                Q_workload.append(self.timesteps - time)
        return Q_id, Q_node_id, Q_workload
    
    def schedule(self, Q_id, Q_node_id, Q_workload):
        Current_RNN_workload = [0 for i in range(self.num_devices)]
        # compute the average workload
        RNN_avg_workload = np.sum(Q_workload)/self.num_devices

        for idx in range(len(Q_id)):  # schedule node-level job to ATT workload
            Load = []
            for m in range(self.num_devices):
                Load.append(1 - float((Current_RNN_workload[m] + Q_workload[idx])/RNN_avg_workload))
            select_m = Load.index(max(Load))
            # for m in range(self.num_devices):
            #     if m == select_m:
            for time in range(self.timesteps)[Q_id[idx]:]:
                # print(self.workloads_GCN[m][time])
                self.workloads_GCN[select_m][time][Q_node_id[idx]] = torch.ones(1, dtype=torch.bool)
                self.workloads_ATT[select_m][time][Q_node_id[idx]] = torch.ones(1, dtype=torch.bool)
                
                # Scheduled_workload[time][Q_node_id[idx]] = torch.ones(1, dtype=torch.bool)
            Current_RNN_workload[select_m] = Current_RNN_workload[select_m] + Q_workload[idx]

    def communication_time(self, GCN_node_size, RNN_node_size, bandwidth, GCN_comp_scale, ATT_comp_scale):
        '''
        Both GCN communication time and ATT communication time are needed
        '''
        # distribution = Distribution(self.nodes_list, self.timesteps, self.num_devices, self.workloads_GCN)
        # print('Node-partition distribution: ',distribution)

        RNN_receive_list, RNN_send_list = RNN_comm_nodes_new(self.nodes_list, self.num_devices, self.workloads_GCN, self.workloads_ATT)

        GCN_receive_list, GCN_send_list = GCN_comm_nodes(self.nodes_list, self.adjs_list, self.num_devices, self.workloads_GCN)
        # RNN_receive_list, RNN_send_list = RNN_comm_nodes(self.nodes_list, self.num_devices, self.workloads_GCN, self.workloads_ATT)

        GCN_receive_comm_time, GCN_send_comm_time = Comm_time(self.num_devices, GCN_receive_list, GCN_send_list, GCN_node_size, bandwidth)
        RNN_receive_comm_time, RNN_send_comm_time = Comm_time(self.num_devices, RNN_receive_list, RNN_send_list, RNN_node_size, bandwidth)
        # print(GCN_receive_list, GCN_send_list)
        GCN_receive = [torch.cat(GCN_receive_list[i], 0).size(0) for i in range(self.num_devices)]
        GCN_send = [torch.cat(GCN_send_list[i], 0).size(0) for i in range(self.num_devices)]
        RNN_receive = [torch.cat(RNN_receive_list[i], 0).size(0) for i in range(self.num_devices)]
        RNN_send = [torch.cat(RNN_send_list[i], 0).size(0) for i in range(self.num_devices)]

        GCN_comm_time = [max(GCN_receive_comm_time[i], GCN_send_comm_time[i]) for i in range(len(GCN_receive_comm_time))]
        RNN_comm_time = [max(RNN_receive_comm_time[i], RNN_send_comm_time[i]) for i in range(len(RNN_receive_comm_time))]
        GPU_total_time = [GCN_comm_time[i] + RNN_comm_time[i] for i in range(len(GCN_comm_time))]
        # Total_time = max(GPU_total_time)
        GCN_comp_time, ATT_comp_time = Computation_time(self.graphs, self.num_devices, len(self.nodes_list), self.workloads_GCN, self.workloads_ATT, GCN_comp_scale, ATT_comp_scale)
        total_comp_time = [GCN_comp_time[i] + ATT_comp_time[i] for i in range(len(GCN_comm_time))]

        print('----------------------------------------------------------')
        print('PTS method:')
        print('GCN | Each GPU receives nodes: {} | Each GPU sends nodes: {}'.format(GCN_receive, GCN_send))
        print('ATT | Each GPU receives nodes: {} | Each GPU sends nodes: {}'.format(RNN_receive, RNN_send))
        print('Each GPU with computation time: {} ( GCN: {} | ATT: {})'.format(total_comp_time, GCN_comp_time, ATT_comp_time))
        print('Each GPU with communication time: {} ( GCN: {} | ATT: {})'.format(GPU_total_time, GCN_comm_time, RNN_comm_time))
        print('Total time: {} | Computation time: {}, Communication time: {}'.format(max(GPU_total_time) + max(GCN_comp_time) + max(ATT_comp_time), max(GCN_comp_time) + max(ATT_comp_time), max(GPU_total_time)))
    

class PSS():
    def __init__(self, args, graphs, nodes_list, adjs_list, num_devices):
        self.args = args
        self.nodes_list = nodes_list
        self.adjs_list = adjs_list
        self.num_devices = num_devices
        self.graphs = graphs
        self.timesteps = len(nodes_list)
        self.workloads_GCN = [[torch.full_like(self.nodes_list[time], False, dtype=torch.bool) for time in range(self.timesteps)] for i in range(num_devices)]
        self.workloads_ATT = [[torch.full_like(self.nodes_list[time], False, dtype=torch.bool) for time in range(self.timesteps)] for i in range(num_devices)]

        # runtime
        P_id, P_workload, P_snapshot = self.partition()

        self.schedule(P_id, P_workload, P_snapshot)
    
    def partition(self):
        '''
        Step 1: partition snapshot into P set; partition nodes into Q set
        '''
        P_id = [] # save the snapshot id
        P_workload = [] # save the workload size
        P_snapshot = []
        for time in range(self.timesteps):
            workload_gcn = self.nodes_list[time]
            P_id.append(time)
            P_workload.append(workload_gcn.size(0))
            P_snapshot.append(workload_gcn)
        return P_id, P_workload, P_snapshot
    
    def schedule(self, P_id, P_workload, P_snapshot):
        Current_GCN_workload = [0 for i in range(self.num_devices)]
        # compute the average workload
        GCN_avg_workload = np.sum(P_workload)/self.num_devices

        for idx in range(len(P_id)): # schedule snapshot-level job to GCN workload
            Load = []
            for m in range(self.num_devices):
                Load.append(1 - float((Current_GCN_workload[m]+P_workload[idx])/GCN_avg_workload))
                # Cross_edge.append(Current_RNN_workload[m][P_id[idx]])
            select_m = Load.index(max(Load))
            workload = torch.full_like(P_snapshot[idx], True, dtype=torch.bool)
            self.workloads_GCN[select_m][P_id[idx]][P_snapshot[idx]] = workload
            self.workloads_ATT[select_m][P_id[idx]][P_snapshot[idx]] = workload
            Current_GCN_workload[select_m] = Current_GCN_workload[select_m]+P_workload[idx]

    def communication_time(self, GCN_node_size, RNN_node_size, bandwidth, GCN_comp_scale, ATT_comp_scale):
        '''
        Both GCN communication time and ATT communication time are needed
        '''
        # distribution = Distribution(self.nodes_list, self.timesteps, self.num_devices, self.workloads_GCN)
        # print('snapshot-partition distribution: ',distribution)

        RNN_receive_list, RNN_send_list = RNN_comm_nodes_new(self.nodes_list, self.num_devices, self.workloads_GCN, self.workloads_ATT)

        GCN_receive_list, GCN_send_list = GCN_comm_nodes(self.nodes_list, self.adjs_list, self.num_devices, self.workloads_GCN)
        # RNN_receive_list, RNN_send_list = RNN_comm_nodes(self.nodes_list, self.num_devices, self.workloads_GCN, self.workloads_ATT)

        GCN_receive_comm_time, GCN_send_comm_time = Comm_time(self.num_devices, GCN_receive_list, GCN_send_list, GCN_node_size, bandwidth)
        RNN_receive_comm_time, RNN_send_comm_time = Comm_time(self.num_devices, RNN_receive_list, RNN_send_list, RNN_node_size, bandwidth)

        GCN_receive = [torch.cat(GCN_receive_list[i], 0).size(0) for i in range(self.num_devices)]
        GCN_send = [torch.cat(GCN_send_list[i], 0).size(0) for i in range(self.num_devices)]
        RNN_receive = [torch.cat(RNN_receive_list[i], 0).size(0) for i in range(self.num_devices)]
        RNN_send = [torch.cat(RNN_send_list[i], 0).size(0) for i in range(self.num_devices)]

        GCN_comm_time = [max(GCN_receive_comm_time[i], GCN_send_comm_time[i]) for i in range(len(GCN_receive_comm_time))]
        RNN_comm_time = [max(RNN_receive_comm_time[i], RNN_send_comm_time[i])*2 for i in range(len(RNN_receive_comm_time))]
        GPU_total_time = [GCN_comm_time[i] + RNN_comm_time[i] for i in range(len(GCN_comm_time))]
        # Total_time = max(GPU_total_time)
        GCN_comp_time, ATT_comp_time = Computation_time(self.graphs, self.num_devices, len(self.nodes_list), self.workloads_GCN, self.workloads_ATT, GCN_comp_scale, ATT_comp_scale)
        total_comp_time = [GCN_comp_time[i] + ATT_comp_time[i] for i in range(len(GCN_comp_time))]

        print('----------------------------------------------------------')
        print('PSS method:')
        print('GCN | Each GPU receives nodes: {} | Each GPU sends nodes: {}'.format(GCN_receive, GCN_send))
        print('ATT | Each GPU receives nodes: {} | Each GPU sends nodes: {}'.format(RNN_receive, RNN_send))
        print('Each GPU with computation time: {} ( GCN: {} | ATT: {})'.format(total_comp_time, GCN_comp_time, ATT_comp_time))
        print('Each GPU with communication time: {} ( GCN: {} | ATT: {})'.format(GPU_total_time, GCN_comm_time, RNN_comm_time))
        print('Total time: {} | Computation time: {}, Communication time: {}'.format(max(GPU_total_time) + max(GCN_comp_time) + max(ATT_comp_time), 
                                                                                    max(GCN_comp_time) + max(ATT_comp_time), max(GPU_total_time)))
        

class PSS_TS():
    def __init__(self, args, graphs, nodes_list, adjs_list, num_devices):
        self.args = args
        self.nodes_list = nodes_list
        self.adjs_list = adjs_list
        self.num_devices = num_devices
        self.graphs = graphs
        self.timesteps = len(nodes_list)
        self.workloads_GCN = [[torch.full_like(self.nodes_list[time], False, dtype=torch.bool) for time in range(self.timesteps)] for i in range(num_devices)]
        self.workloads_ATT = [[torch.full_like(self.nodes_list[time], False, dtype=torch.bool) for time in range(self.timesteps)] for i in range(num_devices)]

        # runtime
        P_id, Q_id, Q_node_id, P_workload, P_snapshot, Q_workload = self.partition()

        self.schedule(P_id, Q_id, Q_node_id, P_workload, P_snapshot, Q_workload)
    
    def partition(self):
        '''
        Step 1: partition snapshot into P set; partition nodes into Q set
        '''
        P_id = [] # save the snapshot id
        Q_id = []
        Q_node_id = []
        P_workload = [] # save the workload size
        P_snapshot = []
        Q_workload = []
        for time in range(self.timesteps):
            if time == 0:
                start = 0
            else:
                start = self.nodes_list[time - 1].size(0)
            end = self.nodes_list[time].size(0)

            workload_gcn = self.nodes_list[time]
            P_id.append(time)
            P_workload.append(workload_gcn.size(0))
            P_snapshot.append(workload_gcn)

            workload_rnn = self.nodes_list[time][start:end]
            for node in workload_rnn.tolist():
                Q_id.append(time)
                # divided_nodes = self.nodes_list[time].size(0) - workload.size(0)
                Q_node_id.append(node)
                Q_workload.append(self.timesteps - time)
        return P_id, Q_id, Q_node_id, P_workload, P_snapshot, Q_workload
    
    def schedule(self, P_id, Q_id, Q_node_id, P_workload, P_snapshot, Q_workload):
        Scheduled_workload = [torch.full_like(self.nodes_list[time], False, dtype=torch.bool) for time in range(self.timesteps)]
        Current_GCN_workload = [0 for i in range(self.num_devices)]
        Current_RNN_workload = [0 for i in range(self.num_devices)]
        # compute the average workload
        GCN_avg_workload = np.sum(P_workload)/self.num_devices
        RNN_avg_workload = np.sum(Q_workload)/self.num_devices

        for idx in range(len(P_id)): # schedule snapshot-level job to GCN workload
            Load = []
            Cross_edge = []
            for m in range(self.num_devices):
                Load.append(1 - float((Current_GCN_workload[m]+P_workload[idx])/GCN_avg_workload))
                # Cross_edge.append(Current_RNN_workload[m][P_id[idx]])
            select_m = Load.index(max(Load))
            workload = torch.full_like(P_snapshot[idx], True, dtype=torch.bool)
            self.workloads_GCN[select_m][P_id[idx]][P_snapshot[idx]] = workload

            Current_GCN_workload[select_m] = Current_GCN_workload[select_m]+P_workload[idx]
            # Current_RNN_workload[select_m][P_id[idx]] += 1

        # print('GCN workload after scheduling snapshot-level jobs: ', self.workloads_GCN)

        for idx in range(len(Q_id)):  # schedule node-level job to ATT workload
            Load = []
            for m in range(self.num_devices):
                Load.append(1 - float((Current_RNN_workload[m] + Q_workload[idx])/RNN_avg_workload))
            select_m = Load.index(max(Load))
            # for m in range(self.num_devices):
            #     if m == select_m:
            for time in range(self.timesteps)[Q_id[idx]:]:
                # print(self.workloads_GCN[m][time])
                self.workloads_ATT[select_m][time][Q_node_id[idx]] = torch.ones(1, dtype=torch.bool)
                
                # Scheduled_workload[time][Q_node_id[idx]] = torch.ones(1, dtype=torch.bool)
            Current_RNN_workload[select_m] = Current_RNN_workload[select_m] + Q_workload[idx]
        # print('GCN workload after scheduling timeseries-level jobs: ', self.workloads_GCN)

    def communication_time(self, GCN_node_size, RNN_node_size, bandwidth, GCN_comp_scale, ATT_comp_scale):
        '''
        Both GCN communication time and ATT communication time are needed
        '''
        # distribution = Distribution(self.nodes_list, self.timesteps, self.num_devices, self.workloads_GCN)
        # print('snapshot-partition distribution: ',distribution)

        RNN_receive_list, RNN_send_list = RNN_comm_nodes_new(self.nodes_list, self.num_devices, self.workloads_GCN, self.workloads_ATT)

        GCN_receive_list, GCN_send_list = GCN_comm_nodes(self.nodes_list, self.adjs_list, self.num_devices, self.workloads_GCN)
        # RNN_receive_list, RNN_send_list = RNN_comm_nodes(self.nodes_list, self.num_devices, self.workloads_GCN, self.workloads_ATT)

        GCN_receive_comm_time, GCN_send_comm_time = Comm_time(self.num_devices, GCN_receive_list, GCN_send_list, GCN_node_size, bandwidth)
        RNN_receive_comm_time, RNN_send_comm_time = Comm_time(self.num_devices, RNN_receive_list, RNN_send_list, RNN_node_size, bandwidth)

        GCN_receive = [torch.cat(GCN_receive_list[i], 0).size(0) for i in range(self.num_devices)]
        GCN_send = [torch.cat(GCN_send_list[i], 0).size(0) for i in range(self.num_devices)]
        RNN_receive = [torch.cat(RNN_receive_list[i], 0).size(0) for i in range(self.num_devices)]
        RNN_send = [torch.cat(RNN_send_list[i], 0).size(0) for i in range(self.num_devices)]

        GCN_comm_time = [max(GCN_receive_comm_time[i], GCN_send_comm_time[i]) for i in range(len(GCN_receive_comm_time))]
        RNN_comm_time = [max(RNN_receive_comm_time[i], RNN_send_comm_time[i])*2 for i in range(len(RNN_receive_comm_time))]
        GPU_total_time = [GCN_comm_time[i] + RNN_comm_time[i] for i in range(len(GCN_comm_time))]
        # Total_time = max(GPU_total_time)
        GCN_comp_time, ATT_comp_time = Computation_time(self.graphs, self.num_devices, len(self.nodes_list), self.workloads_GCN, self.workloads_ATT, GCN_comp_scale, ATT_comp_scale)
        total_comp_time = [GCN_comp_time[i] + ATT_comp_time[i] for i in range(len(GCN_comp_time))]

        print('----------------------------------------------------------')
        print('PSS_TS method:')
        print('GCN | Each GPU receives nodes: {} | Each GPU sends nodes: {}'.format(GCN_receive, GCN_send))
        print('ATT | Each GPU receives nodes: {} | Each GPU sends nodes: {}'.format(RNN_receive, RNN_send))
        print('Each GPU with computation time: {} ( GCN: {} | ATT: {})'.format(total_comp_time, GCN_comp_time, ATT_comp_time))
        print('Each GPU with communication time: {} ( GCN: {} | ATT: {})'.format(GPU_total_time, GCN_comm_time, RNN_comm_time))
        print('Total time: {} | Computation time: {}, Communication time: {}'.format(max(GPU_total_time) + max(GCN_comp_time) + max(ATT_comp_time), 
                                                                                    max(GCN_comp_time) + max(ATT_comp_time), max(GPU_total_time)))


class Diana():
    def __init__(self, args, graphs, nodes_list, adjs_list, num_devices, gcn_node_size, ATT_node_size, bandwidth, GCN_comp_scale, ATT_comp_scale):
        super(Diana, self).__init__()
        self.args = args
        self.nodes_list = nodes_list
        self.adjs_list = adjs_list
        self.num_devices = num_devices
        self.graphs = graphs
        self.timesteps = len(nodes_list)
        # self.workloads_GCN = [[torch.full_like(self.nodes_list[time], False, dtype=torch.bool) for time in range(self.timesteps)] for i in range(num_devices)]
        # self.workloads_ATT = [[torch.full_like(self.nodes_list[time], False, dtype=torch.bool) for time in range(self.timesteps)] for i in range(num_devices)]
        self.workloads_GCN = [[torch.full_like(self.nodes_list[-1], False, dtype=torch.bool) for time in range(self.timesteps)] for i in range(num_devices)]
        self.workloads_ATT = [[torch.full_like(self.nodes_list[-1], False, dtype=torch.bool) for time in range(self.timesteps)] for i in range(num_devices)]

        self.Degrees = [list(dict(nx.degree(self.graphs[t])).values()) for t in range(self.timesteps)]
        self.gcn_node_size = gcn_node_size
        self.ATT_node_size = ATT_node_size
        self.bandwidth = bandwidth

        self.total_nodes = np.sum([len(nodes) for nodes in self.nodes_list])

        # hyper-parameter
        self.k1 = 0.001
        self.k2 = 0.001
        self.alpha = 0.01
        self.beta = 0.01

        self.GCN_comp_scale = GCN_comp_scale
        self.ATT_comp_scale = ATT_comp_scale

        self.device = torch.device("cuda")
        self.model_str = MLP_Predictor(in_feature = 2)
        self.model_str.load_state_dict(torch.load('./Model_evaluation/model/str_{}.pt'.format(10)))
        self.model_str = self.model_str.to(self.device)

        self.model_tem = MLP_Predictor(in_feature = 2)
        self.model_tem.load_state_dict(torch.load('./Model_evaluation/model/tem_{}.pt'.format(10)))
        self.model_tem = self.model_tem.to(self.device)

        self.model_str.eval()
        self.model_tem.eval()
        self.coarsening()
        # self.cluster_id_list, self.cluster_time_list = self.clustering()
      
        
        # print(self.workloads_GCN)
        gcn_cost = 0
        att_cost = 0
        full_nodes = []
        total_time_step = 0
        for t in range(self.timesteps):
            graph = self.graphs[t]
            nodes_list = self.nodes_list[t]
            num_vertices = graph.number_of_nodes()
            num_edges = graph.number_of_edges()
            input = torch.Tensor([float(num_vertices/10000), float(num_edges/10000)]).to(self.device)
            cost = self.model_str(input)
            gcn_cost += cost.item()
            if len(nodes_list) > 0:
                full_nodes.extend(nodes_list)
                total_time_step += 1
        full_nodes = list(set(full_nodes))
        tem_input = torch.Tensor([float(len(full_nodes)/10000), float(total_time_step/10)]).to(self.device)
        cost = self.model_tem(tem_input)
        att_cost += cost.item()
        self.total_computation_cost = gcn_cost + att_cost

    def coarsening(self):
        """
        coarsen the original dynamic graph to a single coarsened graph
        :return self.coarsened_graph: single coarsened graph
        :return self.node_to_nodes_list: mask each node in the coarsened graph to a node lists in the original dynamic graph
        """
        # self.full_graph = graph_concat(self.graphs)
        # self.coarsened_graph, self.node_to_nodes_list = coarsener(self.args, self.graphs, self.full_graph)

        # partition visiablity test without coarsening: coarsening step = 1000
        self.full_graph = graph_concat(self.graphs)
        self.coarsened_graph, self.node_to_nodes_list = coarsener(self.args, self.graphs, self.full_graph)
    
    def partitioning(self, alg):
        """
        partition the coarsened graph
        """
        self.alg = alg
        num_nodes_process = 0
        # for node in self.coarsened_graph.nodes():
        for node in tqdm(self.coarsened_graph.nodes(), desc='Partitioning...:', leave=True):
            nodes_to_partition = self.node_to_nodes_list[node]
            # calculate the inter-device communication costs
            scores = []
            for m in range(self.num_devices):
                GCN_communication = 0
                ATT_communication = 0
                for t in range(self.timesteps):
                    nodes = nodes_to_partition[t]
                    adj = self.adjs_list[t].clone()
                    edge_source = adj._indices()[0]
                    edge_target = adj._indices()[1]
                    nodes_mask = torch.zeros(self.nodes_list[t].size(0), dtype=torch.bool)
                    nodes_mask[nodes] = torch.ones(len(nodes), dtype=torch.bool)
                    has_edge_mask = nodes_mask[edge_source]
                    workload_edge_idx = torch.nonzero(has_edge_mask == True, as_tuple=False).view(-1)  # edges with source as 'nodes'
                    edge_target = edge_target[workload_edge_idx]                                        # target nodes with source as 'nodes'
                    cross_edges = torch.nonzero(self.workloads_GCN[m][t][edge_target] == True).view(-1)  # how many target nodes in device m
                    GCN_communication += cross_edges.size(0)

                temporal_node = torch.cat([self.workloads_GCN[m][t][nodes_to_partition[t]] for t in range(self.timesteps)], dim = 0)
                # workload_node = torch.cat(self.workloads_GCN[m][:][workload], dim = 0)
                local_temporal_node = torch.nonzero(temporal_node == True, as_tuple=False).view(-1)
                ATT_communication = ATT_communication + local_temporal_node.size(0)

                if alg == 'LDG_base':
                    compute_nodes = torch.nonzero(torch.cat(self.workloads_GCN[m], dim=0) == True, as_tuple=False).view(-1)
                    num_node = compute_nodes.size(0)
                    inter_edges = GCN_communication + ATT_communication
                    score = inter_edges*(1-num_node/self.total_nodes)
                    # score = inter_edges
                elif alg == 'LDG_DyG':
                    gcn_cost = 0
                    att_cost = 0
                    full_nodes = []
                    total_time_step = 0
                    for t in range(self.timesteps):
                        nodes_list = torch.nonzero(self.workloads_GCN[m][t] == True, as_tuple=False).view(-1).tolist()
                        if len(nodes_list) > 0:
                            graph = self.graphs[t].subgraph(nodes_list)
                            num_vertices = graph.number_of_nodes()
                            num_edges = graph.number_of_edges()
                            input = torch.Tensor([float(num_vertices/10000), float(num_edges/10000)]).to(self.device)
                            cost = self.model_str(input)
                            gcn_cost += cost.item()
                            full_nodes.extend(nodes_list)
                            total_time_step += 1
                    full_nodes = list(set(full_nodes))
                    tem_input = torch.Tensor([float(len(full_nodes)/10000), float(total_time_step/10)]).to(self.device)
                    cost = self.model_tem(tem_input)
                    att_cost += cost.item()

                    # total_comp_cost = gcn_cost+att_cost
                    # total_comp_cost = GCN_workload*self.GCN_comp_scale + ATT_workload*self.ATT_comp_scale
                    inter_communication = ((GCN_communication*self.gcn_node_size + ATT_communication*self.ATT_node_size)/self.bandwidth)
                    # inter_communication = GCN_communication + ATT_communication
                    # score = inter_communication*(1-(gcn_cost+att_cost)/self.total_computation_cost)
                    score = 10*inter_communication+(1-(gcn_cost+att_cost)/self.total_computation_cost)
                elif alg == 'Fennel_base':
                    alpha = 3
                    beta = 1.5
                    compute_nodes = torch.nonzero(torch.cat(self.workloads_GCN[m], dim=0) == True, as_tuple=False).view(-1)
                    num_node = compute_nodes.size(0)
                    inter_edges = GCN_communication + ATT_communication
                    score = inter_edges - alpha*beta*pow(num_node, (beta - 1))
                # elif alg == 'Fennel_DyG':
                #     alpha = 3
                #     beta = 1.5
                #     total_comp_cost = GCN_workload*self.GCN_comp_scale + ATT_workload*self.ATT_comp_scale
                #     inter_communication = (GCN_communication*self.gcn_node_size + ATT_communication*self.ATT_node_size)/self.bandwidth
                #     score = 10*inter_communication - alpha*beta*pow(total_comp_cost, (beta - 1))

                scores.append(score)
            select_m = scores.index(max(scores))
            for t in range(self.timesteps):
                nodes = nodes_to_partition[t]
                if len(nodes) > 0:
                    num_nodes_process += 1
                    self.workloads_GCN[select_m][t][nodes] = torch.ones(len(nodes), dtype=torch.bool)
                    self.workloads_ATT[select_m][t][nodes] = torch.ones(len(nodes), dtype=torch.bool)
                    # assigin attribute
                    attr = {node: {'partition': select_m}}
                    nx.set_node_attributes(self.coarsened_graph, attr)
        # print(num_nodes_process)
        # plot
        # step 1: remove temporal links
        temporal_edges = []
        for edge in self.coarsened_graph.edges():
            if self.coarsened_graph.edges[edge]["type"] == 'tem':
                temporal_edges.append(edge)
        self.coarsened_graph.remove_edges_from(temporal_edges)

        # step 2: define node position
        pos_list = [(random.uniform(0, 2), random.uniform(0, 2)) for _ in self.coarsened_graph.nodes()]
        # for node in coarsened_graph.nodes():
        # original_nodes_pos = {node: pos_list[node_mask[node].item()] for node in full_graph.nodes()}
        original_nodes_pos = {}
        for node in self.coarsened_graph.nodes():
            pos_tem = list(pos_list[self.coarsened_graph.nodes[node]['orig_id']])
            pos_tem[0] += (self.coarsened_graph.nodes[node]['snap_id'][0]*3)
            original_nodes_pos[node] = tuple(pos_tem)

        # step 3: degine node colors
        colorArr = ['1','2','3','4','5','6','7','8','9','A','B','C','D','E','F']
        color_list = []
        print([self.coarsened_graph.nodes[node]['partition'] for node in self.coarsened_graph.nodes()])
        for num_color in range(self.num_devices):
            color = ""
            for i in range(6):
                color += colorArr[random.randint(0,14)]
            color_list.append("#"+color)
        node_color_mask = {node: self.coarsened_graph.nodes[node]['partition'] for node in self.coarsened_graph.nodes()}
        node_color = [color_list[node_color_mask[node]] for node in self.coarsened_graph.nodes()]

        # plot_graph(args, self.coarsened_graph, node_color, 'black', original_nodes_pos, 'partition')
    def communication_time(self, GCN_node_size, ATT_node_size, bandwidth, GCN_comp_scale, ATT_comp_scale):
        '''
        Both GCN communication time and ATT communication time are needed
        '''

        # distribution = Distribution(self.nodes_list, self.timesteps, self.num_devices, self.workloads_GCN)
        # print('ours partition distribution: ',distribution)
        
        start = time.time()
        ATT_receive_list, ATT_send_list = RNN_comm_nodes_new(self.nodes_list, self.num_devices, self.workloads_GCN, self.workloads_ATT)
        ATT_get_node_time = time.time() - start

        start = time.time()
        GCN_receive_list, GCN_send_list = GCN_comm_nodes(self.nodes_list, self.adjs_list, self.num_devices, self.workloads_GCN)
        GCN_get_node_time = time.time() - start
        # ATT_receive_list, ATT_send_list = ATT_comm_nodes(self.nodes_list, self.num_devices, self.workloads_GCN, self.workloads_ATT)

        start = time.time()
        GCN_receive_comm_time, GCN_send_comm_time = Comm_time(self.num_devices, GCN_receive_list, GCN_send_list, GCN_node_size, bandwidth)
        ATT_receive_comm_time, ATT_send_comm_time = Comm_time(self.num_devices, ATT_receive_list, ATT_send_list, ATT_node_size, bandwidth)
        Communication_time = time.time() - start

        start = time.time()
        GCN_receive = [torch.cat(GCN_receive_list[i], 0).size(0) for i in range(self.num_devices)]
        GCN_send = [torch.cat(GCN_send_list[i], 0).size(0) for i in range(self.num_devices)]
        ATT_receive = [torch.cat(ATT_receive_list[i], 0).size(0) for i in range(self.num_devices)]
        ATT_send = [torch.cat(ATT_send_list[i], 0).size(0) for i in range(self.num_devices)]
        Sperate_time = time.time() - start
        
        GCN_comm_time = [max(GCN_receive_comm_time[i], GCN_send_comm_time[i]) for i in range(len(GCN_receive_comm_time))]
        ATT_comm_time = [max(ATT_receive_comm_time[i], ATT_send_comm_time[i]) for i in range(len(ATT_receive_comm_time))]
        GPU_total_time = [GCN_comm_time[i] + ATT_comm_time[i] for i in range(len(GCN_comm_time))]
        # Total_time = max(GPU_total_time)

        start = time.time()
        GCN_comp_time, ATT_comp_time = Computation_time(self.graphs, self.num_devices, len(self.nodes_list), self.workloads_GCN, self.workloads_ATT, GCN_comp_scale, ATT_comp_scale)
        Comp_time = time.time() - start
        total_comp_time = [GCN_comp_time[i] + ATT_comp_time[i] for i in range(len(GCN_comm_time))]
        # print('ATT get node cost: {:.3f}, GNN get node cost: {:.3f}, communication cost: {:.3f}, sperate cost: {:.3f}, computation cost: {:.3f}'.format(ATT_get_node_time,
        #                                                                                                 GCN_get_node_time,
        #                                                                                                 Communication_time,
        #                                                                                                 Sperate_time,
        #                                                                                                 Comp_time))

        print('----------------------------------------------------------')
        print('Diana + {}:'.format(self.alg))
        print('Original graph size V: {}, E: {} -> coarsened graph size V: {}, E: {}'.format(self.full_graph.number_of_nodes(), self.full_graph.number_of_edges(),
                                                                                                self.coarsened_graph.number_of_nodes(), self.coarsened_graph.number_of_edges()))
        # print('Number of clusters: ', len(self.cluster_id_list))
        print('GCN | Each GPU receives nodes: {} | Each GPU sends nodes: {}'.format(GCN_receive, GCN_send))
        print('ATT | Each GPU receives nodes: {} | Each GPU sends nodes: {}'.format(ATT_receive, ATT_send))
        print('Each GPU with computation time: {} ( GCN: {} | ATT: {})'.format(total_comp_time, GCN_comp_time, ATT_comp_time))
        print('Each GPU with communication time: {} ( GCN: {} | ATT: {})'.format(GPU_total_time, GCN_comm_time, ATT_comm_time))
        print('Total time: {} | Computation time: {}, Communication time: {}'.format(max(GPU_total_time) + max(GCN_comp_time) + max(ATT_comp_time), max(GCN_comp_time) + max(ATT_comp_time), max(GPU_total_time)))
        print('Total costs: {} | Computation costs: {}, Communication costs: {}'.format(np.sum(GPU_total_time) + np.sum(GCN_comp_time) + np.sum(ATT_comp_time), np.sum(GCN_comp_time) + np.sum(ATT_comp_time), np.sum(GPU_total_time)))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test parameters')
    # for experimental configurations
    parser.add_argument('--featureless', type=bool, default= True,
                        help='generate feature with one-hot encoding')
    parser.add_argument('--timesteps', type=int, nargs='?', default=8,
                    help="total time steps used for train, eval and test")
    parser.add_argument('--world_size', type=int, default=2,
                        help='method for DGNN training')
    parser.add_argument('--dataset', type=str, default='Epinion',
                        help='method for DGNN training')
    parser.add_argument('--real', type=str, nargs='?', default='True',
                    help='Whether use the real graph')
    args = vars(parser.parse_args())

    # print(args['real'])
    if args['real'] == 'False':
        # validation
        nodes_list, adjs_list = generate_test_graph()
        graphs = [nx.complete_graph(nodes_list[i].size(0)) for i in range(len(nodes_list))]
        time_steps = len(graphs)
        GCN_node_size = 25600
        RNN_node_size = 12800
        Degrees = [list(dict(nx.degree(graphs[t])).values()) for t in range(time_steps)]
        print('Number of graphs: ', len(graphs))
        print('Number of features: ', GCN_node_size)
        print('Average degrees: ', [np.mean(Degrees[t]) for t in range(time_steps)])

    else:
        raw_graphs = load_data(args)
        graphs = raw_graphs
        _, raw_adj, raw_feats, num_feats = generate_graphs(args, graphs)
        total_graphs = len(graphs)
        # print('Generate graphs!')
        start = len(graphs) - args['timesteps']
        # print(len(graph), args['time_steps'], start)

        graphs = graphs[start:]

        Num_nodes = args['nodes_info']
        Num_edges = args['edges_info']
        time_steps = len(graphs)

        nodes_list = [torch.tensor([j for j in range(Num_nodes[i])]) for i in range(time_steps)]
        # nodes_list = [torch.tensor([j for j in range(Num_nodes[i])]) for i in time_idx]
        # print('Generate nodes list!')

        adjs_list = []
        for i in range(time_steps):
            # print(type(adj_matrices[i]))
            adj_coo = raw_adj[i].tocoo()
            values = adj_coo.data
            indices = np.vstack((adj_coo.row, adj_coo.col))

            i = torch.LongTensor(indices)
            v = torch.FloatTensor(values)
            shape = adj_coo.shape

            adj_tensor_sp = torch.sparse_coo_tensor(i, v, torch.Size(shape))
            adjs_list.append(adj_tensor_sp)
        Degrees = [list(dict(nx.degree(graphs[t])).values()) for t in range(time_steps)]
        # Degrees = [list(dict(nx.degree(graphs[t])).values()) for t in time_idx]
        print('Number of total graphs ', total_graphs)
        print('Number of used graphs: ', len(graphs))
        print('Number of nodes: ', nodes_list[-1].size(0))
        print('Number of features: ', raw_feats[0].size(1))
        print('Node distribution: ', Num_nodes)
        print('Edge distribution: ', Num_edges)
        print('Average degrees: ', [np.mean(Degrees[t]) for t in range(time_steps)])

        # GCN_node_size = raw_feats[0].size(1)*32*
        GCN_node_size = 128*32
        RNN_node_size = 128*32
    
    GCN_comp_scale = 4*math.pow(10, -5)
    ATT_comp_scale = 4*math.pow(10, -5)

    # start = 15
    # window = 5
    # nodes_list = nodes_list[start: start+window]
    # adjs_list = adjs_list[start: start+window]

    # # # balance methods

    PTS_obj = PTS(args, graphs, nodes_list, adjs_list, args['world_size'])
    PTS_obj.communication_time(GCN_node_size, RNN_node_size, bandwidth_100MB, GCN_comp_scale, ATT_comp_scale)

    PSS_obj = PSS(args, graphs, nodes_list, adjs_list, args['world_size'])
    PSS_obj.communication_time(GCN_node_size, RNN_node_size, bandwidth_100MB, GCN_comp_scale, ATT_comp_scale)

    PSS_TS_obj = PSS_TS(args, graphs, nodes_list, adjs_list, args['world_size'])
    PSS_TS_obj.communication_time(GCN_node_size, RNN_node_size, bandwidth_100MB, GCN_comp_scale, ATT_comp_scale)
    # print(PSS_obj.workloads_ATT[0][1], PSS_TS_obj.workloads_ATT[0][1])

    # Diana_obj = Diana(args, graphs, nodes_list, adjs_list, args['world_size'], GCN_node_size, RNN_node_size, bandwidth_10MB, GCN_comp_scale, ATT_comp_scale)
    # Diana_obj.partitioning('LDG_DyG')
    # Diana_obj.communication_time(GCN_node_size, RNN_node_size, bandwidth_10MB, GCN_comp_scale, ATT_comp_scale)