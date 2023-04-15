import numpy as np
import scipy.sparse as sp
import argparse
import torch
import networkx as nx
import math
import random

# import matplotlib.pyplot as plt

from tqdm import tqdm
# from data_process import load_data, generate_graphs, graph_concat

import sys, os
sys.path.append(os.path.abspath('/home/DyG/'))

from MLDP.Model_evaluation.mlp import MLP_Predictor
from MLDP.util import graph_concat
# from MLDP.coarsen import coarsening

# Simulation setting
# node_size = 10
# bandwidth = float(1000)
bandwidth_1MB = float(1024*1024*8)
bandwidth_10MB = float(10*1024*1024*8)
bandwidth_100MB = float(100*1024*1024*8)
bandwidth_GB = float(1024*1024*1024*8)

'''
    public communication function
'''
def generate_test_graph():
    num_snapshots = 10
    nodes_list = [torch.tensor(np.array([j for j in range(3+i*3)])) for i in range(num_snapshots)]
    adjs_list = [torch.ones(nodes_list[i].size(0), nodes_list[i].size(0)).to_sparse() for i in range(num_snapshots)]

    return nodes_list, adjs_list

def GCN_comm_nodes(nodes_list, adjs_list, num_devices, workloads_GCN):
    '''
    每个GPU只关心出口处和入口处的节点总数量 （big switch）
    Receive:
        STEP 1: 获取每个GPU需要接受的节点列表(mask方法)
        STEP 2: 计算接收节点的时间开销
    Send:
        STEP 1: 获取每个GPU需要向其他GPU发送的节点列表(mask方法)
        STEP 2: 计算发送节点的时间开销
    Total:
        Max(Receive, Send)
    '''
    receive_list = [[] for i in range(num_devices)]
    send_list = [[] for i in range(num_devices)]
    for time in range(len(nodes_list)):
        for device_id in range(num_devices):
            adj = adjs_list[time].clone()
            local_node_mask = workloads_GCN[device_id][time]
            remote_node_mask = ~workloads_GCN[device_id][time]
            edge_source = adj._indices()[0]
            edge_target = adj._indices()[1]

            # receive
            edge_source_local_mask = local_node_mask[edge_source] # check each source node whether it belongs to device_id
            # need_receive_nodes = torch.unique(edge_target[edge_source_local_mask]) # get the target nodes with the source nodes belong to device_id
            need_receive_nodes = edge_target[edge_source_local_mask] # get the target nodes with the source nodes belong to device_id
            receive_node_local = local_node_mask[need_receive_nodes] # check whether the received nodes in local?
            receive = torch.nonzero(receive_node_local == False, as_tuple=False).squeeze() # only the received nodes are not in local
            receive_list[device_id].append(receive.view(-1))

            # send
            edge_source_remote_mask = remote_node_mask[edge_source] # check each source node whether it belongs to other devices
            need_send_nodes = edge_target[edge_source_remote_mask] # get the target nodes with the source nodes belong to other devices
            send_node_local = local_node_mask[need_send_nodes] # check whether the send nodes in local?
            send = torch.nonzero(send_node_local == True, as_tuple=False).squeeze() # only the send nodes are in local
            send_list[device_id].append(send.view(-1))
    
    return receive_list, send_list

# GCN workload is the same as the ATT workload
def RNN_comm_nodes(nodes_list, num_devices, workloads_GCN, workloads_RNN):
    '''
    Compute the communication for ATT processing
    Receive:
        STEP 1: 比较GCN workload和RNN workload的区别
        STEP 2: ATT workload中为True，而GCN workload中为False的点即为要接收的点
    Send:
        STEP 1: 比较GCN workload和RNN workload的区别
        STEP 2: ATT workload中为False，而GCN workload中为True的点即为要发送的点
    '''
    receive_list = [[] for i in range(num_devices)]
    send_list = [[] for i in range(num_devices)]
    for device_id in range(num_devices):
        for time in range(len(nodes_list)):
            GCN_workload = workloads_GCN[device_id][time]
            RNN_workload = workloads_RNN[device_id][time]
            RNN_true_where = torch.nonzero(RNN_workload == True, as_tuple=False).squeeze()
            RNN_false_where = torch.nonzero(RNN_workload == False, as_tuple=False).squeeze()

            RNN_true_GCN_mask = GCN_workload[RNN_true_where]
            RNN_false_GCN_mask = GCN_workload[RNN_false_where]

            # receive: ATT true and GCN false
            receive = torch.nonzero(RNN_true_GCN_mask == False, as_tuple=False).squeeze()
            # send: ATT false and GCN true
            send = torch.nonzero(RNN_false_GCN_mask == True, as_tuple=False).squeeze()

            receive_list[device_id].append(receive.view(-1))
            send_list[device_id].append(send.view(-1))
    
    return receive_list, send_list

def Comm_time(num_devices, receive_list, send_list, node_size, bandwidth):
    # compute time
    receive_comm_time = [0 for i in range(num_devices)]
    send_comm_time = [0 for i in range(num_devices)]
    for device_id in range(num_devices):
        # receive
        total_nodes = 0
        for receive in receive_list[device_id]:
            if receive != torch.Size([]):
                total_nodes += receive.view(-1).size(0)
        receive_comm_time[device_id] += np.around(float(total_nodes*node_size)/bandwidth, 3)

        # send
        total_nodes = 0
        for send in send_list[device_id]:
            if send != torch.Size([]):
                total_nodes += send.view(-1).size(0)
        send_comm_time[device_id] += np.around(float(total_nodes*node_size)/bandwidth, 3)
    return receive_comm_time, send_comm_time

# GCN workload is different from the ATT workload
def RNN_comm_nodes_new(nodes_list, num_devices, workload_GCN, workloads_RNN):
    '''
    Step 1: generate the required nodes list for each device
    Step 2: compare the required list with the ATT(GCN) workload list to compute the number of received nodes
    '''
    Req = [[torch.full_like(nodes_list[time], False, dtype=torch.bool) for time in range(len(nodes_list))] for m in range(num_devices)]
    receive_list = [[] for i in range(num_devices)]
    send_list = [[] for i in range(num_devices)]

    for m in range(num_devices):
        # compute the required node list
        for time in range(len(workloads_RNN[m])):
            where_need_comp = torch.nonzero(workloads_RNN[m][time] == True, as_tuple=False).view(-1)
            if where_need_comp!= torch.Size([]):
                for k in range(len(workloads_RNN[m]))[0:time+1]:
                    idx = torch.tensor([i for i in range(Req[m][k].size(0))])
                    need_nodes_mask = workloads_RNN[m][time][idx]
                    where_need = torch.nonzero(need_nodes_mask == True, as_tuple=False).view(-1)
                    # print(where_need)
                    if (where_need.size(0) > 0):
                        Req[m][k][where_need] = torch.ones(where_need.size(0), dtype=torch.bool)
        # remove already owned nodes
        for time in range(len(workload_GCN[m])):
            where_have_nodes = torch.nonzero(workload_GCN[m][time] == True, as_tuple=False).view(-1)
            # print(where_have_nodes)
            if where_have_nodes!= torch.Size([]):
                # print(where_have_nodes)
                Req[m][time][where_have_nodes] = torch.zeros(where_have_nodes.size(0), dtype=torch.bool)
    # print(Req)
    # Compute the number of nodes need to be sent
    for m in range(num_devices):
        for time in range(len(nodes_list)):
            receive = torch.nonzero(Req[m][time] == True, as_tuple=False).squeeze()
            receive_list[m].append(receive.view(-1))

            others_need = torch.zeros(nodes_list[time].size(0), dtype=torch.bool)
            for k in range(num_devices):
                where_other_need = torch.nonzero(Req[k][time] == True, as_tuple=False).view(-1)
                others_need[where_other_need] = torch.ones(where_other_need.size(0), dtype=torch.bool)
            where_have = torch.nonzero(workload_GCN[m][time] == True, as_tuple=False).view(-1)
            send_mask = others_need[where_have]
            send = torch.nonzero(send_mask == True, as_tuple=False).view(-1)
            send_list[m].append(send.view(-1))
    return receive_list, send_list

# compute the cross edges when schedule workload p(or q) on m device
def Cross_edges(timesteps, adjs, nodes_list, Degrees, current_workload, workload, flag):
    num = 0
    if flag == 0:
        # graph-graph cross edges at a timestep
        # method 1: compute cross edges per node with sparse tensor (slow but memory efficient)
        time = workload[0]
        nodes = workload[1].tolist()
        adj = adjs[time].clone()
        edge_source = adj._indices()[0]
        edge_target = adj._indices()[1]
        idx_list = [torch.nonzero(edge_source == node, as_tuple=False).view(-1) for node in nodes]
        nodes_idx_list = [edge_target[idx] for idx in idx_list if idx.dim() != 0]
        if len(nodes_idx_list) > 0:
            nodes_idx = torch.cat((nodes_idx_list), dim=0)
            has_nodes = torch.nonzero(current_workload[time][nodes_idx] == True, as_tuple=False).view(-1)
            num += has_nodes.size(0)/sum(Degrees[time])
        # print(num)

    # node-graph cross edges at multiple timesteps
    else:
        time = workload[0]
        node_id = workload[1]
        adj = adjs[time].clone()
        edge_source = adj._indices()[0]
        edge_target = adj._indices()[1]
        # print(edge_source, edge_target)
        idx = torch.nonzero(edge_source == node_id, as_tuple=False).view(-1)
        # print(idx)
        nodes_idx = edge_target[idx]
        # print(nodes_idx)
        has_nodes = torch.nonzero(current_workload[time][nodes_idx] == True, as_tuple=False).view(-1)
        # print('all degrees: ',sum(Degrees[time]))
        num += has_nodes.size(0)/sum(Degrees[time])
    return num

# compute the cross nodes when schedule workload p on m device
def Cross_nodes(timesteps, nodes_list, current_workload, workload):
    num = 0
    same_nodes = []
    for time in range(timesteps):
        if nodes_list[time][-1] >= workload[-1]:
            same_nodes.append(current_workload[time][workload])
    if len(same_nodes) > 0:
        same_nodes_tensor = torch.cat((same_nodes), dim=0)
        has_nodes = torch.nonzero(same_nodes_tensor == True, as_tuple=False).view(-1)
        num += has_nodes.size(0)/(workload.size(0)*len(same_nodes))
    # print(num)
    return num


# def plot_graph(args,G, node_color, edge_color, pos, flag):
    # node_color_mask = {}
    # node_color_mask[0] = 'red'
    # node_color_mask[1] = 'blue'
    # node_color_mask[2] = 'yellow'
    # node_color_mask[3] = 'purple'
    # # print([full_graph.nodes[node]['snap_id'][0] for node in list(full_graph)])
    # nodes_colors = [node_color_mask[G.nodes[node]['snap_id'][0]] for node in list(G)]
    
    # edge_color_mask = {}
    # edge_color_mask['str'] = 'black'
    # edge_color_mask['tem'] = 'green'
    # edges_colors = [edge_color_mask[G.edges[edge]['type']] for edge in G.edges()]

    # fig, ax = plt.subplots()
    # nx.draw(G, ax=ax, node_size=5, 
	#         width=0.5,
    #         pos=pos,
    #         node_color=node_color,
    #         edge_color=edge_color)
    # plt.savefig('./experiment_results/{}_graph_{}.png'.format(flag, args['dataset']), dpi=300)
    # plt.close()

def Computation_time(graphs, num_devices, timesteps, workload_GCN, workload_RNN, GCN_comp_scale, ATT_comp_scale):
    device = torch.device("cuda")
    model_str = MLP_Predictor(in_feature = 2)
    model_str.load_state_dict(torch.load('./Model_evaluation/model/str_{}.pt'.format(10)))
    model_str = model_str.to(device)

    model_tem = MLP_Predictor(in_feature = 2)
    model_tem.load_state_dict(torch.load('./Model_evaluation/model/tem_{}.pt'.format(10)))
    model_tem = model_tem.to(device)

    model_str.eval()
    model_tem.eval()

    GCN_time = [0 for i in range(num_devices)]
    RNN_time = [0 for i in range(num_devices)]
    for m in range(num_devices):
        gcn_comp_time = 0
        rnn_comp_time = 0
        full_nodes = []
        total_time_step = 0
        for t in range(timesteps):
            nodes_list = torch.nonzero(workload_GCN[m][t] == True, as_tuple=False).view(-1).tolist()
            graph = graphs[t].subgraph(nodes_list)
            num_vertices = graph.number_of_nodes()
            num_edges = graph.number_of_edges()
            input = torch.Tensor([float(num_vertices/10000), float(num_edges/10000)]).to(device)
            cost = model_str(input)
            gcn_comp_time += cost.item()
            if len(nodes_list) > 0:
                full_nodes.extend(nodes_list)
                total_time_step += 1
        full_nodes = list(set(full_nodes))
        tem_input = torch.Tensor([float(len(full_nodes)/10000), float(total_time_step/10)]).to(device)
        cost = model_tem(tem_input)
        rnn_comp_time += cost.item()
        GCN_time[m] += round(gcn_comp_time, 4)
        RNN_time[m] += round(rnn_comp_time, 4)


    # GCN_time = [0 for i in range(num_devices)]
    # RNN_time = [0 for i in range(num_devices)]
    # for m in range(num_devices):
    #     GCN_nodes = sum([torch.nonzero(workload_GCN[m][t] == True, as_tuple=False).view(-1).size(0) for t in range(timesteps)])
    #     RNN_nodes = [torch.nonzero(workload_RNN[m][t] == True, as_tuple=False).view(-1) for t in range(timesteps)]
    #     RNN_all_nodes = torch.cat(RNN_nodes, dim=0).unique()
    #     GCN_time[m] += np.around(float(GCN_nodes * GCN_comp_scale), 3)
    #     RNN_time[m] += np.around(float(GCN_nodes * ATT_comp_scale), 3)

    return GCN_time, RNN_time

def Distribution(nodes_list, timesteps, num_device, workload):
    distribution = torch.tensor([[0 for t in range(timesteps - 10)] for node in range(nodes_list[9].size(0))])


    for m in range(num_device):
        for t in range(timesteps)[10:]:
            idx = torch.nonzero(workload[m][t][:nodes_list[9].size(0)] == True, as_tuple=False).view(-1)
            # print(idx)
            distribution[idx,t-10] = m
    mask = []
    for gen in range(10):
        for i in range(4):
            mask.append(nodes_list[gen].size(0) + i -4)

    # print(mask)
    mask_tensor = torch.tensor(mask)
    distribution = distribution[mask_tensor,:]
    return distribution.tolist()

def Cross_edge_between_generations(nodes_list, adjs_list, timesteps):
    cluster_timestep_list = []
    cluster_node_list = []
    num_edge_list = []
    num_node_list_0 = []
    num_node_list_1 = []
    for t in range(timesteps):
        generation = t + 1
        for gen in range(generation):
            cluster_timestep_list.append(t)
            if gen > 0:
                cluster_node_list.append(nodes_list[t][nodes_list[gen - 1].size(0) : nodes_list[gen].size(0)])
            else:
                cluster_node_list.append(nodes_list[t][:nodes_list[gen].size(0)])

        if t >= 1:
            gen_0 = nodes_list[t][:nodes_list[0].size(0)]
            gen_1 = nodes_list[t][nodes_list[0].size(0):nodes_list[1].size(0)]
            adj = adjs_list[t].clone()
            edge_source = adj._indices()[0]
            edge_target = adj._indices()[1]

            has_gen_0_mask = torch.zeros(nodes_list[t].size(0), dtype=torch.bool)
            has_gen_0_mask[gen_0] = torch.ones(gen_0.size(0), dtype=torch.bool)
            has_edge_mask = has_gen_0_mask[edge_source]
            gen_0_edge_idx = torch.nonzero(has_edge_mask == True, as_tuple=False).view(-1)
            # gen_0_edge = edge_source[has_edge_mask]

            has_gen_1_mask = torch.zeros(nodes_list[t].size(0), dtype=torch.bool)
            has_gen_1_mask[gen_1] = torch.ones(gen_1.size(0), dtype=torch.bool)
            edge_target = edge_target[gen_0_edge_idx]
            edge_between_generations = torch.nonzero(has_gen_1_mask[edge_target] == True).view(-1)


            # edge_between_generations = torch.nonzero(has_gen_1_mask[gen_0_has_edge] == True, as_tuple=False).view(-1)

            num_edge_list.append(edge_between_generations.size(0))
            num_node_list_0.append(gen_0.size(0))
            num_node_list_1.append(gen_1.size(0))
    print('cross edge info: ', num_edge_list, num_node_list_0, num_node_list_1)


# different partition methods
class node_partition():
    def __init__(self, args, graphs, nodes_list, adjs_list, num_devices):
        super(node_partition, self).__init__()
        '''
        //Parameter
            nodes_list: a list, in which each element is a [N_i,1] tensor to represent a node list of i-th snapshot
            adjs_list: a list, in which each element is a [N_i, N_i] tensor to represent a adjacency matrix of i-th snapshot
            num_devices: an integer constant to represent the number of devices
        '''
        self.args = args
        self.graphs = graphs
        self.nodes_list = nodes_list
        self.adjs_list = adjs_list
        self.num_devices = num_devices
        self.workload = [[] for i in range(num_devices)]

        self.workload_partition()

    def workload_partition(self):
        '''
        用bool来表示每个snapshot中的每个点属于哪一块GPU
        '''
        # for time, nodes in enumerate(self.nodes_list):
        #     num_of_nodes = nodes.size(0)
        #     nodes_per_device = num_of_nodes//self.num_devices
        #     for device_id in range(self.num_devices):
        #         work = torch.full_like(nodes, False, dtype=torch.bool)
        #         if device_id != self.num_devices - 1:
        #             work[nodes_per_device*device_id:nodes_per_device*(device_id+1)] = torch.ones(nodes_per_device, dtype=torch.bool)
        #         else:
        #             work[nodes_per_device*device_id:] = torch.ones(num_of_nodes - ((self.num_devices -1)*nodes_per_device), dtype=torch.bool)

        #         self.workload[device_id].append(work)
        
        num_nodes = self.nodes_list[-1].size(0)
        nodes_per_device = num_nodes // self.num_devices  # to guarantee the all temporal information of a same node will be in the same device
        node_partition_id = torch.tensor([0 for i in range(num_nodes)])
        for device_id in range(self.num_devices):
            if device_id != self.num_devices - 1:
                temp = torch.tensor([device_id for i in range(nodes_per_device)])
                node_partition_id[device_id*nodes_per_device:(device_id + 1)*nodes_per_device] = temp
            else:
                temp = torch.tensor([device_id for i in range(num_nodes - (self.num_devices - 1)*nodes_per_device)])
                node_partition_id[device_id*nodes_per_device:] = temp
        for device_id in range(self.num_devices):
            where_nodes = torch.nonzero(node_partition_id == device_id, as_tuple=False).squeeze()
            # print(where_nodes)
            nodes_local_mask = torch.full_like(self.nodes_list[-1], False, dtype=torch.bool)
            nodes_local_mask[where_nodes] = torch.ones(where_nodes.size(0), dtype=torch.bool)
            for (time, nodes) in enumerate(self.nodes_list):
                work = nodes_local_mask[nodes]
                self.workload[device_id].append(work)

    def communication_time(self, GCN_node_size, RNN_node_size, bandwidth, GCN_comp_scale, ATT_comp_scale):
        GCN_receive_list, GCN_send_list = GCN_comm_nodes(self.nodes_list, self.adjs_list, self.num_devices, self.workload)
        # print(GCN_receive_list)
        GCN_receive_comm_time, GCN_send_comm_time = Comm_time(self.num_devices, GCN_receive_list, GCN_send_list, GCN_node_size, bandwidth)

        GCN_receive = [torch.cat(GCN_receive_list[i], 0).size(0) for i in range(self.num_devices)]
        GCN_send = [torch.cat(GCN_send_list[i], 0).size(0) for i in range(self.num_devices)]
        RNN_receive = [0 for i in range(self.num_devices)]
        RNN_send = [0 for i in range(self.num_devices)]

        GCN_comm_time = [max(GCN_receive_comm_time[i], GCN_send_comm_time[i]) for i in range(len(GCN_receive_comm_time))]
        RNN_comm_time = [0 for i in range(self.num_devices)]
        GPU_total_time = [GCN_comm_time[i] + RNN_comm_time[i] for i in range(len(GCN_comm_time))]

        # Total_time = max(GPU_total_time)
        GCN_comp_time, ATT_comp_time = Computation_time(self.graphs, self.num_devices, len(self.nodes_list), self.workload, self.workload, GCN_comp_scale, ATT_comp_scale)
        total_comp_time = [GCN_comp_time[i] + ATT_comp_time[i] for i in range(len(GCN_comm_time))]
        print('----------------------------------------------------------')
        print('node partition method:')
        print('GCN | Each GPU receives nodes: {} | Each GPU sends nodes: {}'.format(GCN_receive, GCN_send))
        print('ATT | Each GPU receives nodes: {} | Each GPU sends nodes: {}'.format(RNN_receive, RNN_send))
        print('Each GPU with computation time: {} ( GCN: {} | ATT: {})'.format(total_comp_time, GCN_comp_time, ATT_comp_time))
        print('Each GPU with communication time: {} ( GCN: {} | ATT: {})'.format(GPU_total_time, GCN_comm_time, RNN_comm_time))
        print('Total time: {} | Computation time: {}, Communication time: {}'.format(max(GPU_total_time) + max(GCN_comp_time) + max(ATT_comp_time), max(GCN_comp_time) + max(ATT_comp_time), max(GPU_total_time)))


class snapshot_partition():
    def __init__(self, args, graphs, nodes_list, adjs_list, num_devices):
        super(snapshot_partition, self).__init__()
        '''
        Snapshot partition [SC'21] first partitions the dynamic graphs via temporal dimention and then partition the workload via spatio dimention
        //Parameter
            nodes_list: a list, in which each element is a [N_i,1] tensor to represent a node list of i-th snapshot
            adjs_list: a list, in which each element is a [N_i, N_i] tensor to represent a adjacency matrix of i-th snapshot
            num_devices: an integer constant to represent the number of devices
        '''
        self.args = args
        self.graphs = graphs
        self.nodes_list = nodes_list
        self.adjs_list = adjs_list
        self.num_devices = num_devices
        self.workloads_GCN = [[] for i in range(num_devices)]
        self.workloads_ATT = [[] for i in range(num_devices)]

        self.workload_partition()
    
    def workload_partition(self):
        '''
        STEP 1: partition graphs via temporal dimension (for GCN process)
        STEP 2: partition graphs via spatio dimension (for ATT process)
        '''
        # temporal partition
        timesteps = len(self.nodes_list)
        time_per_device = timesteps // self.num_devices
        time_partition_id = torch.tensor([0 for i in range(timesteps)])
        for device_id in range(self.num_devices):
            if device_id != self.num_devices - 1:
                temp = torch.tensor([device_id for i in range(time_per_device)])
                time_partition_id[device_id*time_per_device:(device_id + 1)*time_per_device] = temp
            else:
                temp = torch.tensor([device_id for i in range(timesteps - (self.num_devices - 1)*time_per_device)])
                time_partition_id[device_id*time_per_device:] = temp
        for (time, nodes) in enumerate(self.nodes_list):
            for device_id in range(self.num_devices):
                if time_partition_id[time] == device_id:
                    work = torch.full_like(nodes, True, dtype=torch.bool)
                else:
                    work = torch.full_like(nodes, False, dtype=torch.bool)
                self.workloads_GCN[device_id].append(work)
        # print(self.workload_GCN[-1])

        # spatio partition
        num_nodes = self.nodes_list[-1].size(0)
        nodes_per_device = num_nodes // self.num_devices  # to guarantee the all temporal information of a same node will be in the same device
        node_partition_id = torch.tensor([0 for i in range(num_nodes)])
        for device_id in range(self.num_devices):
            if device_id != self.num_devices - 1:
                temp = torch.tensor([device_id for i in range(nodes_per_device)])
                node_partition_id[device_id*nodes_per_device:(device_id + 1)*nodes_per_device] = temp
            else:
                temp = torch.tensor([device_id for i in range(num_nodes - (self.num_devices - 1)*nodes_per_device)])
                node_partition_id[device_id*nodes_per_device:] = temp
        for device_id in range(self.num_devices):
            where_nodes = torch.nonzero(node_partition_id == device_id, as_tuple=False).squeeze()
            # print(where_nodes)
            nodes_local_mask = torch.full_like(self.nodes_list[-1], False, dtype=torch.bool)
            nodes_local_mask[where_nodes] = torch.ones(where_nodes.size(0), dtype=torch.bool)
            for (time, nodes) in enumerate(self.nodes_list):
                work = nodes_local_mask[nodes]
                self.workloads_ATT[device_id].append(work)

    def communication_time(self, GCN_node_size, RNN_node_size, bandwidth, GCN_comp_scale, ATT_comp_scale):
        RNN_receive_list, RNN_send_list = RNN_comm_nodes(self.nodes_list, self.num_devices, self.workloads_GCN, self.workloads_ATT)

        RNN_receive_comm_time, RNN_send_comm_time = Comm_time(self.num_devices, RNN_receive_list, RNN_send_list, RNN_node_size, bandwidth)

        GCN_receive = [0 for i in range(self.num_devices)]
        GCN_send = [0 for i in range(self.num_devices)]
        RNN_receive = [torch.cat(RNN_receive_list[i], 0).size(0) for i in range(self.num_devices)]
        RNN_send = [torch.cat(RNN_send_list[i], 0).size(0) for i in range(self.num_devices)]

        GCN_comm_time = [0 for i in range(self.num_devices)]
        RNN_comm_time = [max(RNN_receive_comm_time[i], RNN_send_comm_time[i])*2 for i in range(len(RNN_receive_comm_time))]
        GPU_total_time = [GCN_comm_time[i] + RNN_comm_time[i] for i in range(len(GCN_comm_time))]
        # Total_time = max(GPU_total_time)
        GCN_comp_time, ATT_comp_time = Computation_time(self.graphs, self.num_devices, len(self.nodes_list), self.workloads_GCN, self.workloads_ATT, GCN_comp_scale, ATT_comp_scale)
        total_comp_time = [GCN_comp_time[i] + ATT_comp_time[i] for i in range(len(GCN_comm_time))]
        print('----------------------------------------------------------')
        print('snapshot partition method:')
        print('GCN | Each GPU receives nodes: {} | Each GPU sends nodes: {}'.format(GCN_receive, GCN_send))
        print('ATT | Each GPU receives nodes: {} | Each GPU sends nodes: {}'.format(RNN_receive, RNN_send))
        print('Each GPU with computation time: {} ( GCN: {} | ATT: {})'.format(total_comp_time, GCN_comp_time, ATT_comp_time))
        print('Each GPU with communication time: {} ( GCN: {} | ATT: {})'.format(GPU_total_time, GCN_comm_time, RNN_comm_time))
        print('Total time: {} | Computation time: {}, Communication time: {}'.format(max(GPU_total_time) + max(GCN_comp_time) + max(ATT_comp_time), max(GCN_comp_time) + max(ATT_comp_time), max(GPU_total_time)))


class hybrid_partition():
    def __init__(self, args, nodes_list, adjs_list, num_devices):
        super(hybrid_partition, self).__init__()
        '''
        Snapshot partition [SC'21] first partitions the dynamic graphs via temporal dimention and then partition the workload via spatio dimention
        //Parameter
            nodes_list: a list, in which each element is a [N_i,1] tensor to represent a node list of i-th snapshot
            adjs_list: a list, in which each element is a [N_i, N_i] tensor to represent a adjacency matrix of i-th snapshot
            num_devices: an integer constant to represent the number of devices
        '''
        self.args = args
        self.nodes_list = nodes_list
        self.adjs_list = adjs_list
        self.num_devices = num_devices
        self.workloads_GCN = [[] for i in range(num_devices)]
        self.workloads_ATT = [[] for i in range(num_devices)]

        self.workload_partition()
    
    def workload_partition(self):
        '''
        先按点划分，因为点列表中，最前面的点拥有最长的时序
        再按时序划分，划分时注意每个时序图中已经被分配的节点
        '''
        partition_method = [0 for i in range(self.num_devices)] # 0: node partition; 1: snapshot partition
        for i in range(self.num_devices):
            if i >= (self.num_devices // 2):
                partition_method[i] = 1 # snapshot partition
        
        # STEP 1: the same ATT workloads
        num_nodes = self.nodes_list[-1].size(0)
        nodes_per_device = num_nodes // self.num_devices  # to guarantee the all temporal information of a same node will be in the same device
        node_partition_id = torch.tensor([0 for i in range(num_nodes)])
        for device_id in range(self.num_devices):
            if device_id != self.num_devices - 1:
                temp = torch.tensor([device_id for i in range(nodes_per_device)])
                node_partition_id[device_id*nodes_per_device:(device_id + 1)*nodes_per_device] = temp
            else:
                temp = torch.tensor([device_id for i in range(num_nodes - (self.num_devices - 1)*nodes_per_device)])
                node_partition_id[device_id*nodes_per_device:] = temp
        for device_id in range(self.num_devices):
            where_nodes = torch.nonzero(node_partition_id == device_id, as_tuple=False).squeeze()
            # print(where_nodes)
            nodes_local_mask = torch.full_like(self.nodes_list[-1], False, dtype=torch.bool)
            nodes_local_mask[where_nodes] = torch.ones(where_nodes.size(0), dtype=torch.bool)
            for (time, nodes) in enumerate(self.nodes_list):
                work = nodes_local_mask[nodes]
                self.workloads_ATT[device_id].append(work)
                # if partition_method[device_id] == 0:
                #     self.workloads_GCN[device_id].append(work)

        # print(self.workloads_ATT[0])
        # STEP 2: different partitions
        # num_nodes = self.nodes_list[-1].size(0)
        # nodes_per_device = num_nodes // 3
        for device_id in range(self.num_devices):
            if partition_method[device_id] == 0:
                if device_id != self.num_devices - 1:
                    temp = torch.tensor([device_id for i in range(nodes_per_device)])
                    node_partition_id[device_id*nodes_per_device:(device_id + 1)*nodes_per_device] = temp
                else:
                    temp = torch.tensor([device_id for i in range(num_nodes - (self.num_devices - 1)*nodes_per_device)])
                    node_partition_id[device_id*nodes_per_device:] = temp
        for device_id in range(self.num_devices):
            if partition_method[device_id] == 0:
                where_nodes = torch.nonzero(node_partition_id == device_id, as_tuple=False).squeeze()
                # print(where_nodes)
                nodes_local_mask = torch.full_like(self.nodes_list[-1], False, dtype=torch.bool)
                nodes_local_mask[where_nodes] = torch.ones(where_nodes.size(0), dtype=torch.bool)
                for (time, nodes) in enumerate(self.nodes_list):
                    work = nodes_local_mask[nodes]
                    self.workloads_GCN[device_id].append(work)

        # print(self.workloads_GCN[0])
        # STEP 3: snapshot partition
        # update graphs: if all nodes in some snapshots are partitioned already, these snapshot should not be partitioned again
        whether_partitioned = torch.zeros(len(self.nodes_list), dtype=torch.bool)
        partition_nodes_list = []
        for (time, node) in enumerate(self.nodes_list):
            partition_nodes = []
            for device_id in range(self.num_devices):
                if partition_method[device_id] == 0:
                    partition_nodes.append(torch.nonzero(self.workloads_GCN[device_id][time] == True, as_tuple=False).squeeze())
            already_partitioned = torch.cat(partition_nodes, dim=0)
            if node.size(0) == already_partitioned.size(0):
                whether_partitioned[time] = True
            partition_nodes_list.append(already_partitioned)
        need_partition_snapshot = torch.nonzero(whether_partitioned == False, as_tuple=False).squeeze()
        timesteps = need_partition_snapshot.size(0)
        partitioned_timesteps = len(self.nodes_list) - timesteps
        devices_for_snapshot_partition = self.num_devices - (self.num_devices//2)
        time_per_device = timesteps // devices_for_snapshot_partition

        snapshot_partition_id = torch.tensor([-1 for i in range(len(self.nodes_list))])
        # print(devices_for_snapshot_partition)
        # print(time_per_device)
        for device_id in range(devices_for_snapshot_partition):
            if device_id != devices_for_snapshot_partition - 1:
                temp = torch.tensor([device_id + self.num_devices//2 for i in range(time_per_device)])
                snapshot_partition_id[partitioned_timesteps + device_id*time_per_device:partitioned_timesteps + (device_id + 1)*time_per_device] = temp
            else:
                temp = torch.tensor([device_id + self.num_devices//2 for i in range(timesteps - (devices_for_snapshot_partition - 1)*time_per_device)])
                snapshot_partition_id[partitioned_timesteps + device_id*time_per_device:] = temp
        # print(snapshot_partition_id)
        for device_id in range(self.num_devices):
            if partition_method[device_id] == 1:
                for (time, nodes) in enumerate(self.nodes_list):
                    if whether_partitioned[time] == False and snapshot_partition_id[time] == device_id:
                        work = torch.full_like(nodes, True, dtype=torch.bool)
                        work[partition_nodes_list[time]] = torch.zeros(partition_nodes_list[time].size(0), dtype=torch.bool)
                        self.workloads_GCN[device_id].append(work)
                    else:
                        work = torch.full_like(nodes, False, dtype=torch.bool)
                        self.workloads_GCN[device_id].append(work)
        # print(self.workloads_GCN[-1])

    def communication_time(self, GCN_node_size, RNN_node_size, bandwidth):
        '''
        Both GCN communication time and ATT communication time are needed
        '''
        GCN_receive_list, GCN_send_list = GCN_comm_nodes(self.nodes_list, self.adjs_list, self.num_devices, self.workloads_GCN)
        RNN_receive_list, RNN_send_list = RNN_comm_nodes(self.nodes_list, self.num_devices, self.workloads_GCN, self.workloads_ATT)

        GCN_receive_comm_time, GCN_send_comm_time = Comm_time(self.num_devices, GCN_receive_list, GCN_send_list, GCN_node_size, bandwidth)
        RNN_receive_comm_time, RNN_send_comm_time = Comm_time(self.num_devices, RNN_receive_list, RNN_send_list, RNN_node_size, bandwidth)

        GCN_receive = [torch.cat(GCN_receive_list[i], 0).size(0) for i in range(self.num_devices)]
        GCN_send = [torch.cat(GCN_send_list[i], 0).size(0) for i in range(self.num_devices)]
        RNN_receive = [torch.cat(RNN_receive_list[i], 0).size(0) for i in range(self.num_devices)]
        RNN_send = [torch.cat(RNN_send_list[i], 0).size(0) for i in range(self.num_devices)]

        GCN_comm_time = [max(GCN_receive_comm_time[i], GCN_send_comm_time[i]) for i in range(len(GCN_receive_comm_time))]
        RNN_comm_time = [max(RNN_receive_comm_time[i], RNN_send_comm_time[i]) for i in range(len(RNN_receive_comm_time))]
        GPU_total_time = [GCN_comm_time[i] + RNN_comm_time[i] for i in range(len(GCN_comm_time))]
        # Total_time = max(GPU_total_time)

        Comp_time = Computation_time(self.num_devices, len(self.nodes_list), self.workloads_GCN, self.workloads_ATT)

        print('----------------------------------------------------------')
        print('Hybrid partition method:')
        print('GCN | Each GPU receives nodes: {} | Each GPU sends nodes: {}'.format(GCN_receive, GCN_send))
        print('ATT | Each GPU receives nodes: {} | Each GPU sends nodes: {}'.format(RNN_receive, RNN_send))
        print('Each GPU with communication time: {} ( GCN: {} | ATT: {})'.format(GPU_total_time, GCN_comm_time, RNN_comm_time))
        print('Total time: {} | Computation time: {}, Communication time: {}'.format(max(GPU_total_time) + Comp_time, Comp_time, max(GPU_total_time)))

class snapshot_partition_balance():
    def __init__(self, args, graphs, nodes_list, adjs_list, num_devices):
        super(snapshot_partition_balance, self).__init__()
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
        Total_workload = [torch.full_like(self.nodes_list[time], 1) for time in range(self.timesteps)]
        Total_workload_RNN = [torch.full_like(self.nodes_list[time], 1) for time in range(self.timesteps)]
        num_generations = [i+1 for i in range(self.timesteps)]
        P_id = [] # save the snapshot id
        Q_id = []
        Q_node_id = []
        P_workload = [] # save the workload size
        P_snapshot = []
        Q_workload = []
        Degree = []
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
        print('Snapshot-clustering + LDG:')
        print('GCN | Each GPU receives nodes: {} | Each GPU sends nodes: {}'.format(GCN_receive, GCN_send))
        print('ATT | Each GPU receives nodes: {} | Each GPU sends nodes: {}'.format(RNN_receive, RNN_send))
        print('Each GPU with computation time: {} ( GCN: {} | ATT: {})'.format(total_comp_time, GCN_comp_time, ATT_comp_time))
        print('Each GPU with communication time: {} ( GCN: {} | ATT: {})'.format(GPU_total_time, GCN_comm_time, RNN_comm_time))
        print('Total time: {} | Computation time: {}, Communication time: {}'.format(max(GPU_total_time) + max(GCN_comp_time) + max(ATT_comp_time), max(GCN_comp_time) + max(ATT_comp_time), max(GPU_total_time)))


class node_partition_balance():
    def __init__(self, args, graphs, nodes_list, adjs_list, num_devices):
        super(node_partition_balance, self).__init__()
        self.args = args
        self.nodes_list = nodes_list
        self.adjs_list = adjs_list
        self.num_devices = num_devices
        self.graphs = graphs
        self.timesteps = len(nodes_list)
        self.workloads_GCN = [[torch.zeros(nodes_list[-1].size(0), dtype=torch.bool) for time in range(self.timesteps)] for i in range(num_devices)]
        self.workloads_ATT = [[torch.zeros(nodes_list[-1].size(0), dtype=torch.bool) for time in range(self.timesteps)] for i in range(num_devices)]

        # runtime
        P_id, Q_id, Q_node_id, P_workload, P_snapshot, Q_workload = self.partition()

        self.schedule(P_id, Q_id, Q_node_id, P_workload, P_snapshot, Q_workload)

    def partition(self):
        '''
        Step 1: partition snapshot into P set; partition nodes into Q set
        '''
        Total_workload = [torch.full_like(self.nodes_list[time], 1) for time in range(self.timesteps)]
        Total_workload_RNN = [torch.full_like(self.nodes_list[time], 1) for time in range(self.timesteps)]
        num_generations = [i+1 for i in range(self.timesteps)]
        P_id = [] # save the snapshot id
        Q_id = []
        Q_node_id = []
        P_workload = [] # save the workload size
        P_snapshot = []
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
        return P_id, Q_id, Q_node_id, P_workload, P_snapshot, Q_workload
    
    def schedule(self, P_id, Q_id, Q_node_id, P_workload, P_snapshot, Q_workload):
        Scheduled_workload = [torch.full_like(self.nodes_list[time], False, dtype=torch.bool) for time in range(self.timesteps)]
        Current_GCN_workload = [0 for i in range(self.num_devices)]
        Current_RNN_workload = [0 for i in range(self.num_devices)]
        # compute the average workload
        GCN_avg_workload = np.sum(P_workload)/self.num_devices
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
        # print('GCN workload after scheduling timeseries-level jobs: ', self.workloads_GCN)
    
    def get_partition(self):
        return self.workloads_GCN, self.workloads_ATT

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
        print('Node-clustering + LDG:')
        print('GCN | Each GPU receives nodes: {} | Each GPU sends nodes: {}'.format(GCN_receive, GCN_send))
        print('ATT | Each GPU receives nodes: {} | Each GPU sends nodes: {}'.format(RNN_receive, RNN_send))
        print('Each GPU with computation time: {} ( GCN: {} | ATT: {})'.format(total_comp_time, GCN_comp_time, ATT_comp_time))
        print('Each GPU with communication time: {} ( GCN: {} | ATT: {})'.format(GPU_total_time, GCN_comm_time, RNN_comm_time))
        print('Total time: {} | Computation time: {}, Communication time: {}'.format(max(GPU_total_time) + max(GCN_comp_time) + max(ATT_comp_time), max(GCN_comp_time) + max(ATT_comp_time), max(GPU_total_time)))

class Ours():
    def __init__(self, args, graphs, nodes_list, adjs_list, num_devices):
        super(Ours, self).__init__()
        self.args = args
        self.nodes_list = nodes_list
        self.adjs_list = adjs_list
        self.num_devices = num_devices
        self.graphs = graphs
        self.timesteps = len(nodes_list)
        self.workloads_GCN = [[torch.full_like(self.nodes_list[time], False, dtype=torch.bool) for time in range(self.timesteps)] for i in range(num_devices)]
        self.workloads_ATT = [[torch.full_like(self.nodes_list[time], False, dtype=torch.bool) for time in range(self.timesteps)] for i in range(num_devices)]

        self.Degrees = [list(dict(nx.degree(self.graphs[t])).values()) for t in range(self.timesteps)]

        # parameters
        self.alpha = args['alpha']
        # self.alpha = 0.08
        # self.alpha = 0.01
        # self.alpha = 0.1

        # runtime
        start = time.time()
        P_id, Q_id, Q_node_id, P_workload, P_snapshot, Q_workload = self.divide()
        # print('divide time cost: ', time.time() - start)
        # print('P_id: ',P_id)
        # print('Q_id: ',Q_id)
        # print('Q_node_id: ',Q_node_id)
        # print('P_workload: ',P_workload)
        # print('Q_workload: ',Q_workload)
        # print('P_snapshot: ',P_snapshot)
        start = time.time()
        self.conquer(P_id, Q_id, Q_node_id, P_workload, P_snapshot, Q_workload)
        # print('conquer time cost: ', time.time() - start)

    def divide(self):
        '''
        Step 1: compute the average degree of each snapshots
        Step 2: divide nodes into different job set according to the degree and time-series length
        '''
        Total_workload = [torch.full_like(self.nodes_list[time], 1) for time in range(self.timesteps)]
        Total_workload_temp = [torch.full_like(self.nodes_list[time], 1) for time in range(self.timesteps)]
        num_generations = [[j for j in range(i+1)] for i in range(self.timesteps)]
        P_id = [] # save the snapshot id
        Q_id = []
        Q_node_id = []
        P_workload = [] # save the workload size
        P_snapshot = []
        Q_workload = []
        Degree = []
        for t in range(self.timesteps):
            for generation in num_generations[t]:
                # compute average degree of nodes in specific generation
                if generation == 0:
                    start = 0
                else:
                    start = self.nodes_list[generation - 1].size(0)
                end = self.nodes_list[generation].size(0)
                Degree_list = list(dict(nx.degree(self.graphs[t])).values())[start:end]
                avg_deg = np.mean(Degree_list)
                Degree.append(avg_deg)
                # print('alpha: ',self.alpha)
                # print('generation; ',generation)
                workload = self.nodes_list[t][start:end]
                if avg_deg > self.alpha*(self.timesteps - t): # GCN-sensitive job
                    P_id.append(t)
                    P_workload.append(workload.size(0))
                    P_snapshot.append(workload)
                else:
                    for node in workload.tolist():
                        Q_id.append(t)
                        # divided_nodes = self.nodes_list[time].size(0) - workload.size(0)
                        Q_node_id.append(node)
                        Q_workload.append(self.timesteps - t)
                    # update following snapshots
                    for k in range(self.timesteps)[t+1:]:
                        mask = torch.full_like(Total_workload[k], True, dtype=torch.bool)
                        mask[start:end] = torch.zeros(mask[start:end].size(0), dtype=torch.bool)
                        where = torch.nonzero(mask == True, as_tuple=False).view(-1)
                        Total_workload[k] = Total_workload[k][where]
                        num_generations[k] = num_generations[k][1:]


            # # compute average degree of the graphs
            # Degree_list = list(dict(nx.degree(self.graphs[time])).values())
            # avg_deg = np.mean(Degree_list)
            # Degree.append(avg_deg)

            # if avg_deg > self.alpha*(self.timesteps - time): # GCN-sensitive job
            #     P_id.append(time)
            #     P_workload.append(Total_workload[time].size(0))
            # else:                                            # ATT-sensitive job
            #     for node in range(Total_workload[time].size(0)):
            #         Q_id.append(time)
            #         divided_nodes = self.nodes_list[time].size(0) - Total_workload[time].size(0)
            #         Q_node_id.append(node + divided_nodes)
            #         Q_workload.append(self.timesteps - time)
            #     # update following snapshots
            #     for k in range(self.timesteps)[time+1:]:
            #         update_size = Total_workload[time].size(0)
            #         Total_workload[k] = Total_workload[k][update_size:]

        return P_id, Q_id, Q_node_id, P_workload, P_snapshot, Q_workload
    
    def conquer(self, P_id, Q_id, Q_node_id, P_workload, P_snapshot, Q_workload):
        '''
        Schedule snapshot-level jobs first or schedule timeseries-level jobs first?
        '''
        Scheduled_workload = [torch.full_like(self.nodes_list[t], False, dtype=torch.bool) for t in range(self.timesteps)]
        Current_workload = [0 for i in range(self.num_devices)]
        Current_RNN_workload = [[0 for i in range(self.timesteps)]for m in range(self.num_devices)]
        # compute the average workload
        avg_workload = (sum(P_workload) + sum(Q_workload))/self.num_devices
        RNN_avg_workload = np.sum(Q_workload)/self.num_devices


        time_cost = 0
        timeseries_per_device = len(Q_id) // self.num_devices
        for idx in range(len(Q_id)):
            Load = []
            Cross_edge = []
            for m in range(self.num_devices):
                Load.append(1 - float((Current_workload[m] + Q_workload[idx])/avg_workload))
                start = time.time()
                # Cross_edge.append(Cross_edges(self.timesteps, self.adjs_list, self.nodes_list, self.Degrees, self.workloads_GCN[m], (Q_id[idx], Q_node_id[idx]), flag=1))
                time_cost += time.time() - start
            # Cross_edge = [ce*self.args['beta'] for ce in Cross_edge]
            # result = np.sum([Load, Cross_edge], axis = 0).tolist()
            # select_m = result.index(max(result))
            # select_m = Load.index(max(Load))
            select_m = idx // timeseries_per_device
            if select_m >= self.num_devices:
                select_m = self.num_devices - 1
            # for m in range(self.num_devices):
            #     if m == select_m:
            for t in range(self.timesteps)[Q_id[idx]:]:
                # print(self.workloads_GCN[m][time])
                self.workloads_GCN[select_m][t][Q_node_id[idx]] = torch.ones(1, dtype=torch.bool)
                self.workloads_ATT[select_m][t][Q_node_id[idx]] = torch.ones(1, dtype=torch.bool)
                # Scheduled_workload[time][Q_node_id[idx]] = torch.ones(1, dtype=torch.bool)
            Current_workload[select_m] = Current_workload[select_m] + Q_workload[idx]
        # print('compute graph-graph cross edges time costs: ', time_cost_edges)
        # print('compute cross nodes time costs: ', time_cost_nodes)
        # print('GCN workload after scheduling snapshot-level jobs: ', self.workloads_GCN)

        # print('compute node-graph cross edges time costs: ', time_cost)
        # print('GCN workload after scheduling timeseries-level jobs: ', self.workloads_GCN)


        time_cost_edges = 0
        time_cost_nodes = 0
        snapshot_per_device = len(P_id) // self.num_devices
        for idx in range(len(P_id)): # schedule snapshot-level job
            Load = []
            Cross_edge = []
            Cross_node = []
            for m in range(self.num_devices):
                Load.append(1 - float((Current_workload[m]+P_workload[idx])/avg_workload))
                # Cross_edge.append(Current_RNN_workload[m][P_id[idx]])
                start = time.time()
                # Cross_edge.append(Cross_edges(self.timesteps, self.adjs_list, self.nodes_list, self.Degrees, self.workloads_GCN[m], (P_id[idx],P_snapshot[idx]), flag=0))
                time_cost_edges += time.time() - start
                start = time.time()
                # Cross_node.append(Cross_nodes(self.timesteps, self.nodes_list, self.workloads_GCN[m], P_snapshot[idx]))
                time_cost_nodes+=  time.time() - start
            # print(Load, Cross_edge, Cross_node)

            # Cross_edge = [ce*self.args['beta'] for ce in Cross_edge]
            # Cross_node = [cn*self.args['beta'] for cn in Cross_node]
            # print()
            # result = np.sum([Load,Cross_node],axis=0).tolist()
            # result = np.sum([result,Cross_edge],axis=0).tolist()
            if snapshot_per_device != 0:
                select_m = idx // snapshot_per_device
            else:
                select_m = 0
            if select_m >= self.num_devices:
                select_m = self.num_devices - 1
            # select_m = result.index(max(result))
            # select_m = Load.index(max(Load))

            Node_start_idx = self.nodes_list[P_id[idx]].size(0) - P_workload[idx]
            workload = torch.full_like(P_snapshot[idx], True, dtype=torch.bool)
            self.workloads_GCN[select_m][P_id[idx]][P_snapshot[idx]] = workload
            self.workloads_ATT[select_m][P_id[idx]][P_snapshot[idx]] = workload
            Current_workload[select_m] = Current_workload[select_m]+P_workload[idx]
            Current_RNN_workload[select_m][P_id[idx]] += 1

    def communication_time(self, GCN_node_size, RNN_node_size, bandwidth):
        '''
        Both GCN communication time and ATT communication time are needed
        '''
        distribution = Distribution(self.nodes_list, self.timesteps, self.num_devices, self.workloads_GCN)
        print('ours partition distribution (no balance): ',distribution)

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
        RNN_comm_time = [max(RNN_receive_comm_time[i], RNN_send_comm_time[i]) for i in range(len(RNN_receive_comm_time))]
        GPU_total_time = [GCN_comm_time[i] + RNN_comm_time[i] for i in range(len(GCN_comm_time))]
        # Total_time = max(GPU_total_time)
        Comp_time = Computation_time(self.num_devices, len(self.nodes_list), self.workloads_GCN, self.workloads_ATT)

        print('----------------------------------------------------------')
        print('Our partition method:')
        print('GCN | Each GPU receives nodes: {} | Each GPU sends nodes: {}'.format(GCN_receive, GCN_send))
        print('ATT | Each GPU receives nodes: {} | Each GPU sends nodes: {}'.format(RNN_receive, RNN_send))
        print('Each GPU with communication time: {} ( GCN: {} | ATT: {})'.format(GPU_total_time, GCN_comm_time, RNN_comm_time))
        print('Total time: {} | Computation time: {}, Communication time: {}'.format(max(GPU_total_time) + Comp_time, Comp_time, max(GPU_total_time)))

class Ours_balance():
    def __init__(self, args, graphs, nodes_list, adjs_list, num_devices):
        super(Ours_balance, self).__init__()
        self.args = args
        self.nodes_list = nodes_list
        self.adjs_list = adjs_list
        self.num_devices = num_devices
        self.graphs = graphs
        self.timesteps = len(nodes_list)
        self.workloads_GCN = [[torch.full_like(self.nodes_list[time], False, dtype=torch.bool) for time in range(self.timesteps)] for i in range(num_devices)]
        self.workloads_ATT = [[torch.full_like(self.nodes_list[time], False, dtype=torch.bool) for time in range(self.timesteps)] for i in range(num_devices)]

        self.Degrees = [list(dict(nx.degree(self.graphs[t])).values()) for t in range(self.timesteps)]

        # parameters
        self.alpha = args['alpha']
        # self.alpha = 0.08
        # self.alpha = 0.01
        # self.alpha = 0.1

        # runtime
        start = time.time()
        P_id, Q_id, Q_node_id, P_workload, P_snapshot, Q_workload = self.divide()
        # print('divide time cost: ', time.time() - start)
        # print('P_id: ',P_id)
        # print('Q_id: ',Q_id)
        # print('Q_node_id: ',Q_node_id)
        # print('P_workload: ',P_workload)
        # print('Q_workload: ',Q_workload)
        # print('P_snapshot: ',P_snapshot)
        start = time.time()
        self.conquer(P_id, Q_id, Q_node_id, P_workload, P_snapshot, Q_workload)
        # print('conquer time cost: ', time.time() - start)

    def divide(self):
        '''
        Step 1: compute the average degree of each snapshots
        Step 2: divide nodes into different job set according to the degree and time-series length
        '''
        Total_workload = [torch.full_like(self.nodes_list[time], 1) for time in range(self.timesteps)]
        Total_workload_temp = [torch.full_like(self.nodes_list[time], 1) for time in range(self.timesteps)]
        num_generations = [[j for j in range(i+1)] for i in range(self.timesteps)]
        P_id = [] # save the snapshot id
        Q_id = []
        Q_node_id = []
        P_workload = [] # save the workload size
        P_snapshot = []
        Q_workload = []
        Degree = []
        for t in range(self.timesteps):
            for generation in num_generations[t]:
                # compute average degree of nodes in specific generation
                if generation == 0:
                    start = 0
                else:
                    start = self.nodes_list[generation - 1].size(0)
                end = self.nodes_list[generation].size(0)
                Degree_list = list(dict(nx.degree(self.graphs[t])).values())[start:end]
                avg_deg = np.mean(Degree_list)
                Degree.append(avg_deg)
                # print('alpha: ',self.alpha)
                # print('generation; ',generation)
                workload = self.nodes_list[t][start:end]
                if avg_deg > self.alpha*(self.timesteps - t): # GCN-sensitive job
                    P_id.append(t)
                    P_workload.append(workload.size(0))
                    P_snapshot.append(workload)
                else:
                    for node in workload.tolist():
                        Q_id.append(t)
                        # divided_nodes = self.nodes_list[time].size(0) - workload.size(0)
                        Q_node_id.append(node)
                        Q_workload.append(self.timesteps - t)
                    # update following snapshots
                    for k in range(self.timesteps)[t+1:]:
                        mask = torch.full_like(Total_workload[k], True, dtype=torch.bool)
                        mask[start:end] = torch.zeros(mask[start:end].size(0), dtype=torch.bool)
                        where = torch.nonzero(mask == True, as_tuple=False).view(-1)
                        Total_workload[k] = Total_workload[k][where]
                        num_generations[k] = num_generations[k][1:]


            # # compute average degree of the graphs
            # Degree_list = list(dict(nx.degree(self.graphs[time])).values())
            # avg_deg = np.mean(Degree_list)
            # Degree.append(avg_deg)

            # if avg_deg > self.alpha*(self.timesteps - time): # GCN-sensitive job
            #     P_id.append(time)
            #     P_workload.append(Total_workload[time].size(0))
            # else:                                            # ATT-sensitive job
            #     for node in range(Total_workload[time].size(0)):
            #         Q_id.append(time)
            #         divided_nodes = self.nodes_list[time].size(0) - Total_workload[time].size(0)
            #         Q_node_id.append(node + divided_nodes)
            #         Q_workload.append(self.timesteps - time)
            #     # update following snapshots
            #     for k in range(self.timesteps)[time+1:]:
            #         update_size = Total_workload[time].size(0)
            #         Total_workload[k] = Total_workload[k][update_size:]

        return P_id, Q_id, Q_node_id, P_workload, P_snapshot, Q_workload
    
    def conquer(self, P_id, Q_id, Q_node_id, P_workload, P_snapshot, Q_workload):
        '''
        Schedule snapshot-level jobs first or schedule timeseries-level jobs first?
        '''
        Scheduled_workload = [torch.full_like(self.nodes_list[t], False, dtype=torch.bool) for t in range(self.timesteps)]
        Current_workload = [0 for i in range(self.num_devices)]
        Current_RNN_workload = [[0 for i in range(self.timesteps)]for m in range(self.num_devices)]
        # compute the average workload
        avg_workload = (sum(P_workload) + sum(Q_workload))/self.num_devices
        RNN_avg_workload = np.sum(Q_workload)/self.num_devices


        time_cost = 0
        for idx in range(len(Q_id)):
            Load = []
            Cross_edge = []
            for m in range(self.num_devices):
                Load.append(1 - float((Current_workload[m] + Q_workload[idx])/avg_workload))
                start = time.time()
                # Cross_edge.append(Cross_edges(self.timesteps, self.adjs_list, self.nodes_list, self.Degrees, self.workloads_GCN[m], (Q_id[idx], Q_node_id[idx]), flag=1))
                time_cost += time.time() - start
            # Cross_edge = [ce*self.args['beta'] for ce in Cross_edge]
            # result = np.sum([Load, Cross_edge], axis = 0).tolist()
            # select_m = result.index(max(result))
            select_m = Load.index(max(Load))
            # for m in range(self.num_devices):
            #     if m == select_m:
            for t in range(self.timesteps)[Q_id[idx]:]:
                # print(self.workloads_GCN[m][time])
                self.workloads_GCN[select_m][t][Q_node_id[idx]] = torch.ones(1, dtype=torch.bool)
                self.workloads_ATT[select_m][t][Q_node_id[idx]] = torch.ones(1, dtype=torch.bool)
                # Scheduled_workload[time][Q_node_id[idx]] = torch.ones(1, dtype=torch.bool)
            Current_workload[select_m] = Current_workload[select_m] + Q_workload[idx]
        # print('compute graph-graph cross edges time costs: ', time_cost_edges)
        # print('compute cross nodes time costs: ', time_cost_nodes)

        # print('compute node-graph cross edges time costs: ', time_cost)
        # print('GCN workload after scheduling timeseries-level jobs: ', self.workloads_GCN)


        time_cost_edges = 0
        time_cost_nodes = 0
        for idx in range(len(P_id)): # schedule snapshot-level job
            Load = []
            Cross_edge = []
            Cross_node = []
            for m in range(self.num_devices):
                Load.append(1 - float((Current_workload[m]+P_workload[idx])/avg_workload))
                # Cross_edge.append(Current_RNN_workload[m][P_id[idx]])
                start = time.time()
                Cross_edge.append(Cross_edges(self.timesteps, self.adjs_list, self.nodes_list, self.Degrees, self.workloads_GCN[m], (P_id[idx],P_snapshot[idx]), flag=0))
                time_cost_edges += time.time() - start
                start = time.time()
                Cross_node.append(Cross_nodes(self.timesteps, self.nodes_list, self.workloads_GCN[m], P_snapshot[idx]))
                time_cost_nodes+=  time.time() - start
            # print(Load, Cross_edge, Cross_node)

            # Cross_edge = [ce*self.args['beta'] for ce in Cross_edge]
            # Cross_node = [cn*self.args['beta'] for cn in Cross_node]
            # print()
            result = np.sum([Load,Cross_node],axis=0).tolist()
            result = np.sum([result,Cross_edge],axis=0).tolist()

            # select_m = result.index(max(result))
            select_m = Load.index(max(Load))

            Node_start_idx = self.nodes_list[P_id[idx]].size(0) - P_workload[idx]
            workload = torch.full_like(P_snapshot[idx], True, dtype=torch.bool)
            self.workloads_GCN[select_m][P_id[idx]][P_snapshot[idx]] = workload
            self.workloads_ATT[select_m][P_id[idx]][P_snapshot[idx]] = workload
            Current_workload[select_m] = Current_workload[select_m]+P_workload[idx]
            Current_RNN_workload[select_m][P_id[idx]] += 1
            # print('GCN workload after scheduling snapshot-level jobs: ', self.workloads_GCN)

    def communication_time(self, GCN_node_size, RNN_node_size, bandwidth):
        '''
        Both GCN communication time and ATT communication time are needed
        '''

        # distribution = Distribution(self.nodes_list, self.timesteps, self.num_devices, self.workloads_GCN)
        # print('ours partition distribution: ',distribution)

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
        ATT_comm_time = [max(RNN_receive_comm_time[i], RNN_send_comm_time[i]) for i in range(len(RNN_receive_comm_time))]
        GPU_total_time = [GCN_comm_time[i] + ATT_comm_time[i] for i in range(len(GCN_comm_time))]
        # Total_time = max(GPU_total_time)
        GCN_comp_time, ATT_comp_time = Computation_time(self.num_devices, len(self.nodes_list), self.workloads_GCN, self.workloads_ATT)

        print('----------------------------------------------------------')
        print('Hybrid-clustering + LDG:')
        print('GCN | Each GPU receives nodes: {} | Each GPU sends nodes: {}'.format(GCN_receive, GCN_send))
        print('ATT | Each GPU receives nodes: {} | Each GPU sends nodes: {}'.format(RNN_receive, RNN_send))
        print('Each GPU with computation time: ( GCN: {} | ATT: {})'.format(GCN_comp_time, ATT_comp_time))
        print('Each GPU with communication time: {} ( GCN: {} | ATT: {})'.format(GPU_total_time, GCN_comm_time, ATT_comm_time))
        print('Total time: {} | Computation time: {}, Communication time: {}'.format(max(GPU_total_time) + max(GCN_comp_time) + max(ATT_comp_time), max(GCN_comp_time) + max(ATT_comp_time), max(GPU_total_time)))

class heuristic_LDG():
    def __init__(self, args, graphs, nodes_list, adjs_list, num_devices, GCN_node_size, ATT_node_size, bandwidth):
        super(heuristic_LDG, self).__init__()
        self.args = args
        self.nodes_list = nodes_list
        self.adjs_list = adjs_list
        self.num_devices = num_devices
        self.graphs = graphs
        self.timesteps = len(nodes_list)
        self.workloads_GCN = [[torch.full_like(self.nodes_list[time], False, dtype=torch.bool) for time in range(self.timesteps)] for i in range(num_devices)]
        self.workloads_ATT = [[torch.full_like(self.nodes_list[time], False, dtype=torch.bool) for time in range(self.timesteps)] for i in range(num_devices)]

        self.Degrees = [list(dict(nx.degree(self.graphs[t])).values()) for t in range(self.timesteps)]

        # hyper-parameter
        self.k1 = 0.001
        self.k2 = 0.001
        self.alpha = 0.01
        self.beta = 0.01

        self.clustering()
        self.partition(GCN_node_size, ATT_node_size, bandwidth)
        # print(self.workloads_GCN)

    def clustering(self):
        self.cluster_timestep_list = []
        self.cluster_node_list = []

        for t in range(self.timesteps):
            generation = t + 1
            for gen in range(generation):
                self.cluster_timestep_list.append(t)
                if gen > 0:
                    self.cluster_node_list.append(self.nodes_list[t][self.nodes_list[gen - 1].size(0) : self.nodes_list[gen].size(0)])
                else:
                    self.cluster_node_list.append(self.nodes_list[t][:self.nodes_list[gen].size(0)])

    def partition(self, GCN_node_size, ATT_node_size, bandwidth):
        GCN_avg_workload = sum([self.nodes_list[t].size(0) for t in range(self.timesteps)])/self.num_devices
        ATT_avg_workload = self.nodes_list[-1].size(0)/self.num_devices
        for idx in range(len(self.cluster_timestep_list)):
            timestep = self.cluster_timestep_list[idx]
            nodes_idx = self.cluster_node_list[idx]
            scores = []
            for m in range(self.num_devices):
                current_workload_GCN = [self.workloads_GCN[m][t].clone() for t in range(self.timesteps)]
                current_workload_ATT = [self.workloads_ATT[m][t].clone() for t in range(self.timesteps)]
                current_workload_GCN[timestep][nodes_idx] = torch.ones(nodes_idx.size(0), dtype=torch.bool)
                current_workload_ATT[timestep][nodes_idx] = torch.ones(nodes_idx.size(0), dtype=torch.bool)
                # compute workload
                GCN_workload = 0
                ATT_workload = 0
                already_compute_ATT_nodes_mask = torch.zeros(self.nodes_list[-1].size(0), dtype=torch.bool)
                for t in range(self.timesteps):
                    compute_nodes = torch.nonzero(current_workload_GCN[t] == True, as_tuple=False).view(-1)
                    GCN_num_node = compute_nodes.size(0)
                    GCN_workload += GCN_num_node   # GCN workload
                    ATT_compute_node = torch.nonzero(already_compute_ATT_nodes_mask[compute_nodes] == False, as_tuple=False).view(-1)
                    ATT_num_node = ATT_compute_node.size(0)
                    ATT_workload += ATT_num_node   # ATT workload
                    already_compute_ATT_nodes_mask[compute_nodes] = torch.ones(compute_nodes.size(0), dtype=torch.bool)
                
                GCN_communication = 0
                ATT_communication = 0
                # compute communication
                for t in range(timestep + 1):
                    if t == timestep:
                        # compute GCN communication
                        # adj = self.adjs_list[t].clone()
                        # edge_source = adj._indices()[0]
                        # edge_target = adj._indices()[1]
                        # has_edge_mask = torch.zeros(self.nodes_list[t].size(0), dtype=torch.bool)
                        # has_edge_mask[edge_source] = torch.ones(edge_source.size(0), dtype = torch.bool)
                        # GCN_communication_node = torch.nonzero(has_edge_mask[nodes_idx] == True, as_tuple=False).view(-1)
                        # GCN_communication += GCN_communication_node.size(0)

                        adj = self.adjs_list[t].clone()
                        edge_source = adj._indices()[0]
                        edge_target = adj._indices()[1]

                        workload_mask = torch.zeros(self.nodes_list[t].size(0), dtype=torch.bool)
                        workload_mask[nodes_idx] = torch.ones(nodes_idx.size(0), dtype=torch.bool)
                        has_edge_mask = workload_mask[edge_source]
                        workload_edge_idx = torch.nonzero(has_edge_mask == True, as_tuple=False).view(-1)

                        local_nodes = torch.nonzero(self.workloads_GCN[m][t] == True, as_tuple=False).view(-1)
                        local_mask = torch.zeros(self.nodes_list[t].size(0), dtype=torch.bool)
                        local_mask[local_nodes] = torch.ones(local_nodes.size(0), dtype=torch.bool)
                        edge_target = edge_target[workload_edge_idx]
                        cross_edges = torch.nonzero(local_mask[edge_target] == True).view(-1)
                        GCN_communication += cross_edges.size(0)
                    else:
                        all_nodes = torch.zeros(self.nodes_list[-1].size(0), dtype=torch.bool)
                        local_nodes = torch.nonzero(self.workloads_GCN[m][t] == True, as_tuple=False).view(-1)
                        all_nodes[local_nodes] = torch.ones(local_nodes.size(0), dtype=torch.bool)
                        local_have = torch.nonzero(all_nodes[nodes_idx] == True, as_tuple=False).view(-1)
                        # all_nodes[nodes_list] = 
                        # ATT_communication_node = torch.nonzero(current_workload_ATT[t][nodes_idx] == True, as_tuple=False).view(-1)
                        ATT_communication += local_have.size(0)
                # score = (1 - GCN_workload/GCN_avg_workload) + (1 - ATT_workload/ATT_avg_workload) + GCN_communication + ATT_communication
                # score = GCN_workload/10000 + ATT_workload/10000 - (GCN_communication*GCN_node_size)/bandwidth + 0.0001*(ATT_communication*ATT_node_size)/bandwidth
                # score = 0 - (GCN_workload + ATT_workload)  + (GCN_communication + ATT_communication)
                # score = 0 - (GCN_workload/10000 + ATT_workload/10000)  + (GCN_communication*GCN_node_size/bandwidth + ATT_communication*ATT_node_size/bandwidth)
                # score = ((1 - GCN_workload/GCN_avg_workload) + (1 - ATT_workload/ATT_avg_workload)) * (GCN_communication + ATT_communication)
                score = ((GCN_avg_workload - GCN_workload) + (ATT_avg_workload - ATT_workload)) * (0.000005*GCN_communication + 0.0000001*ATT_communication + 1)
                # score = ((GCN_avg_workload - GCN_workload) + (ATT_avg_workload - ATT_workload)) * (0.00005*GCN_communication + 1)
                # score = ((GCN_avg_workload - GCN_workload)/GCN_workload) + (0.000005*GCN_communication )
                # print('GCN computation nodes: ', GCN_workload)
                # print('ATT computation nodes: ', ATT_workload)
                # print('GCN communication nodes (within snapshot): ', GCN_communication)
                # print('ATT communication nodes (across snapshots): ', ATT_communication)
                # print('device: {}, Score: {}'.format(m, score))
                scores.append(score)
                # scores.append(GCN_workload/100000 + ATT_workload/100000 - (GCN_communication*GCN_node_size)/bandwidth - (ATT_communication*ATT_node_size)/bandwidth)
            select_m = scores.index(max(scores))
            # print('selected device :', select_m)
            self.workloads_GCN[select_m][timestep][nodes_idx] = torch.ones(nodes_idx.size(0), dtype=torch.bool)
            self.workloads_ATT[select_m][timestep][nodes_idx] = torch.ones(nodes_idx.size(0), dtype=torch.bool)

    def communication_time(self, GCN_node_size, ATT_node_size, bandwidth):
        '''
        Both GCN communication time and ATT communication time are needed
        '''

        # distribution = Distribution(self.nodes_list, self.timesteps, self.num_devices, self.workloads_GCN)
        # print('ours partition distribution: ',distribution)

        ATT_receive_list, ATT_send_list = RNN_comm_nodes_new(self.nodes_list, self.num_devices, self.workloads_GCN, self.workloads_ATT)

        GCN_receive_list, GCN_send_list = GCN_comm_nodes(self.nodes_list, self.adjs_list, self.num_devices, self.workloads_GCN)
        # ATT_receive_list, ATT_send_list = ATT_comm_nodes(self.nodes_list, self.num_devices, self.workloads_GCN, self.workloads_ATT)

        GCN_receive_comm_time, GCN_send_comm_time = Comm_time(self.num_devices, GCN_receive_list, GCN_send_list, GCN_node_size, bandwidth)
        ATT_receive_comm_time, ATT_send_comm_time = Comm_time(self.num_devices, ATT_receive_list, ATT_send_list, ATT_node_size, bandwidth)

        GCN_receive = [torch.cat(GCN_receive_list[i], 0).size(0) for i in range(self.num_devices)]
        GCN_send = [torch.cat(GCN_send_list[i], 0).size(0) for i in range(self.num_devices)]
        ATT_receive = [torch.cat(ATT_receive_list[i], 0).size(0) for i in range(self.num_devices)]
        ATT_send = [torch.cat(ATT_send_list[i], 0).size(0) for i in range(self.num_devices)]

        GCN_comm_time = [max(GCN_receive_comm_time[i], GCN_send_comm_time[i]) for i in range(len(GCN_receive_comm_time))]
        ATT_comm_time = [max(ATT_receive_comm_time[i], ATT_send_comm_time[i]) for i in range(len(ATT_receive_comm_time))]
        GPU_total_time = [GCN_comm_time[i] + ATT_comm_time[i] for i in range(len(GCN_comm_time))]
        # Total_time = max(GPU_total_time)
        GCN_comp_time, ATT_comp_time = Computation_time(self.num_devices, len(self.nodes_list), self.workloads_GCN, self.workloads_ATT)

        print('----------------------------------------------------------')
        print('Generation-clustering + LDG:')
        print('GCN | Each GPU receives nodes: {} | Each GPU sends nodes: {}'.format(GCN_receive, GCN_send))
        print('ATT | Each GPU receives nodes: {} | Each GPU sends nodes: {}'.format(ATT_receive, ATT_send))
        print('Each GPU with computation time: ( GCN: {} | ATT: {})'.format(GCN_comp_time, ATT_comp_time))
        print('Each GPU with communication time: {} ( GCN: {} | ATT: {})'.format(GPU_total_time, GCN_comm_time, ATT_comm_time))
        print('Total time: {} | Computation time: {}, Communication time: {}'.format(max(GPU_total_time) + max(GCN_comp_time) + max(ATT_comp_time), max(GCN_comp_time) + max(ATT_comp_time), max(GPU_total_time)))

class Algorithm_heuristic():
    def __init__(self, args, graphs, nodes_list, adjs_list, num_devices, gcn_node_size, ATT_node_size, bandwidth, GCN_comp_scale, ATT_comp_scale):
        super(Algorithm_heuristic, self).__init__()
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

        start = time.time()
        self.cluster_id_list, self.cluster_time_list = self.clustering()
        # print('clustering time cost: ', time.time() - start)
        # print(self.cluster_id_list, self.cluster_time_list)
        start = time.time()
        # self.partitioning()
        # print('partitioning time cost: ', time.time() - start)
        
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

    def clustering(self):
        num_generations = [[j for j in range(i+1)] for i in range(self.timesteps)]
        vertex_cluster_hash = torch.torch.full_like(self.nodes_list[-1], -1)
        cluster_id_list = []
        cluster_time_list = []
        num_cluster = 0
        for t in range(self.timesteps)[::-1]:
            for generation in num_generations[t][::-1]:
                if generation == 0:
                    start = 0
                else:
                    start = self.nodes_list[generation - 1].size(0)
                end = self.nodes_list[generation].size(0)
                Degree_list = self.Degrees[t][start:end]
                avg_deg = np.mean(Degree_list)
                gen_vertices = self.nodes_list[t][start:end]
                # determin group or not
                # if 50*avg_deg*self.gcn_node_size > self.ATT_node_size*(self.timesteps - t):  # group
                if  avg_deg*self.gcn_node_size >= self.ATT_node_size*(self.timesteps - t) or num_cluster >= 1000:  # group
                    cluster_id_list.append(gen_vertices)
                    cluster_time_list.append([t])
                    num_cluster += 1
                else:  # divide
                    cluster_idx = vertex_cluster_hash[gen_vertices]
                    for id in range(cluster_idx.size(0)):
                        if cluster_idx[id] == -1: # no cluster
                            cluster_id_list.append(gen_vertices[id].view(-1))
                            cluster_time_list.append([t])
                            vertex_cluster_hash[gen_vertices[id]] = len(cluster_id_list) - 1
                        else: # already has cluster
                            cluster_time_list[cluster_idx[id]].append(t)
                        num_cluster += 1
        return cluster_id_list, cluster_time_list
    
    def partitioning(self, alg):
        self.alg = alg
        GCN_avg_workload = sum([self.nodes_list[t].size(0) for t in range(self.timesteps)])/self.num_devices
        ATT_avg_workload = self.nodes_list[-1].size(0)/self.num_devices
        compute_GNN_comm_time = 0
        compute_ATT_comm_time = 0
        compute_workload_balance_time = 0
        partiton_time = 0

        for idx in range(len(self.cluster_time_list)):
            workload = self.cluster_id_list[idx]
            scores = []
            for m in range(self.num_devices):
                GCN_communication = 0
                ATT_communication = 0
                GCN_workload = 0
                ATT_workload = 0
                # matric 1: GNN communication
                start = time.time()
                t_temp = self.cluster_time_list[idx][0]
                adj = self.adjs_list[t_temp].clone()
                edge_source = adj._indices()[0]
                edge_target = adj._indices()[1]
                workload_mask = torch.zeros(self.nodes_list[t_temp].size(0), dtype=torch.bool)
                workload_mask[workload] = torch.ones(workload.size(0), dtype=torch.bool)
                has_edge_mask = workload_mask[edge_source]
                workload_edge_idx = torch.nonzero(has_edge_mask == True, as_tuple=False).view(-1)
                edge_target = edge_target[workload_edge_idx]
                cross_edges = torch.nonzero(self.workloads_GCN[m][t_temp][edge_target] == True).view(-1)
                GCN_communication += cross_edges.size(0)*len(self.cluster_time_list[idx])
                compute_GNN_comm_time += time.time() - start

                # matric 2: ATT communication
                start = time.time()
                # all_node = [torch.zeros(self.nodes_list[-1].size(0), dtype=torch.bool) for t in range(self.timesteps)]
                # for t in range(self.timesteps):
                #     all_node[t][self.nodes_list[t]] = self.workloads_GCN[m][t]
                workload_node = torch.cat([self.workloads_GCN[m][t][workload] for t in range(self.timesteps)], dim = 0)
                # workload_node = torch.cat(self.workloads_GCN[m][:][workload], dim = 0)
                local_workload_node = torch.nonzero(workload_node == True, as_tuple=False).view(-1)
                ATT_communication = ATT_communication + local_workload_node.size(0) + len(self.cluster_time_list[idx])
                compute_ATT_comm_time += time.time() - start

                # matric 3 & 4: GNN and ATT computation load
                start = time.time()
                # compute_nodes = torch.cat([torch.nonzero(self.workloads_GCN[m][t] == True, as_tuple=False).view(-1) for t in range (self.timesteps)], dim=0)
                compute_nodes = torch.nonzero(torch.cat(self.workloads_GCN[m], dim=0) == True, as_tuple=False).view(-1)
                # matric 3 & 4: GNN and ATT computation load
                num_node = compute_nodes.size(0)
                GCN_workload += num_node   # GCN workload
                ATT_workload += num_node
                compute_workload_balance_time += time.time() - start
                # print('worker {} has local {} nodes'.format(m, num_node))
                start = time.time()
                
                if alg == 'LDG_base':
                    total_vertices = GCN_workload + ATT_workload
                    inter_edges = GCN_communication + ATT_communication
                    score = inter_edges*(1-total_vertices/(GCN_avg_workload + ATT_avg_workload))
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
                    # inter_communication = (GCN_communication*self.gcn_node_size + ATT_communication*self.ATT_node_size)/self.bandwidth
                    inter_communication = GCN_communication + ATT_communication
                    score = inter_communication*(1-(gcn_cost+att_cost)/self.total_computation_cost)
                elif alg == 'Fennel_base':
                    alpha = 3
                    beta = 1.5
                    total_vertices = GCN_workload + ATT_workload
                    inter_edges = GCN_communication + ATT_communication
                    score = inter_edges - alpha*beta*pow(total_vertices, (beta - 1))
                elif alg == 'Fennel_DyG':
                    alpha = 3
                    beta = 1.5
                    total_comp_cost = GCN_workload*self.GCN_comp_scale + ATT_workload*self.ATT_comp_scale
                    inter_communication = (GCN_communication*self.gcn_node_size + ATT_communication*self.ATT_node_size)/self.bandwidth
                    score = 10*inter_communication - alpha*beta*pow(total_comp_cost, (beta - 1))

                scores.append(score)
            select_m = scores.index(max(scores))
            for t in self.cluster_time_list[idx]:
                self.workloads_GCN[select_m][t][workload] = torch.ones(workload.size(0), dtype=torch.bool)
                self.workloads_ATT[select_m][t][workload] = torch.ones(workload.size(0), dtype=torch.bool)
            partiton_time += time.time() - start

        # print('compute GNN communication cost: {:.3f}, compute ATT communication cost: {:.3f}, compute workload balance: {:.3f}, partitioning cost: {:.3f}'.format(compute_GNN_comm_time,
                                                                                                                                                                # compute_ATT_comm_time,
                                                                                                                                                                # compute_workload_balance_time,
                                                                                                                                                                # partiton_time))

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
        print('heuristic + {}:'.format(self.alg))
        print('Number of clusters: ', len(self.cluster_id_list))
        print('GCN | Each GPU receives nodes: {} | Each GPU sends nodes: {}'.format(GCN_receive, GCN_send))
        print('ATT | Each GPU receives nodes: {} | Each GPU sends nodes: {}'.format(ATT_receive, ATT_send))
        print('Each GPU with computation time: {} ( GCN: {} | ATT: {})'.format(total_comp_time, GCN_comp_time, ATT_comp_time))
        print('Each GPU with communication time: {} ( GCN: {} | ATT: {})'.format(GPU_total_time, GCN_comm_time, ATT_comm_time))
        print('Total time: {} | Computation time: {}, Communication time: {}'.format(max(GPU_total_time) + max(GCN_comp_time) + max(ATT_comp_time), max(GCN_comp_time) + max(ATT_comp_time), max(GPU_total_time)))
        print('Total costs: {} | Computation costs: {}, Communication costs: {}'.format(np.sum(GPU_total_time) + np.sum(GCN_comp_time) + np.sum(ATT_comp_time), np.sum(GCN_comp_time) + np.sum(ATT_comp_time), np.sum(GPU_total_time)))

class Partition_DyG():
    def __init__(self, args, graphs, nodes_list, adjs_list, num_devices, gcn_node_size, ATT_node_size, bandwidth):
        super(Partition_DyG, self).__init__()
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

        # self.GCN_comp_scale = GCN_comp_scale
        # self.ATT_comp_scale = ATT_comp_scale

        self.device = torch.device("cuda")
        self.model_str = MLP_Predictor(in_feature = 2)
        self.model_str.load_state_dict(torch.load('./MLDP/Model_evaluation/model/str_{}.pt'.format(10)))
        self.model_str = self.model_str.to(self.device)

        self.model_tem = MLP_Predictor(in_feature = 2)
        self.model_tem.load_state_dict(torch.load('./MLDP/Model_evaluation/model/tem_{}.pt'.format(10)))
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
        for node in tqdm(self.coarsened_graph.nodes(), desc='Partitioning', leave=True):
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
        # # plot
        # # step 1: remove temporal links
        # temporal_edges = []
        # for edge in self.coarsened_graph.edges():
        #     if self.coarsened_graph.edges[edge]["type"] == 'tem':
        #         temporal_edges.append(edge)
        # self.coarsened_graph.remove_edges_from(temporal_edges)

        # # step 2: define node position
        # pos_list = [(random.uniform(0, 2), random.uniform(0, 2)) for _ in self.coarsened_graph.nodes()]
        # # for node in coarsened_graph.nodes():
        # # original_nodes_pos = {node: pos_list[node_mask[node].item()] for node in full_graph.nodes()}
        # original_nodes_pos = {}
        # for node in self.coarsened_graph.nodes():
        #     pos_tem = list(pos_list[self.coarsened_graph.nodes[node]['orig_id']])
        #     pos_tem[0] += (self.coarsened_graph.nodes[node]['snap_id'][0]*3)
        #     original_nodes_pos[node] = tuple(pos_tem)

        # # step 3: degine node colors
        # colorArr = ['1','2','3','4','5','6','7','8','9','A','B','C','D','E','F']
        # color_list = []
        # print([self.coarsened_graph.nodes[node]['partition'] for node in self.coarsened_graph.nodes()])
        # for num_color in range(self.num_devices):
        #     color = ""
        #     for i in range(6):
        #         color += colorArr[random.randint(0,14)]
        #     color_list.append("#"+color)
        # node_color_mask = {node: self.coarsened_graph.nodes[node]['partition'] for node in self.coarsened_graph.nodes()}
        # node_color = [color_list[node_color_mask[node]] for node in self.coarsened_graph.nodes()]

        # plot_graph(args, self.coarsened_graph, node_color, 'black', original_nodes_pos, 'partition')

    def get_partition(self):
        self.partitioning('LDG_base')
        print('----------------------------------------------------------')
        print('Original graph size V: {}, E: {} -> coarsened graph size V: {}, E: {}'.format(self.full_graph.number_of_nodes(), self.full_graph.number_of_edges(),
                                                                                                self.coarsened_graph.number_of_nodes(), self.coarsened_graph.number_of_edges()))
        return self.workloads_GCN, self.workloads_ATT

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
        print('heuristic + {}:'.format(self.alg))
        print('Original graph size V: {}, E: {} -> coarsened graph size V: {}, E: {}'.format(self.full_graph.number_of_nodes(), self.full_graph.number_of_edges(),
                                                                                                self.coarsened_graph.number_of_nodes(), self.coarsened_graph.number_of_edges()))
        # print('Number of clusters: ', len(self.cluster_id_list))
        print('GCN | Each GPU receives nodes: {} | Each GPU sends nodes: {}'.format(GCN_receive, GCN_send))
        print('ATT | Each GPU receives nodes: {} | Each GPU sends nodes: {}'.format(ATT_receive, ATT_send))
        print('Each GPU with computation time: {} ( GCN: {} | ATT: {})'.format(total_comp_time, GCN_comp_time, ATT_comp_time))
        print('Each GPU with communication time: {} ( GCN: {} | ATT: {})'.format(GPU_total_time, GCN_comm_time, ATT_comm_time))
        print('Total time: {} | Computation time: {}, Communication time: {}'.format(max(GPU_total_time) + max(GCN_comp_time) + max(ATT_comp_time), max(GCN_comp_time) + max(ATT_comp_time), max(GPU_total_time)))
        print('Total costs: {} | Computation costs: {}, Communication costs: {}'.format(np.sum(GPU_total_time) + np.sum(GCN_comp_time) + np.sum(ATT_comp_time), np.sum(GCN_comp_time) + np.sum(ATT_comp_time), np.sum(GPU_total_time)))


import time

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test parameters')
    # parser.add_argument('--json-path', type=str, required=True,
    #                     help='the path of hyperparameter json file')
    # parser.add_argument('--test-type', type=str, required=True, choices=['local', 'dp', 'ddp'],
    #                     help='method for DGNN training')
    
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
        Cross_edge_between_generations(nodes_list, adjs_list, time_steps)

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
        Cross_edge_between_generations(nodes_list, adjs_list, time_steps)

        # GCN_node_size = raw_feats[0].size(1)*32*
        GCN_node_size = 128*32
        RNN_node_size = 128*32
    
    GCN_comp_scale = 4*math.pow(10, -5)
    ATT_comp_scale = 4*math.pow(10, -5)

    # # computation and communication scales
    # if args['dataset'] == 'Epinion':
    #     GCN_comp_scale = 4*math.pow(10, -7)
    #     ATT_comp_scale = 4*math.pow(10, -8)
    # elif args['dataset'] == 'Copresence':
    #     GCN_comp_scale = 5*math.pow(10, -5)
    #     ATT_comp_scale = 1*math.pow(10, -5)
    # elif args['dataset'] == 'Movie_rating':
    #     GCN_comp_scale = 5*math.pow(10, -7)
    #     ATT_comp_scale = 1*math.pow(10, -7)
    # elif args['dataset'] == 'Stack_overflow':
    #     GCN_comp_scale = 3*math.pow(10, -7)
    #     ATT_comp_scale = 1*math.pow(10, -7)

    # start = 15
    # window = 5
    # nodes_list = nodes_list[start: start+window]
    # adjs_list = adjs_list[start: start+window]

    node_partition_obj = node_partition(args, graphs, nodes_list, adjs_list, args['world_size'])
    node_partition_obj.communication_time(GCN_node_size, RNN_node_size, bandwidth_10MB, GCN_comp_scale, ATT_comp_scale)

    snapshot_partition_obj = snapshot_partition(args, graphs, nodes_list, adjs_list, args['world_size'])
    snapshot_partition_obj.communication_time(GCN_node_size, RNN_node_size, bandwidth_10MB, GCN_comp_scale, ATT_comp_scale)

    # proposed_partition_obj = Ours(args, graphs, nodes_list, adjs_list, args['world_size'])
    # proposed_partition_obj.communication_time(GCN_node_size, RNN_node_size, bandwidth_10MB)

    # # # balance methods
    node_partition_balance_obj = node_partition_balance(args, graphs, nodes_list, adjs_list, args['world_size'])
    node_partition_balance_obj.communication_time(GCN_node_size, RNN_node_size, bandwidth_100MB, GCN_comp_scale, ATT_comp_scale)

    snapshot_partition_balance_obj = snapshot_partition_balance(args, graphs, nodes_list, adjs_list, args['world_size'])
    snapshot_partition_balance_obj.communication_time(GCN_node_size, RNN_node_size, bandwidth_100MB, GCN_comp_scale, ATT_comp_scale)


    heuristic_2 = Partition_DyG(args, graphs, nodes_list, adjs_list, args['world_size'], GCN_node_size, RNN_node_size, bandwidth_10MB, GCN_comp_scale, ATT_comp_scale)
    heuristic_2.partitioning('LDG_DyG')
    heuristic_2.communication_time(GCN_node_size, RNN_node_size, bandwidth_10MB, GCN_comp_scale, ATT_comp_scale)

    # heuristic_3 = Algorithm_heuristic(args, graphs, nodes_list, adjs_list, args['world_size'], GCN_node_size, RNN_node_size, bandwidth_10MB, GCN_comp_scale, ATT_comp_scale)
    # heuristic_3.partitioning('Fennel_base')
    # heuristic_3.communication_time(GCN_node_size, RNN_node_size, bandwidth_10MB, GCN_comp_scale, ATT_comp_scale)

    # heuristic_4 = Algorithm_heuristic(args, graphs, nodes_list, adjs_list, args['world_size'], GCN_node_size, RNN_node_size, bandwidth_10MB, GCN_comp_scale, ATT_comp_scale)
    # heuristic_4.partitioning('Fennel_DyG')
    # heuristic_4.communication_time(GCN_node_size, RNN_node_size, bandwidth_10MB, GCN_comp_scale, ATT_comp_scale)