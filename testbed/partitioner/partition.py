import numpy as np
import scipy.sparse as sp
import argparse
import torch
import networkx as nx
import matplotlib.pyplot as plt
import math
import random
import os, sys
sys.path.append("..") 

from tqdm import tqdm
from .util import graph_concat
from method import MLP_Predictor, coarsener

# Simulation setting
bandwidth_1MB = float(1024*1024*8)
bandwidth_10MB = float(10*1024*1024*8)
bandwidth_100MB = float(100*1024*1024*8)
bandwidth_GB = float(1024*1024*1024*8)

current_path = os.path.abspath(os.path.join(os.getcwd(), ".."))
path = current_path + '/method/cost_evaluator/model/'
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = 'cpu'
model_str = MLP_Predictor(in_feature = 2)
model_str.load_state_dict(torch.load(path + 'str_10.pt'))
model_str = model_str.to(device)

model_tem = MLP_Predictor(in_feature = 2)
model_tem.load_state_dict(torch.load(path + 'tem_10.pt'))
model_tem = model_tem.to(device)

model_str.eval()
model_tem.eval()

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


def plot_graph(args,G, node_color, edge_color, pos, flag):
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

    fig, ax = plt.subplots()
    nx.draw(G, ax=ax, node_size=5, 
	        width=0.5,
            pos=pos,
            node_color=node_color,
            edge_color=edge_color)
    plt.savefig('./experiment_results/{}_graph_{}.png'.format(flag, args['dataset']), dpi=300)
    plt.close()

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
class PTS():
    def __init__(self, args, graphs, nodes_list, adjs_list, num_devices, logger):
        self.args = args
        self.nodes_list = nodes_list
        self.adjs_list = adjs_list
        self.num_devices = num_devices
        self.logger = logger
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

        GCN_comm_time = [round(max(GCN_receive_comm_time[i], GCN_send_comm_time[i]), 3) for i in range(len(GCN_receive_comm_time))]
        RNN_comm_time = [round(max(RNN_receive_comm_time[i], RNN_send_comm_time[i]), 3) for i in range(len(RNN_receive_comm_time))]
        GPU_total_time = [GCN_comm_time[i] + RNN_comm_time[i] for i in range(len(GCN_comm_time))]
        # Total_time = max(GPU_total_time)
        GCN_comp_time, ATT_comp_time = Computation_time(self.graphs, self.num_devices, len(self.nodes_list), self.workloads_GCN, self.workloads_ATT, model_str, model_tem, device)
        total_comp_time = [round(GCN_comp_time[i] + ATT_comp_time[i], 3) for i in range(len(GCN_comm_time))]

        self.logger.info('PTS method:')
        self.logger.info('GCN | Each GPU receives nodes: {} | Each GPU sends nodes: {}'.format(GCN_receive, GCN_send))
        self.logger.info('ATT | Each GPU receives nodes: {} | Each GPU sends nodes: {}'.format(RNN_receive, RNN_send))
        self.logger.info('Each GPU with computation time: {} ( GCN: {} | ATT: {})'.format(total_comp_time, GCN_comp_time, ATT_comp_time))
        self.logger.info('Each GPU with communication time: {} ( GCN: {} | ATT: {})'.format(GPU_total_time, GCN_comm_time, RNN_comm_time))
        self.logger.info('Total time: {:.3f} | Computation time: {:.3f}, Communication time: {:.3f}'.format(max(GPU_total_time) + max(GCN_comp_time) + max(ATT_comp_time), max(GCN_comp_time) + max(ATT_comp_time), max(GPU_total_time)))
        self.logger.info('Total costs: {:.3f} | Computation costs: {:.3f}, Communication costs: {:.3f}'.format(np.sum(GPU_total_time) + np.sum(GCN_comp_time) + np.sum(ATT_comp_time), np.sum(GCN_comp_time) + np.sum(ATT_comp_time), np.sum(GPU_total_time)))
        self.logger.info('----------------------------------------------------------')
        return round(max(GPU_total_time) + max(GCN_comp_time) + max(ATT_comp_time), 4), round(np.sum(GPU_total_time) + np.sum(GCN_comp_time) + np.sum(ATT_comp_time), 4)

    def get_partition(self):
        return self.workloads_GCN, self.workloads_ATT


class PSS():
    def __init__(self, args, graphs, nodes_list, adjs_list, num_devices, logger = None):
        self.args = args
        self.nodes_list = nodes_list
        self.adjs_list = adjs_list
        self.num_devices = num_devices
        self.logger = logger
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

        GCN_comm_time = [round(max(GCN_receive_comm_time[i], GCN_send_comm_time[i]), 3) for i in range(len(GCN_receive_comm_time))]
        RNN_comm_time = [round(max(RNN_receive_comm_time[i], RNN_send_comm_time[i]), 3) for i in range(len(RNN_receive_comm_time))]
        GPU_total_time = [GCN_comm_time[i] + RNN_comm_time[i] for i in range(len(GCN_comm_time))]
        # Total_time = max(GPU_total_time)
        GCN_comp_time, ATT_comp_time = Computation_time(self.graphs, self.num_devices, len(self.nodes_list), self.workloads_GCN, self.workloads_ATT, model_str, model_tem, device)
        total_comp_time = [round(GCN_comp_time[i] + ATT_comp_time[i], 3) for i in range(len(GCN_comp_time))]

        self.logger.info('PSS method:')
        self.logger.info('GCN | Each GPU receives nodes: {} | Each GPU sends nodes: {}'.format(GCN_receive, GCN_send))
        self.logger.info('ATT | Each GPU receives nodes: {} | Each GPU sends nodes: {}'.format(RNN_receive, RNN_send))
        self.logger.info('Each GPU with computation time: {} ( GCN: {} | ATT: {})'.format(total_comp_time, GCN_comp_time, ATT_comp_time))
        self.logger.info('Each GPU with communication time: {} ( GCN: {} | ATT: {})'.format(GPU_total_time, GCN_comm_time, RNN_comm_time))
        self.logger.info('Total time: {:.3f} | Computation time: {:.3f}, Communication time: {:.3f}'.format(max(GPU_total_time) + max(GCN_comp_time) + max(ATT_comp_time), max(GCN_comp_time) + max(ATT_comp_time), max(GPU_total_time)))
        self.logger.info('Total costs: {:.3f} | Computation costs: {:.3f}, Communication costs: {:.3f}'.format(np.sum(GPU_total_time) + np.sum(GCN_comp_time) + np.sum(ATT_comp_time), np.sum(GCN_comp_time) + np.sum(ATT_comp_time), np.sum(GPU_total_time)))
        self.logger.info('----------------------------------------------------------')
        return round(max(GPU_total_time) + max(GCN_comp_time) + max(ATT_comp_time), 4), round(np.sum(GPU_total_time) + np.sum(GCN_comp_time) + np.sum(ATT_comp_time), 4)

    def get_partition(self):
        return self.workloads_GCN, self.workloads_ATT


class PSS_TS():
    def __init__(self, args, graphs, nodes_list, adjs_list, num_devices, logger):
        self.args = args
        self.nodes_list = nodes_list
        self.adjs_list = adjs_list
        self.num_devices = num_devices
        self.logger = logger
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

        GCN_comm_time = [round(max(GCN_receive_comm_time[i], GCN_send_comm_time[i]), 3) for i in range(len(GCN_receive_comm_time))]
        RNN_comm_time = [round(max(RNN_receive_comm_time[i], RNN_send_comm_time[i]), 3) for i in range(len(RNN_receive_comm_time))]
        GPU_total_time = [GCN_comm_time[i] + RNN_comm_time[i] for i in range(len(GCN_comm_time))]
        # Total_time = max(GPU_total_time)
        GCN_comp_time, ATT_comp_time = Computation_time(self.graphs, self.num_devices, len(self.nodes_list), self.workloads_GCN, self.workloads_ATT, model_str, model_tem, device)
        total_comp_time = [round(GCN_comp_time[i] + ATT_comp_time[i], 3) for i in range(len(GCN_comp_time))]

        self.logger.info('PSS-TS method:')
        self.logger.info('GCN | Each GPU receives nodes: {} | Each GPU sends nodes: {}'.format(GCN_receive, GCN_send))
        self.logger.info('ATT | Each GPU receives nodes: {} | Each GPU sends nodes: {}'.format(RNN_receive, RNN_send))
        self.logger.info('Each GPU with computation time: {} ( GCN: {} | ATT: {})'.format(total_comp_time, GCN_comp_time, ATT_comp_time))
        self.logger.info('Each GPU with communication time: {} ( GCN: {} | ATT: {})'.format(GPU_total_time, GCN_comm_time, RNN_comm_time))
        self.logger.info('Total time: {:.3f} | Computation time: {:.3f}, Communication time: {:.3f}'.format(max(GPU_total_time) + max(GCN_comp_time) + max(ATT_comp_time), max(GCN_comp_time) + max(ATT_comp_time), max(GPU_total_time)))
        self.logger.info('Total costs: {:.3f} | Computation costs: {:.3f}, Communication costs: {:.3f}'.format(np.sum(GPU_total_time) + np.sum(GCN_comp_time) + np.sum(ATT_comp_time), np.sum(GCN_comp_time) + np.sum(ATT_comp_time), np.sum(GPU_total_time)))
        self.logger.info('----------------------------------------------------------')
        return round(max(GPU_total_time) + max(GCN_comp_time) + max(ATT_comp_time), 4), round(np.sum(GPU_total_time) + np.sum(GCN_comp_time) + np.sum(ATT_comp_time), 4)

    def get_partition(self):
        return self.workloads_GCN, self.workloads_ATT


class Diana():
    def __init__(self, args, graphs, nodes_list, adjs_list, num_devices, gcn_node_size, ATT_node_size, bandwidth, logger):
        super(Diana, self).__init__()
        self.args = args
        self.nodes_list = nodes_list
        self.adjs_list = adjs_list
        self.num_devices = num_devices
        self.logger = logger
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

        self.device = device
        # self.model_str = model_str
        # self.model_tem = model_tem

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
            cost = model_str(input)
            gcn_cost += cost.item()
            if len(nodes_list) > 0:
                full_nodes.extend(nodes_list)
                total_time_step += 1
        full_nodes = list(set(full_nodes))
        tem_input = torch.Tensor([float(len(full_nodes)/10000), float(total_time_step/10)]).to(self.device)
        cost = model_tem(tem_input)
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
        self.coarsened_graph, self.node_to_nodes_list = coarsener(self.args, self.graphs, self.full_graph, model_str, model_tem, device)
    
    def partitioning(self, alg):
        """
        partition the coarsened graph
        """
        self.alg = alg
        num_nodes_process = 0
        # for node in self.coarsened_graph.nodes():
        for node in tqdm(self.coarsened_graph.nodes(), desc='Partitioning...', leave=False):
            nodes_to_partition = self.node_to_nodes_list[node]
            # print('node {} map to original nodes {}'.format(node, nodes_to_partition))
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
                    # score = inter_edges+(1-num_node/self.total_nodes)
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
                            cost = model_str(input)
                            gcn_cost += cost.item()
                            full_nodes.extend(nodes_list)
                            total_time_step += 1
                    full_nodes = list(set(full_nodes))
                    tem_input = torch.Tensor([float(len(full_nodes)/10000), float(total_time_step/10)]).to(self.device)
                    cost = model_tem(tem_input)
                    att_cost += cost.item()

                    # total_comp_cost = gcn_cost+att_cost
                    # total_comp_cost = GCN_workload*self.GCN_comp_scale + ATT_workload*self.ATT_comp_scale
                    inter_communication = ((GCN_communication*self.gcn_node_size + ATT_communication*self.ATT_node_size)/self.bandwidth)
                    # inter_communication = GCN_communication + ATT_communication
                    # score = inter_communication*(1-(gcn_cost+att_cost)/self.total_computation_cost)
                    score = inter_communication+(1-(gcn_cost+att_cost)/self.total_computation_cost)
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
        
        GCN_comm_time = [round(max(GCN_receive_comm_time[i], GCN_send_comm_time[i]), 3) for i in range(len(GCN_receive_comm_time))]
        ATT_comm_time = [round(max(ATT_receive_comm_time[i], ATT_send_comm_time[i]), 3) for i in range(len(ATT_receive_comm_time))]
        GPU_total_time = [GCN_comm_time[i] + ATT_comm_time[i] for i in range(len(GCN_comm_time))]
        # Total_time = max(GPU_total_time)

        start = time.time()
        GCN_comp_time, ATT_comp_time = Computation_time(self.graphs, self.num_devices, len(self.nodes_list), self.workloads_GCN, self.workloads_ATT, model_str, model_tem, device)
        Comp_time = time.time() - start
        total_comp_time = [round(GCN_comp_time[i] + ATT_comp_time[i], 3) for i in range(len(GCN_comm_time))]
        # print('ATT get node cost: {:.3f}, GNN get node cost: {:.3f}, communication cost: {:.3f}, sperate cost: {:.3f}, computation cost: {:.3f}'.format(ATT_get_node_time,
        #                                                                                                 GCN_get_node_time,
        #                                                                                                 Communication_time,
        #                                                                                                 Sperate_time,
        #                                                                                                 Comp_time))

        self.logger.info('Diana + {}:'.format(self.alg))
        self.logger.info('Original graph size V: {}, E: {} -> coarsened graph size V: {}, E: {}'.format(self.full_graph.number_of_nodes(), self.full_graph.number_of_edges(),
                                                                                                self.coarsened_graph.number_of_nodes(), self.coarsened_graph.number_of_edges()))
        self.logger.info('GCN | Each GPU receives nodes: {} | Each GPU sends nodes: {}'.format(GCN_receive, GCN_send))
        self.logger.info('ATT | Each GPU receives nodes: {} | Each GPU sends nodes: {}'.format(ATT_receive, ATT_send))
        self.logger.info('Each GPU with computation time: {} ( GCN: {} | ATT: {})'.format(total_comp_time, GCN_comp_time, ATT_comp_time))
        self.logger.info('Each GPU with communication time: {} ( GCN: {} | ATT: {})'.format(GPU_total_time, GCN_comm_time, ATT_comm_time))
        self.logger.info('Total time: {:.3f} | Computation time: {:.3f}, Communication time: {:.3f}'.format(max(GPU_total_time) + max(GCN_comp_time) + max(ATT_comp_time), max(GCN_comp_time) + max(ATT_comp_time), max(GPU_total_time)))
        self.logger.info('Total costs: {:.3f} | Computation costs: {:.3f}, Communication costs: {:.3f}'.format(np.sum(GPU_total_time) + np.sum(GCN_comp_time) + np.sum(ATT_comp_time), np.sum(GCN_comp_time) + np.sum(ATT_comp_time), np.sum(GPU_total_time)))
        self.logger.info('----------------------------------------------------------')
        return round(max(GPU_total_time) + max(GCN_comp_time) + max(ATT_comp_time), 4), round(np.sum(GPU_total_time) + np.sum(GCN_comp_time) + np.sum(ATT_comp_time), 4)

    def get_partition(self):
        return self.workloads_GCN, self.workloads_ATT
