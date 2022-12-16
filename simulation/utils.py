
r"""
Utils to play with PyTorch.
"""
from torch._C._distributed_c10d import GatherOptions

import torch.distributed as dist
import torch
import numpy as np
import os

from cost_evaluator import MLP_Predictor
# import dgnn_collectives


# pylint: disable=broad-except
# pylint: disable=protected-access
def get_torch_default_comm():
    r"""
    The NCCL communicator is needed so that Fast MoE can perform customized
    communication operators in the C code. However, it is not a publicly
    available variable. Therefore, a hacking class of the `ProcessGroupNCCL`
    in Fast MoE's C code takes the `_default_pg` and tries to dig the
    communicator out from the object. As PyTorch's private interface varies from
    time to time, different hacking techniques are tried one-by-one to be
    compatible with various versions of PyTorch.
    """
    try:
        comm = dist.distributed_c10d._get_default_group()
        return comm
    except Exception as _:
        pass
    try:
        comm = dist.distributed_c10d._default_pg
        if comm is not None:
            return comm
    except Exception as _:
        pass
    raise RuntimeError("Unsupported PyTorch version")

def make_sparse_tensor(adj, tensor_type, torch_size):
    if len(torch_size) == 2:
        tensor_size = torch.Size(torch_size)
    elif len(torch_size) == 1:
        tensor_size = torch.Size(torch_size*2)

    if tensor_type == 'float':
        test = torch.sparse.FloatTensor(adj['idx'].t(),
                                      adj['vals'].type(torch.float),
                                      tensor_size)
        return torch.sparse.FloatTensor(adj['idx'].t(),
                                      adj['vals'].type(torch.float),
                                      tensor_size)
    elif tensor_type == 'long':
        return torch.sparse.LongTensor(adj['idx'].t(),
                                      adj['vals'].type(torch.long),
                                      tensor_size)
    else:
        raise NotImplementedError('only make floats or long sparse tensors')

def gather():
    output = [[torch.zeros(5) for i in range(5)]]
    input = [torch.zeros(5)]
    opts = GatherOptions()
    opts.rootRank = 1
    # try:
    #     dgnn_collectives.emb_exchange(output, input, opts)
    # except RuntimeError:
    #     print("got RuntimeError as emb_exchange is not implemented.")


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
    # print('receive list: ', receive_list)
    # print('send list: ', send_list)
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

def Computation_time(graphs, num_devices, timesteps, workload_GCN, workload_RNN, model_str, model_tem, device):
    GCN_time = [0 for i in range(num_devices)]
    RNN_time = [0 for i in range(num_devices)]
    # device = model_str.device
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
            att_nodes_list = torch.nonzero(workload_RNN[m][t] == True, as_tuple=False).view(-1).tolist()
            if len(att_nodes_list) > 0:
                full_nodes.extend(nodes_list)
                total_time_step += 1
        full_nodes = list(set(full_nodes))
        # if m == 0:
        #     print('number of full nodes: {} total timesteps: {}'.format(len(full_nodes), total_time_step))
        tem_input = torch.Tensor([float(len(full_nodes)/10000), float(total_time_step/10)]).to(device)
        cost = model_tem(tem_input)
        rnn_comp_time += cost.item()
        GCN_time[m] += round(gcn_comp_time, 3)
        RNN_time[m] += round(rnn_comp_time, 3)


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