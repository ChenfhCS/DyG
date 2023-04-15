from torch_geometric.data import Data
from torch_geometric.data import Batch

import networkx as nx

import numpy as np

import torch
import multiprocessing as mp
import torch.distributed.rpc as rpc
import time,os

from torch_geometric.utils import from_networkx
from Diana.distributed.kvstore import (KVStoreServer, KVStoreClient)
from Diana.distributed.utils import get_remote_neighbors, get_local_belong_remote_neighbors
import torch

# tensor_A = torch.tensor([5, 1, 8])
# tensor_B = torch.tensor([[1, 1, 5], [8, 8, 1]])

# # 获取 tensor_A 中每个元素在排好序的 tensor_A 中的位置
# sorted_A_idx = torch.argsort(tensor_A)
# tensor_A_id = torch.cat([tensor_A[sorted_A_idx].unsqueeze(0), sorted_A_idx.unsqueeze(0)], dim=0)

# # 获取 tensor_B 中每个元素在 tensor_A 中的位置
# idx_1 = torch.searchsorted(tensor_A[sorted_A_idx], tensor_B[0])
# idx_2 = torch.searchsorted(tensor_A[sorted_A_idx], tensor_B[1])

# # 使用位置更新 tensor_B
# tensor_B[0] = tensor_A_id[1][idx_1]
# tensor_B[1] = tensor_A_id[1][idx_2]

# print(tensor_B)

local_node_index_1 = torch.tensor([1,2,3])
local_node_index_2 = torch.tensor([5,6])
edge_index = torch.tensor([[4,5,6,3],[1,3,2,2]])
local_edges = edge_index[:, torch.where(torch.isin(edge_index[1], local_node_index_1))[0]]
print(local_edges)

# local_belong_neighbors = get_local_belong_remote_neighbors(local_node_index_2, edge_index)
# remote_neighbors = get_remote_neighbors(local_node_index_1, edge_index)
# print(local_belong_neighbors)
# print(remote_neighbors)

# key = torch.tensor([1,2,3])
# value = torch.tensor([4,5,6])

# dicts = {}
# dicts[torch.tensor([1])] = torch.tensor(1)
# # dicts.update(dict(zip([1],[2])))
# # dicts.update(dict(zip(key,value)))
# ten = dicts[torch.tensor([1])]

# print(dicts)  # 输出结果为 tensor([1, 2])




# from multiprocessing import Manager

# global KVSTORESEVER

# os.environ["MASTER_ADDR"] = "localhost"
# os.environ["MASTER_PORT"] = "12347"

# def _main(rank, args, shared_dict):
#     world_size = args['world_size']

#     torch.distributed.init_process_group(
#         backend="gloo",
#         rank=rank,
#         world_size=world_size,
#         init_method="env://"
#     )
#     if rank == 0:
#         # register kvstore_server
#         rpc.init_rpc(
#             name=f"server_{rank}",
#             rank=rank,
#             world_size=world_size,
#             rpc_backend_options=rpc.TensorPipeRpcBackendOptions(
#                 init_method="env://"
#             )
#         )
#         kvstore_server = KVStoreServer(rank, world_size, 'DySAT', (6,1,1))
#         shared_dict['kvstore_server'] = kvstore_server
#         print("KV store server has registered!")
#         node_id_1 = torch.tensor([0,1,2], dtype=torch.long)
#         tensor_1 = torch.tensor([1,2,3], dtype=torch.float32).unsqueeze(1)
#         node_id_2 = torch.tensor([3,4,5], dtype=torch.long)
#         tensor_2 = torch.tensor([4,5,6], dtype=torch.float32).unsqueeze(1)
#         kvstore_server.push(1, node_id_1, tensor_1)
#         kvstore_server.push(2, node_id_2, tensor_2)
#         print("KV store server has updated!")
#         while True:
#             time.sleep(1)

#     if rank == 1:
#         rpc.init_rpc(
#                 name=f"worker_{rank}",
#                 rank=rank,
#                 world_size=world_size
#             )
#         time.sleep(3)
#         kvstore_server = shared_dict['kvstore_server']
#         kvsotre_client = KVStoreClient(server_name='server_0', rank=rank, world_size=world_size, module_name=kvstore_server)
#         print(f"Rank {rank}: KV store client has registered!")
#         node_id_1 = torch.tensor([0,1,2], dtype=torch.long)
#         node_id_2 = torch.tensor([3,4,5], dtype=torch.long)
#         tensor_1 = kvsotre_client.pull(1, node_id_1)
#         tensor_2 = kvsotre_client.pull(2, node_id_2)
#         print(f"Rank {rank}: pull finished")
#         print(tensor_1.data, tensor_2.data)

# if __name__ == '__main__':
#     torch.multiprocessing.set_start_method('spawn')
#     shared_dict = mp.Manager().dict()
#     workers = []
#     world_size = 2
#     args = {}
#     args['world_size'] = 2
#     for rank in range(world_size):
#         p = mp.Process(target=_main, args=(rank, args, shared_dict))
#         p.start()
#         workers.append(p)
#     for p in workers:
#         p.join()







# nx_graph = nx.Graph()

# # 添加带有标签属性的节点
# nx_graph.add_node(0, label='A')
# nx_graph.add_node(1, label='B')
# nx_graph.add_node(2, label='A')
# nx_graph.add_node(3, label='C')
# nx_graph.add_edge(0,1)
# nx_graph.add_edge(1,2)
# nx_graph.add_edge(2,3)

# node_labels = nx.get_node_attributes(nx_graph, 'label')

# # Create label-to-nodes dictionary
# label_to_nodes = {}
# for node, label in node_labels.items():
#     if label not in label_to_nodes:
#         label_to_nodes[label] = []
#     label_to_nodes[label].append(node)

# # Loop over label-to-nodes dictionary
# pyg_graphs = []
# for label, nodes in label_to_nodes.items():
#     # Create new PyG graph object
#     pyg_graph = Data(x=[], edge_index=[], y=[])
    
#     # Add nodes to PyG graph object
#     pyg_graph.x = torch.tensor([node for node in nodes], dtype=torch.long)
    
#     # Extract subgraph induced by current list of nodes
#     subgraph = nx_graph.subgraph(nodes).to_directed()
    
#     # Convert subgraph to PyG graph
#     subgraph_pyg = from_networkx(subgraph)
    
#     # Add edges to PyG graph object
#     pyg_graph.edge_index = subgraph_pyg.edge_index
#     for edge in nx_graph.edges():
#         # check if the source and target nodes of the edge are included in the current subgraph
#         if edge[1] in nodes:
#             pyg_graph.edge_index = torch.cat([pyg_graph.edge_index, torch.tensor([[edge[0]], [edge[1]]])], dim=1)


#     # Append PyG graph to list of graphs
#     pyg_graphs.append(pyg_graph)

# for pyg_graph in pyg_graphs:
#     print('Nodes: ', pyg_graph.x)
#     print('Edges: ', pyg_graph.edge_index)