import networkx as nx
import time

import torch

from torch_geometric.data import Data
from torch_geometric.utils import to_networkx, from_networkx
from torch_geometric.utils import add_self_loops, remove_self_loops

def graph_concat(graphs):
    """
    Function to concatenate snapshots along the temporal dimension
    :param graphs: list of snapshots
    """

    # merged_graph = None
    # for graph in graphs:
    #     pyg_graph = graph
    #     if merged_graph is None:
    #         merged_graph = pyg_graph
    #     else:
    #         # Shift the node indices of the incoming graph
    #         pyg_graph.node_index += merged_graph.node_index.size(0)
    #         pyg_graph.edge_index[0] += merged_graph.node_index.size(0)
    #         pyg_graph.edge_index[1] += merged_graph.node_index.size(0)
    #         # Concatenate the graphs
    #         merged_graph.x = torch.cat([merged_graph.x, pyg_graph.x], dim=0)
    #         merged_graph.node_index = torch.cat([merged_graph.node_index, pyg_graph.node_index])
    #         merged_graph.edge_index = torch.cat([merged_graph.edge_index, pyg_graph.edge_index], dim=1)
    
    # print(merged_graph.node_index, merged_graph.edge_index)
    # return merged_graph

    merged_graph = nx.Graph()
    # merged_graph = Data(x=[], edge_index=[])

    networkx_graphs = [to_networkx(graph) for graph in graphs]
    for i, graph in enumerate(networkx_graphs):
        snap_id = i
        node_idx = {node: {'orig_id': node} for node in list(graph.nodes())}
        nx.set_node_attributes(graph, snap_id, "snap_id")
        nx.set_node_attributes(graph, node_idx)
        attr = {edge: {"type": 'str'} for edge in graph.edges()}
        nx.set_edge_attributes(graph, attr)
        merged_graph = nx.disjoint_union(merged_graph, graph)
    for node in list(merged_graph):
        snap_id = merged_graph.nodes[node]['snap_id']
        tem_idx = 0
        # for i in range(snap_id):
        if snap_id > 0:
            tem_neighbor = node - networkx_graphs[snap_id - 1].number_of_nodes() - tem_idx
            tem_idx += networkx_graphs[snap_id - 1].number_of_nodes()
            if merged_graph.nodes[tem_neighbor]['snap_id'] == snap_id - 1:
                merged_graph.add_edge(tem_neighbor, node)
                attr = {(tem_neighbor, node): {"type": 'tem'}}
                nx.set_edge_attributes(merged_graph, attr)

    return merged_graph

def get_edges(graph):
    spatial_edges = []
    temporal_edges = []
    for edge in graph.edges():
        if 'str' not in graph.edges[edge]['type']:
            temporal_edges.append(torch.tensor(list(edge), dtype=torch.long).unsqueeze(1))
        if 'tem' not in graph.edges[edge]['type']:
            spatial_edges.append(torch.tensor(list(edge), dtype=torch.long).unsqueeze(1))
    
    spatial_edge_index = torch.cat(spatial_edges, dim=1)
    temporal_edge_index = torch.cat(temporal_edges, dim=1)
    return spatial_edge_index, temporal_edge_index

def get_neighbors(graph):
    all_neighbors = [graph.neighbors(node) for node in graph.nodes()]
    return all_neighbors

def get_pyg_full_graph(graph):
    # remove temporal edges
    temporal_edges = []
    for edge in graph.edges():
        if 'str' not in graph.edges[edge]['type']:
            temporal_edges.append(edge)
    graph.remove_edges_from(temporal_edges)

    pyg_full_graph = from_networkx(graph)
    return pyg_full_graph

def chunk_generation(graph, features, spatial_edge_index, temporal_edge_index):
    '''
    para:
        graph: a networkx graph with spatial and temporal edges ['type]
        features: a list of feature tensor, from Data.x
        spatial_edge_index: all spatial edges
        temporal_edge_index: all temporal edges
    return:
        A list of chunks, where each chunk is a Data
    '''
    feats = torch.cat(features, dim = 0)
    num_nodes = feats.size(0)

    # Create label-to-nodes dictionary
    node_labels = nx.get_node_attributes(graph, 'label')
    label_to_nodes = {}
    for node, label in node_labels.items():
        if label not in label_to_nodes:
            label_to_nodes[label] = []
        label_to_nodes[label].append(node)

    # Loop over label-to-nodes dictionary
    graph_chunks = []
    chunk_node_map = {}
    chunk_id = 0

    get_remote_neighbors_time = 0
    get_all_edges_time = 0
    remap_edges_time = 0

    for label, nodes in label_to_nodes.items():
        # Create new PyG graph object
        chunk = Data(x=[], edge_index=[])

        # Add original node index, which is used to get cross-chunk edges
        chunk.local_node_index = torch.tensor(nodes, dtype=torch.long)

        # get remote neighbors
        start_time = time.time()
        chunk.remote_spatial_neighbors =get_remote_neighbors(chunk.local_node_index, num_nodes, spatial_edge_index)
        chunk.remote_temporal_neighbors =get_remote_neighbors(chunk.local_node_index, num_nodes, temporal_edge_index)
        get_remote_neighbors_time += time.time() - start_time

        chunk.all_node_index = torch.cat([chunk.local_node_index, chunk.remote_spatial_neighbors], dim=0)

        # get node features
        chunk.x = torch.cat([feats[chunk.local_node_index], feats[chunk.remote_spatial_neighbors]])

        # get edge index (include remote neighbors)
        # for efficiency (wrong edge_index but fast)
        start_time = time.time()
        subgraph = graph.subgraph(nodes)
        subgraph_pyg = from_networkx(subgraph)
        chunk.edge_index = subgraph_pyg.edge_index
        remap_edges_time += time.time() - start_time

        # start_time = time.time()
        # local_edges = spatial_edge_index[:, torch.where(torch.isin(spatial_edge_index[1], chunk.local_node_index))[0]]
        # get_all_edges_time += time.time() - start_time

        # start_time = time.time()
        # if len(local_edges)>0:
        #     # local_edges = torch.stack(local_edges, dim=1)
        #     # step 2: update node id in local_edges
        #     sort_node = torch.argsort(chunk.all_node_index)
        #     node_id = torch.cat([chunk.all_node_index[sort_node].unsqueeze(0), sort_node.unsqueeze(0)], dim=0)

        #     # 获取 tensor_B 中每个元素在 tensor_A 中的位置
        #     idx_1 = torch.searchsorted(chunk.all_node_index[sort_node], local_edges[0])
        #     idx_2 = torch.searchsorted(chunk.all_node_index[sort_node], local_edges[1])

        #     # 使用位置更新 tensor_B
        #     local_edges[0] = node_id[1][idx_1]
        #     local_edges[1] = node_id[1][idx_2]
        #     chunk.edge_index = local_edges
        #     remap_edges_time += time.time() - start_time
        # else:
        #     # Extract subgraph induced by current list of nodes
        #     subgraph = graph.subgraph(nodes)
            
        #     # Convert subgraph to PyG graph
        #     subgraph_pyg = from_networkx(subgraph)
            
        #     # # Add edges to PyG graph object
        #     chunk.edge_index = subgraph_pyg.edge_index
        graph_chunks.append(chunk)
    print(f'get remote neighbors time {get_remote_neighbors_time}, get all edges time {get_all_edges_time}, remap edges time {remap_edges_time}')
    return graph_chunks

def sequence_generation(graph, features, spatial_edge_index, temporal_edge_index):
    '''
    para:
        graph: a networkx graph with spatial and temporal edges ['type]
        features: a list of feature tensor, from Data.x
        spatial_edge_index: all spatial edges
        temporal_edge_index: all temporal edges
    return:
        A list of sequences, where each sequence is a Data
    '''
    feats = torch.cat(features, dim = 0)
    num_nodes = feats.size(0)

    # Create label-to-nodes dictionary
    node_orig_id = nx.get_node_attributes(graph, 'orig_id') # nodes with same orig_id belong to the same sequence
    orig_id_to_nodes = {}
    for node, orig_id in node_orig_id.items():
        if orig_id not in orig_id_to_nodes:
            orig_id_to_nodes[orig_id] = []
        orig_id_to_nodes[orig_id].append(node)
    
    temporal_sequence = []
    get_remote_neighbors_time = 0
    get_all_edges_time = 0
    remap_edges_time = 0
    for orig_id, nodes in orig_id_to_nodes.items():
        sequence = Data(x=[], edge_index=[])

        # Add original node index, which is used to get cross-sequence edges
        sequence.local_node_index = torch.tensor(nodes, dtype=torch.long)

        # get remote neighbors
        start_time = time.time()
        sequence.remote_spatial_neighbors =get_remote_neighbors(sequence.local_node_index, num_nodes, spatial_edge_index)
        # sequence.remote_temporal_neighbors =get_remote_neighbors(sequence.local_node_index, num_nodes, temporal_edge_index)
        sequence.remote_temporal_neighbors = torch.tensor([],dtype=torch.long)
        get_remote_neighbors_time += time.time() - start_time

        # # get local belong remote neighbors
        # start_time = time.time()
        # sequence.local_spatial_neighbors =get_local_belong_remote_neighbors(sequence.local_node_index, spatial_edge_index)
        # sequence.local_temporal_neighbors = torch.tensor([],dtype=torch.long)
        # get_remote_neighbors_time += time.time() - start_time

        sequence.all_node_index = torch.cat([sequence.local_node_index, sequence.remote_spatial_neighbors], dim=0)

        # get node features
        sequence.x = torch.cat([feats[sequence.local_node_index], feats[sequence.remote_spatial_neighbors]])

        # get edge index (include remote neighbors)
        # for efficiency (wrong edge_index but fast)
        start_time = time.time()
        subgraph = graph.subgraph(nodes)
        subgraph_pyg = from_networkx(subgraph)
        sequence.edge_index = subgraph_pyg.edge_index
        remap_edges_time += time.time() - start_time

        # correct edge_index but slow
        # local_edges = []
        # for i in range(spatial_edge_index.shape[1]):
        #     if spatial_edge_index[1, i] in sequence.local_node_index:
        #         local_edges.append(spatial_edge_index[:, i])
        # get_all_edges_time += time.time() - start_time
        # start_time = time.time()
        # if len(local_edges)>0:
        #     local_edges = torch.stack(local_edges, dim=1)
        #     # step 2: update node id in local_edges
        #     sort_node = torch.argsort(sequence.all_node_index)
        #     node_id = torch.cat([sequence.all_node_index[sort_node].unsqueeze(0), sort_node.unsqueeze(0)], dim=0)

        #     # 获取 tensor_B 中每个元素在 tensor_A 中的位置
        #     idx_1 = torch.searchsorted(sequence.all_node_index[sort_node], local_edges[0])
        #     idx_2 = torch.searchsorted(sequence.all_node_index[sort_node], local_edges[1])

        #     # 使用位置更新 tensor_B
        #     local_edges[0] = node_id[1][idx_1]
        #     local_edges[1] = node_id[1][idx_2]
        #     sequence.edge_index = local_edges
        # else:
        #     # Extract subgraph induced by current list of nodes
        #     subgraph = graph.subgraph(nodes)
            
        #     # Convert subgraph to PyG graph
        #     subgraph_pyg = from_networkx(subgraph)
            
        #     # # Add edges to PyG graph object
        #     sequence.edge_index = subgraph_pyg.edge_index
        temporal_sequence.append(sequence)   
    return temporal_sequence

def snapshots_generation(graph, features, spatial_edge_index, temporal_edge_index):
    '''
    para:
        graph: a networkx graph with spatial and temporal edges ['type]
        features: a list of feature tensor, from Data.x
        spatial_edge_index: all spatial edges
        temporal_edge_index: all temporal edges
    return:
        A list of snapshots, where each snapshot is a Data
    '''
    feats = torch.cat(features, dim = 0)
    num_nodes = feats.size(0)

    # Create label-to-nodes dictionary
    node_snap_id = nx.get_node_attributes(graph, 'snap_id') # nodes with same orig_id belong to the same sequence
    snap_id_to_nodes = {}
    for node, snap_id in node_snap_id.items():
        if snap_id not in snap_id_to_nodes:
            snap_id_to_nodes[snap_id] = []
        snap_id_to_nodes[snap_id].append(node)
    
    spatial_snapshots = []
    get_remote_neighbors_time = 0
    get_all_edges_time = 0
    remap_edges_time = 0
    for snap_id, nodes in snap_id_to_nodes.items():
        snapshot = Data(x=[], edge_index=[])

        # Add original node index, which is used to get cross-snapshot edges
        snapshot.local_node_index = torch.tensor(nodes, dtype=torch.long)

        # get remote neighbors
        start_time = time.time()
        # snapshot.remote_spatial_neighbors =get_remote_neighbors(snapshot.local_node_index, num_nodes, spatial_edge_index)
        snapshot.remote_spatial_neighbors = torch.tensor([], dtype=torch.long)
        snapshot.remote_temporal_neighbors =get_remote_neighbors(snapshot.local_node_index, num_nodes, temporal_edge_index)
        get_remote_neighbors_time += time.time() - start_time

        # # get local belong remote neighbors
        # start_time = time.time()
        # snapshot.local_spatial_neighbors =get_local_belong_remote_neighbors(snapshot.local_node_index, spatial_edge_index)
        # snapshot.local_temporal_neighbors =get_local_belong_remote_neighbors(snapshot.local_node_index, temporal_edge_index)
        # get_remote_neighbors_time += time.time() - start_time

        snapshot.all_node_index = torch.cat([snapshot.local_node_index, snapshot.remote_spatial_neighbors], dim=0)

        # get node features
        snapshot.x = torch.cat([feats[snapshot.local_node_index], feats[snapshot.remote_spatial_neighbors]])

        # get edge index (include remote neighbors)
        # for efficiency (correct edge_index but fast)
        start_time = time.time()
        subgraph = graph.subgraph(nodes)
        subgraph_pyg = from_networkx(subgraph)
        snapshot.edge_index = subgraph_pyg.edge_index
        remap_edges_time += time.time() - start_time
        
        # # step 1: get all edges
        # start_time = time.time()
        # local_edges = []
        # for i in range(spatial_edge_index.shape[1]):
        #     if spatial_edge_index[1, i] in snapshot.local_node_index:
        #         local_edges.append(spatial_edge_index[:, i])
        # get_all_edges_time += time.time() - start_time
        # start_time = time.time()
        # if len(local_edges)>0:
        #     local_edges = torch.stack(local_edges, dim=1)
        #     # step 2: update node id in local_edges
        #     sort_node = torch.argsort(snapshot.all_node_index)
        #     node_id = torch.cat([snapshot.all_node_index[sort_node].unsqueeze(0), sort_node.unsqueeze(0)], dim=0)

        #     # 获取 tensor_B 中每个元素在 tensor_A 中的位置
        #     idx_1 = torch.searchsorted(snapshot.all_node_index[sort_node], local_edges[0])
        #     idx_2 = torch.searchsorted(snapshot.all_node_index[sort_node], local_edges[1])

        #     # 使用位置更新 tensor_B
        #     local_edges[0] = node_id[1][idx_1]
        #     local_edges[1] = node_id[1][idx_2]
        #     snapshot.edge_index = local_edges
        #     remap_edges_time += time.time() - start_time
        # else:
        #     # Extract subgraph induced by current list of nodes
        #     subgraph = graph.subgraph(nodes)
            
        #     # Convert subgraph to PyG graph
        #     subgraph_pyg = from_networkx(subgraph)
            
        #     # # Add edges to PyG graph object
        #     snapshot.edge_index = subgraph_pyg.edge_index
        spatial_snapshots.append(snapshot)   
    return spatial_snapshots

def assignment(graphs, num_gpus):
    """
    parameters:
        graphs: a list of PyG.Data, graph can be snapshot or sequence or chunk
    return:
        graph to GPU map
    """
    Workload = [0 for i in range(num_gpus)]
    graph_gpu_map = {} # key: gpu_id, value: [chunks]
    for graph in graphs:
        m = Workload.index(min(Workload))
        if m not in graph_gpu_map.keys():
            graph_gpu_map[m] = []
        graph_gpu_map[m].append(graph)
        Workload[m] += graph.x.size(0)
    return graph_gpu_map

def assignment_advance(graphs, num_gpus):
    """
    parameters:
        graphs: a list of PyG.Data, graph can be snapshot or sequence or chunk
    return:
        graph to GPU map
    """
    workload = [0 for i in range(num_gpus)]
    nodes_in_gpu = [torch.tensor([], dtype=torch.long) for i in range(num_gpus)]
    graph_gpu_map = {}
    average_workload = sum([graph.local_node_index.size(0) for graph in graphs])/num_gpus
    num_chunks = [0 for i in range(num_gpus)]
    for graph in graphs:
        scores = []
        for m in range(num_gpus):
            workload_capacity = average_workload - workload[m]
            communication_1 = len(set(nodes_in_gpu[m].tolist()).intersection(set(graph.remote_spatial_neighbors.tolist())))
            communication_2 = len(set(nodes_in_gpu[m].tolist()).intersection(set(graph.remote_temporal_neighbors.tolist())))
            # score = workload_capacity*(communication_1 + communication_2 + 1)
            score = workload_capacity*(len(graphs) - num_chunks[m])
            # score = len(graphs) - num_chunks[m]
            scores.append(score)
        m_ = scores.index(max(scores))
        # update
        workload[m_] += graph.local_node_index.size(0)
        nodes_in_gpu[m_] = torch.cat([nodes_in_gpu[m_], graph.local_node_index], dim=0)
        num_chunks[m_] += 1
        # assignment
        if m_ not in graph_gpu_map.keys():
            graph_gpu_map[m_] = []
        graph_gpu_map[m_].append(graph)
    return graph_gpu_map

def get_remote_neighbors(local_node_index, num_nodes, all_edge_index):
    """
    parameters:
        local_node_index: tensor of local node indices
        all_edge_index: sparse tensor of edges, shpae [E, 2]
    return:
        tensor of remote node indices
    """
    # # find the neighbors of the local nodes
    # remote_neighbors = []
    # for i in range(all_edge_index.shape[1]):
    #     if all_edge_index[1, i] in local_node_index:
    #         neighbor = all_edge_index[0, i]
    #         if neighbor not in local_node_index and neighbor not in remote_neighbors:
    #             remote_neighbors.append(neighbor)
    
    # remote_neighbors = torch.tensor(remote_neighbors, dtype=torch.long)

    # return remote_neighbors

    # find the neighbors of the local nodes
    mask = torch.zeros(num_nodes+1, dtype=torch.bool)
    mask[local_node_index] = True
    remote_neighbors = all_edge_index[0, mask[all_edge_index[1]]]
    remote_neighbors = remote_neighbors[~mask[remote_neighbors]].unique()

    return remote_neighbors

def get_local_belong_remote_neighbors(local_node_index, all_edge_index):
    """
    get all local nodes need to be aggregaed by others
    parameters:
        local_node_index: tensor of local node indices
        all_edge_index: sparse tensor of edges, shpae [2,E]
    return:
        tensor of local edge indices
    """
    # local_belong_remote_neighbors = []
    # for i in range(all_edge_index.shape[1]):
    #     if all_edge_index[0, i] in local_node_index:
    #         neighbor = all_edge_index[1, i]
    #         if neighbor not in local_node_index:
    #             local_belong_remote_neighbors.append(all_edge_index[0, i])
    
    # local_belong_remote_neighbors = torch.tensor(local_belong_remote_neighbors, dtype=torch.long)
    # local_belong_remote_neighbors = torch.unique(local_belong_remote_neighbors)

    # return local_belong_remote_neighbors
    is_local = torch.isin(all_edge_index[0], local_node_index)
    is_remote = ~torch.isin(all_edge_index[1], local_node_index)
    local_belong_remote_neighbors_idx = torch.nonzero(is_local & is_remote, as_tuple=False)[:, 0]
    local_belong_remote_neighbors = all_edge_index[0, local_belong_remote_neighbors_idx]
    local_belong_remote_neighbors = torch.unique(local_belong_remote_neighbors)

    return local_belong_remote_neighbors

def pull_tensors(kvstore_client, layer, keys):
    # if pull_type == 'spatial':
    #     remote_neighbors = graph.remote_spatial_neighbors
    # else:
    #     remote_neighbors = graph.remote_temporal_neighbors
    # if len(remote_neighbors) > 0:
    #     mes = kvstore_client.pull(layer, remote_neighbors)
    # else:
    #     mes = []
    # return mes
    if keys.size(0) > 0:
        return kvstore_client.pull(layer, keys)
    else:
        return []

def push_tensors(kvstore_client, layer, keys, values):
    # if push_type == 'spatial':
    #     nodes_id = graph.local_spatial_neighbors
    # else:
    #     nodes_id = graph.local_temporal_neighbors
    # if nodes_id.size(0)>0:
    #     tensor_id = torch.where(graph.local_node_index == nodes_id.view(-1, 1))[1]
    #     tensors = values[tensor_id].to('cpu')
    #     kvstore_client.push(layer, nodes_id, tensors)
    if keys.size(0) > 0:
        kvstore_client.push(layer, keys, values)

def pull_all_tensors(kvstore_client, layer, graphs, pull_type):
    mes = []
    for i, graph in enumerate(graphs):
        mes.append(pull_tensors(kvstore_client, graph, layer, pull_type))
    return mes

def push_all_tensors(kvstore_client, layer, graphs, values, push_type):

    for i, graph in enumerate(graphs):
        push_tensors(kvstore_client, graph, layer, values[i], push_type)
        # if push_type == 'spatial':
        #     nodes_id = graph.local_spatial_neighbors
        # else:
        #     nodes_id = graph.local_temporal_neighbors
        # if nodes_id.size(0)>0:
        #     tensor_id = torch.where(graph.local_node_index == nodes_id.view(-1, 1))[1]
        #     tensors = values[i][tensor_id].to('cpu')
        #     push_tensors(kvstore_client, layer, nodes_id, tensors)