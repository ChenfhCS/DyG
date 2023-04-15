"""Tools to calculate edge scores."""

import networkx as nx
from tqdm import tqdm
import torch

def normalized_overlap(g, node_1, node_2):
    """
    Calculating the normalized neighbourhood overlap.
    :param g: NetworkX graph.
    :param node_1: First end node of edge.
    :param node_2: Second end node of edge.
    """
    inter = len(set(nx.neighbors(g, node_1)).intersection(set(nx.neighbors(g, node_2))))
    unio = len(set(nx.neighbors(g, node_1)).union(set(nx.neighbors(g, node_2))))
    return float(inter)/float(unio)

def overlap(g, node_1, node_2):
    """
    Calculating the neighbourhood overlap.
    :param g: NetworkX graph.
    :param node_1: First end node of edge.
    :param node_2: Second end node of edge.
    """
    inter = len(set(nx.neighbors(g, node_1)).intersection(set(nx.neighbors(g, node_2))))
    return float(inter)

def unit(g, node_1, node_2):
    """
    Creating unit weights for edge.
    :param g: NetworkX graph.
    :param node_1: First end node of edge.
    :param node_2: Second end node of edge.
    """
    # print()
    if g.edges[node_1, node_2]['type'] == 'tem':
        return 1
    elif g.edges[node_1, node_2]['type'] == 'str':
        return 1
    else:
        raise ValueError("No such kind of edge...")

def min_norm(g, node_1, node_2):
    """
    Calculating the min normalized neighbourhood overlap.
    :param g: NetworkX graph.
    :param node_1: First end node of edge.
    :param node_2: Second end node of edge.
    """
    inter = len(set(nx.neighbors(g, node_1)).intersection(set(nx.neighbors(g, node_2))))
    min_norm = min(len(set(nx.neighbors(g, node_1))), len(set(nx.neighbors(g, node_2))))
    return float(inter)/float(min_norm)

# def cost(g, node_1, node_2):


def overlap_generator(metric, graph):
    """
    Calculating the overlap for each edge.
    :param metric: Weight metric.
    :param graph: NetworkX object.
    :return : Edge weight hash table.
    """
    edges =[(edge[0], edge[1]) for edge in nx.edges(graph)]
    edges = edges + [(edge[1], edge[0]) for edge in nx.edges(graph)]
    return {edge: metric(graph, edge[0], edge[1]) for edge in edges}

def update_node_weight(graphs, nodes_list_mask, model_str, model_tem, device):
    """
    Update the node weight with computation costs
    :param graphs: original graph lists without coarsened
    :param nodes_list_mask: dict of nodes lists
    :param model_str, model_tem: prediction model for structural and temporal costs
    """
    timesteps = len(nodes_list_mask[0])
    gcn_cost = 0
    att_cost = 0
    node_weight = {}
    for node in nodes_list_mask.keys():
        nodes_lists = nodes_list_mask[node]
        full_nodes = []
        total_time_step = 0
        for t in range(timesteps):
            nodes = nodes_lists[t]
            if len(nodes) > 0:
                # print('nodes {} with same label'.format(nodes))
                graph = graphs[t].subgraph(nodes)
                num_vertices = graph.number_of_nodes()
                num_edges = graph.number_of_edges()
                input = torch.Tensor([float(num_vertices/10000), float(num_edges/10000)]).to(device)
                input = torch.Tensor([float(num_vertices/10000), float(num_edges/10000)]).to(device)
                cost = model_str(input)
                gcn_cost += cost.item()
                full_nodes.extend(nodes)
                total_time_step += 1
        full_nodes = list(set(full_nodes))
        tem_input = torch.Tensor([float(len(full_nodes)/10000), float(total_time_step/10)]).to(device)
        cost = model_tem(tem_input)
        att_cost += cost.item()
        if node not in node_weight.keys():
            node_weight[node] = (1 - (gcn_cost + att_cost))
    
    return node_weight
