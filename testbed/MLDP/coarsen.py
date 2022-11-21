import argparse
import networkx as nx
import torch
import random
import numpy as np

from .lp_mpdel import LabelPropagator
# from print_and_read import graph_reader, argument_printer

def parameter_parser():
    """
    A method to parse up command line parameters. By default it does community detection on the Facebook politicians network.
    The default hyperparameters give a good quality clustering. Default weighting happens by neighborhood overlap.
    """
    parser = argparse.ArgumentParser(description="Run Label Propagation.")

    parser.add_argument("--input",
                        nargs="?",
                        default="./data/politician_edges.csv",
	                help="Input graph path.")

    parser.add_argument("--assignment-output",
                        nargs="?",
                        default="/home/LabelPropagation/output/politician.json",
	                help="Assignment path.")

    parser.add_argument("--weighting",
                        nargs="?",
                        default="overlap",
	                help="Overlap weighting.")

    parser.add_argument("--rounds",
                        type=int,
                        default=30,
	                help="Number of iterations. Default is 30.")

    parser.add_argument("--seed",
                        type=int,
                        default=42,
	                help="Random seed. Default is 42.")
    parser.add_argument("--dataset",
                        type=str,
                        default='Epinion_rating',
	                help="dataset name.")
    parser.add_argument("--timesteps",
                        type=int,
                        default=3,
	                help="Timesteps.")
    parser.add_argument("--method",
                        type=str,
                        default='weight',
	                help="propagation method.")

    args = vars(parser.parse_args())
    return args

def run_coarsening(args):
    # graph_list, full_graph = graph_reader(args.dataset, args)
    model = LabelPropagator(graph_list, full_graph, args)
    model.do_a_series_of_propagations()

import matplotlib.pyplot as plt
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
    plt.savefig('./experiment_results/{}_graph_{}.png'.format(flag, args['dataset']))
    plt.close()

def coarsener(args, graph_list, full_graph):
    # plot_graph(full_graph, args)
    args['rounds'] = 10
    args['seed'] = 42
    args['weighting'] = 'unit'
    args['method'] = 'cost'
    # graph_list, full_graph = graph_reader(args, graphs)
    model = LabelPropagator(graph_list, full_graph, args)
    coarsened_graph, node_to_nodes_list, node_to_nodes_list_full = model.graph_coarsening()


    # # remove temporal edges from the original graph
    # temporal_edges = []
    # for edge in full_graph.edges():
    #     if full_graph.edges[edge]["type"] == 'tem':
    #         temporal_edges.append(edge)
    # full_graph.remove_edges_from(temporal_edges)

    # #remove temporal edges from the coarsened graph
    # temporal_edges = []
    # for edge in coarsened_graph.edges():
    #     if 'str' not in coarsened_graph.edges[edge]['type']:
    #         temporal_edges.append(edge)
    # coarsened_graph.remove_edges_from(temporal_edges)

    # # print('edge attributes: ', [coarsened_graph.edges[edge]['type'] for edge in coarsened_graph.edges()])

    # # assign same idx for vertices that coarsened together in the coarsened graph
    # # print('number of coarsened vertices: ', coarsened_graph.number_of_nodes())
    # # print(node_to_nodes_list_full)
    # node_mask = torch.zeros(len(list(full_graph)), dtype=torch.int)
    # for node in list(coarsened_graph):
    #     nodes_list = node_to_nodes_list_full[node]
    #     node_mask[nodes_list] = torch.tensor([node for i in range(len(nodes_list))], dtype=torch.int)

    # pos_list = [(random.uniform(0, 1), random.uniform(0, 1)) for _ in coarsened_graph.nodes()]
    # # for node in coarsened_graph.nodes():
    # # original_nodes_pos = {node: pos_list[node_mask[node].item()] for node in full_graph.nodes()}
    # original_nodes_pos = {}
    # same_pos_list = []
    # smae_pos_neigh = {}
    # for node in full_graph.nodes():
    #     if node_mask[node].item() not in same_pos_list:
    #         pos_tem = list(pos_list[node_mask[node].item()])
    #         pos_tem[0] += (full_graph.nodes[node]['snap_id'][0]*2)
    #         original_nodes_pos[node] = tuple(pos_tem)
    #         # original_nodes_pos[node]=pos_list[node_mask[node].item()]
    #         same_pos_list.append(node_mask[node].item())
    #         smae_pos_neigh[node_mask[node].item()]= [node]

    #     else:
    #         pos_tem = list(pos_list[node_mask[node].item()])
    #         pos_tem[0] += (full_graph.nodes[node]['snap_id'][0]*2)
    #         str_or_tem = False
    #         for same_node in smae_pos_neigh[node_mask[node].item()]:
    #             if full_graph.has_edge(same_node, node) or full_graph.has_edge(node, same_node):
    #                 str_or_tem = True
    #         if str_or_tem == True:
    #             pos_tem[0] += random.uniform(-0.5, 0.5)
    #             pos_tem[1] += random.uniform(-0.5, 0.5)
    #         else:
    #             pos_tem[0] += random.uniform(-0.01, 0.01)
    #             pos_tem[1] += random.uniform(-0.01, 0.01)
    #         original_nodes_pos[node] = tuple(pos_tem)
    #         smae_pos_neigh[node_mask[node].item()].append(node)
        
    # # coarsened_nodes_pos = {node: pos_list[node] for node in coarsened_graph.nodes()}
    # coarsened_nodes_pos = {}
    # print([coarsened_graph.nodes[node]['snap_id'] for node in coarsened_graph.nodes()])
    # for node in coarsened_graph.nodes():
    #     snap_id = min(coarsened_graph.nodes[node]['snap_id'])
    #     print(snap_id)
    #     pos_tem = list(pos_list[node])
    #     pos_tem[0] += snap_id*2
    #     coarsened_nodes_pos[node] = tuple(pos_tem)
        

    # print(node_mask)
    # # random generate a color list
    # colorArr = ['1','2','3','4','5','6','7','8','9','A','B','C','D','E','F']
    # color_list = []
    # for num_color in range(coarsened_graph.number_of_nodes()):
    #     color = ""
    #     for i in range(6):
    #         color += colorArr[random.randint(0,14)]
    #     color_list.append("#"+color)
    
    # # get color list for nodes and edges in original graph
    # original_node_color_list = [color_list[node_mask[node].item()] for node in full_graph.nodes()]
    # edge_color_mask = {}
    # edge_color_mask['str'] = 'black'
    # edge_color_mask['tem'] = 'green'
    # edges_colors = [edge_color_mask[full_graph.edges[edge]['type']] for edge in full_graph.edges()]

    # plot_graph(args, full_graph, original_node_color_list, edges_colors, original_nodes_pos, 'original')
    # plot_graph(args, coarsened_graph, color_list, 'red', coarsened_nodes_pos, 'coarse')
    return coarsened_graph, node_to_nodes_list

if __name__ == '__main__':
    args = parameter_parser()
    # argument_printer(args)
    run_coarsening(args)


def randomcolor():
    colorArr = ['1','2','3','4','5','6','7','8','9','A','B','C','D','E','F']
    color = ""
    for i in range(6):
        color += colorArr[random.randint(0,14)]
    return "#"+color