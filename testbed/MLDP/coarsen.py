import argparse
import networkx as nx
import torch
import random
import numpy as np
import os, sys
import colorsys

import copy

sys.path.append(os.path.abspath('/home/DyG/'))
from tqdm import tqdm
# from .lp_mpdel import LabelPropagator
from dataset.util import remap
from MLDP.lp_mpdel import LabelPropagator
from MLDP.util import graph_concat

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import math
# from print_and_read import graph_reader, argument_printer

timesteps = 4

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

# import matplotlib.pyplot as plt
# def plot_graph(args,G, node_color, edge_color, pos, flag):
#     # node_color_mask = {}
#     # node_color_mask[0] = 'red'
#     # node_color_mask[1] = 'blue'
#     # node_color_mask[2] = 'yellow'
#     # node_color_mask[3] = 'purple'
#     # # print([full_graph.nodes[node]['snap_id'][0] for node in list(full_graph)])
#     # nodes_colors = [node_color_mask[G.nodes[node]['snap_id'][0]] for node in list(G)]
    
#     # edge_color_mask = {}
#     # edge_color_mask['str'] = 'black'
#     # edge_color_mask['tem'] = 'green'
#     # edges_colors = [edge_color_mask[G.edges[edge]['type']] for edge in G.edges()]

#     fig, ax = plt.subplots()
#     nx.draw(G, ax=ax, node_size=5, 
# 	        width=0.5,
#             pos=pos,
#             node_color=node_color,
#             edge_color=edge_color)
#     plt.savefig('./experiment_results/{}_graph_{}.png'.format(flag, args['dataset']))
#     plt.close()

# def coarsener(args, graph_list, full_graph):
#     # plot_graph(full_graph, args)
#     args['rounds'] = 10
#     args['seed'] = 42
#     args['weighting'] = 'unit'
#     args['method'] = 'cost'
#     # graph_list, full_graph = graph_reader(args, graphs)
#     model = LabelPropagator(graph_list, full_graph, args)
#     coarsened_graph, node_to_nodes_list, node_to_nodes_list_full = model.graph_coarsening()


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
    # return coarsened_graph, node_to_nodes_list

def randomcolor():
    colorArr = ['1','2','3','4','5','6','7','8','9','A','B','C','D','E','F']
    color = ""
    for i in range(6):
        color += colorArr[random.randint(0,14)]
    return "#"+color


def _coarsening(args, graph_list):
    args['rounds'] = 50
    args['seed'] = 42
    args['weighting'] = 'unit'
    args['method'] = 'cost'
    # graph_list, full_graph = graph_reader(args, graphs)
    full_graph = graph_concat(graph_list)
    model = LabelPropagator(graph_list, full_graph, args)
    # coarsened_graph, node_to_nodes_list, node_to_nodes_list_full = model.graph_coarsening()
    labels = model.graph_coarsening()
    return full_graph, labels

def _load_graphs(timesteps):
    current_path = os.path.abspath(os.path.join(os.getcwd(), "../../"))
    graph_list = []
    pbar = tqdm(range(timesteps+2), desc='Loading snapshots', leave=False)
    for snapshot_id in pbar:
        graph_path = current_path + f'/dataset/Amazon/data/snapshots/snapshot_{snapshot_id}.gpickle'
        snapshot = nx.read_gpickle(graph_path)
        if snapshot_id > 0:
            previous_snapshot = graph_list[snapshot_id-1]
            snapshot.add_nodes_from(previous_snapshot.nodes(data=True))
            graph_list[snapshot_id-1] = remap(previous_snapshot)
        graph_list.append(snapshot)
    graph_list[-1] = remap(graph_list[-1])
    return graph_list[2:]

def _plot_chunk(graph, labels):
    graph_chunk = copy.deepcopy(graph)
    #remove temporal edges from the coarsened graph
    temporal_edges = []
    for edge in graph.edges():
        if 'str' not in graph.edges[edge]['type']:
            temporal_edges.append(edge)
    graph.remove_edges_from(temporal_edges)

    colors = {}
    # for i, label in enumerate(labels):
    #     hue = i / len(labels)  # 在色轮上均匀取样
    #     saturation = 0.8 + 0.2 * random.random()  # 饱和度随机
    #     lightness = 0.5 + 0.2 * random.random()  # 明度随机
    #     r, g, b = colorsys.hls_to_rgb(hue, lightness, saturation)
    #     colors[label] = (int(r*255)/255, int(g*255)/255, int(b*255)/255)
    
    cmap = plt.get_cmap('tab10')
    for i, snap_id in enumerate(range(timesteps)):
        hue = (i) / (len(labels) + 1)  # 在色轮上均匀取样
        saturation = 0.8 + 0.2 * random.random()  # 饱和度随机
        lightness = 0.5 + 0.2 * random.random()  # 明度随机
        r, g, b = colorsys.hls_to_rgb(hue, lightness, saturation)
        # colors[snap_id] = (int(r*255)/255, int(g*255)/255, int(b*255)/255)
        colors[snap_id] = cmap(i)

    local_pos_x = {}
    local_pos_y = {}
    labels = {}
    label_count = 0
    for node in graph.nodes():
        if graph.nodes[node]['orig_id'] not in local_pos_x.keys():
            local_pos_x[graph.nodes[node]['orig_id']] = random.uniform(0, 0.25)
        if graph.nodes[node]['orig_id'] not in local_pos_y.keys():
            # local_pos_y[graph.nodes[node]['orig_id']] = random.uniform(0+labels[graph.nodes[node]['label']]*size_label, 0+(labels[graph.nodes[node]['label']]+1)*size_label)
            local_pos_y[graph.nodes[node]['orig_id']] = random.uniform(0, 1)
        if graph.nodes[node]['label'] not in labels.keys():
            labels[graph.nodes[node]['label']] = label_count
            label_count += 1
        # if graph.nodes[node]['label'] not in colors:
        #     colors[graph.nodes[node]['label']] = randomcolor()
        pos_x = local_pos_x[graph.nodes[node]['orig_id']] + 0.35*graph.nodes[node]['snap_id']
        # pos_x = local_pos_x[graph.nodes[node]['orig_id']] + 0.15*labels[graph.nodes[node]['label']]
        # pos_y = local_pos_y[graph.nodes[node]['orig_id']] + 0.2*labels[graph.nodes[node]['label']]
        pos_y = local_pos_y[graph.nodes[node]['orig_id']]
        graph.nodes[node]['pos'] = (pos_x, pos_y)


    # 设置节点的标签和颜色
    node_colors = [colors[graph.nodes[node]['snap_id']] for node in graph.nodes()]
    node_labels = {node: graph.nodes[node]['orig_id'] for node in graph.nodes()}

    # 绘制图形
    pos = nx.get_node_attributes(graph, 'pos')
    fig, ax = plt.subplots()
    nx.draw(graph, pos=pos, with_labels=True, labels=node_labels, node_color=node_colors, node_size=50, font_size=4)
    plt.savefig('graph.png', dpi=900,bbox_inches='tight')
    plt.close()



    # graph chunk
    #remove links inside chunk
    edges = []
    for edge in graph_chunk.edges():
        if graph_chunk.nodes[edge[0]]['label'] == graph_chunk.nodes[edge[1]]['label']:
            edges.append(edge)
    graph_chunk.remove_edges_from(edges)
    local_pos_x = {}
    local_pos_y = {}
    labels = {}
    label_count = 0
    for node in graph_chunk.nodes():
        if node not in local_pos_x.keys():
            local_pos_x[node] = random.uniform(0, 0.1)
        if node not in local_pos_y.keys():
            # local_pos_y[node] = random.uniform(0+labels[node['label']]*size_label, 0+(labels[node['label']]+1)*size_label)
            local_pos_y[node] = random.uniform(node*(1/graph_chunk.number_of_nodes()), (node + 1)*(1/graph_chunk.number_of_nodes()))
        if graph_chunk.nodes[node]['label'] not in labels.keys():
            labels[graph_chunk.nodes[node]['label']] = label_count
            label_count += 1
        # if graph_chunk.nodes[node]['label'] not in colors:
        #     colors[graph_chunk.nodes[node]['label']] = randomcolor()
        # pos_x = local_pos_x[graph_chunk.nodes[node]['orig_id']] + 0.25*graph_chunk.nodes[node]['snap_id']
        pos_x = local_pos_x[node] + 0.2*labels[graph_chunk.nodes[node]['label']]
        # pos_y = local_pos_y[node] + 0.2*labels[node['label']]
        pos_y = local_pos_y[node]
        graph_chunk.nodes[node]['pos_chunk'] = (pos_x, pos_y)

    # 设置节点的标签和颜色
    node_colors = [colors[graph_chunk.nodes[node]['snap_id']] for node in graph_chunk.nodes()]
    node_labels = {node: graph_chunk.nodes[node]['orig_id'] for node in graph_chunk.nodes()}

    # 绘制图形
    pos_chunk = nx.get_node_attributes(graph_chunk, 'pos_chunk')
    fig, ax = plt.subplots()
    nx.draw(graph_chunk, pos=pos_chunk, with_labels=True, labels=node_labels, node_color=node_colors, node_size=50, font_size=4)
    for i in range(len(labels)):
        pos_rec = (-0.05 + 0.2*i, 0)
        rect = Rectangle(pos_rec, 0.15, 1, fill=False, linestyle='--', linewidth=2)
        ax.add_patch(rect)
    plt.savefig('graph_chunk.png', dpi=900,bbox_inches='tight')
    plt.close()

    return 0


def _plot_chunk_label_same_color(graph, labels):
    graph_chunk = copy.deepcopy(graph)
    #remove temporal edges from the coarsened graph
    temporal_edges = []
    for edge in graph.edges():
        if 'str' not in graph.edges[edge]['type']:
            temporal_edges.append(edge)
    graph.remove_edges_from(temporal_edges)

    colors = {}
    for i, label in enumerate(labels):
        hue = i / len(labels)  # 在色轮上均匀取样
        saturation = 0.8 + 0.2 * random.random()  # 饱和度随机
        lightness = 0.5 + 0.2 * random.random()  # 明度随机
        r, g, b = colorsys.hls_to_rgb(hue, lightness, saturation)
        colors[label] = (int(r*255)/255, int(g*255)/255, int(b*255)/255)
    
    cmap = plt.get_cmap('tab10')
    # for i, label in enumerate(labels):
    for i, snap_id in enumerate(range(timesteps)):
        hue = (i) / (len(labels) + 1)  # 在色轮上均匀取样
        saturation = 0.8 + 0.2 * random.random()  # 饱和度随机
        lightness = 0.5 + 0.2 * random.random()  # 明度随机
        r, g, b = colorsys.hls_to_rgb(hue, lightness, saturation)
        # colors[snap_id] = (int(r*255)/255, int(g*255)/255, int(b*255)/255)
        colors[snap_id] = cmap(i)

    local_pos_x = {}
    local_pos_y = {}
    labels = {}
    label_count = 0
    for node in graph.nodes():
        if graph.nodes[node]['orig_id'] not in local_pos_x.keys():
            local_pos_x[graph.nodes[node]['orig_id']] = random.uniform(0, 0.25)
        if graph.nodes[node]['orig_id'] not in local_pos_y.keys():
            # local_pos_y[graph.nodes[node]['orig_id']] = random.uniform(0+labels[graph.nodes[node]['label']]*size_label, 0+(labels[graph.nodes[node]['label']]+1)*size_label)
            local_pos_y[graph.nodes[node]['orig_id']] = random.uniform(0, 1)
        if graph.nodes[node]['label'] not in labels.keys():
            labels[graph.nodes[node]['label']] = label_count
            label_count += 1
        # if graph.nodes[node]['label'] not in colors:
        #     colors[graph.nodes[node]['label']] = randomcolor()
        pos_x = local_pos_x[graph.nodes[node]['orig_id']] + 0.35*graph.nodes[node]['snap_id']
        # pos_x = local_pos_x[graph.nodes[node]['orig_id']] + 0.15*labels[graph.nodes[node]['label']]
        # pos_y = local_pos_y[graph.nodes[node]['orig_id']] + 0.2*labels[graph.nodes[node]['label']]
        pos_y = local_pos_y[graph.nodes[node]['orig_id']]
        graph.nodes[node]['pos'] = (pos_x, pos_y)


    # 设置节点的标签和颜色
    node_colors = ['grey' for node in graph.nodes()]
    node_labels = {node: graph.nodes[node]['orig_id'] for node in graph.nodes()}

    # 绘制图形
    pos = nx.get_node_attributes(graph, 'pos')
    fig, ax = plt.subplots()
    nx.draw(graph, pos=pos, with_labels=True, labels=node_labels, node_color=node_colors, node_size=50, font_size=4)
    plt.savefig('graph_label_same_color.png', dpi=900,bbox_inches='tight')
    plt.close()



    # graph chunk
    #remove links inside chunk
    edges = []
    for edge in graph_chunk.edges():
        if graph_chunk.nodes[edge[0]]['label'] == graph_chunk.nodes[edge[1]]['label']:
            edges.append(edge)
    graph_chunk.remove_edges_from(edges)
    # remove structural links
    edges = []
    for edge in graph_chunk.edges():
        if 'tem' not in graph_chunk.edges[edge]['type']:
            edges.append(edge)
    graph_chunk.remove_edges_from(edges)
    local_pos_x = {}
    local_pos_y = {}
    labels_dict = {}
    label_count = 0
    for node in graph_chunk.nodes():
        if node not in local_pos_x.keys():
            # local_pos_x[node] = random.uniform(0, 0.25) + 0.35*graph_chunk.nodes[node]['snap_id']
            local_pos_x[node] = random.uniform(graph.nodes[node]['snap_id']*(1/timesteps) + graph.nodes[node]['snap_id']*0.1, (graph.nodes[node]['snap_id']+1)*(1/timesteps) + graph.nodes[node]['snap_id']*0.1)
        if graph_chunk.nodes[node]['label'] not in labels_dict.keys():
            labels_dict[graph_chunk.nodes[node]['label']] = label_count
            label_count += 1
        if node not in local_pos_y.keys():
            # local_pos_y[node] = random.uniform(0+labels_dict[graph_chunk.nodes[node]['label']]*size_label, 0+(labels_dict[graph_chunk.nodes[node]['label']]+1)*size_label)
            local_pos_y[node] = random.uniform(labels_dict[graph.nodes[node]['label']]*(1/len(labels)) + labels_dict[graph.nodes[node]['label']]*0.1, (labels_dict[graph.nodes[node]['label']]+1)*(1/len(labels)) + labels_dict[graph.nodes[node]['label']]*0.1)
            # local_pos_y[node] = random.uniform(0, 0.15) + 0.2*labels_dict[graph.nodes[node]['label']]
        # if graph_chunk.nodes[node]['label'] not in colors:
        #     colors[graph_chunk.nodes[node]['label']] = randomcolor()
        pos_x = local_pos_x[node]
        # pos_x = local_pos_x[node] + 0.15*labels_dict[graph_chunk.nodes[node]['label']]
        # pos_y = local_pos_y[node] + 0.2*labels_dict[graph_chunk.nodes[node]['label']]
        pos_y = local_pos_y[node]
        graph_chunk.nodes[node]['pos_chunk'] = (pos_x, pos_y)

    # 设置节点的标签和颜色
    node_colors = [colors[graph_chunk.nodes[node]['snap_id']] for node in graph_chunk.nodes()]
    node_labels = {node: graph_chunk.nodes[node]['orig_id'] for node in graph_chunk.nodes()}

    # 绘制图形
    pos_chunk = nx.get_node_attributes(graph_chunk, 'pos_chunk')
    fig, ax = plt.subplots(figsize=(5, 5))
    for i in range(len(labels)):
        pos_rec = (0, 0 + i * 1/(len(labels)) + (i*0.1))
        # rect = Rectangle(pos_rec, 0.95, min(0.16, 0.99 - pos_rec[1] + 0.1), fill=False, linestyle='--', linewidth=2)
        # rect = Rectangle(pos_rec, 1, min(0.16, 0.99 - pos_rec[1] + 0.1), fill=True, facecolor='grey')
        rect = Rectangle(pos_rec, 1 + timesteps*0.08, 1/(len(labels))+0.05, fill=True, facecolor='grey')
        ax.add_patch(rect)
    for i in range(timesteps):
        pos_rec = (0 + i * 1/timesteps + (i*0.1), 0)
        # rect = Rectangle(pos_rec, 0.3, 1.2, fill=True, facecolor='grey')
        rect = Rectangle(pos_rec, 1/timesteps, 1 + len(labels)*0.1, fill=False, linestyle='--', linewidth=2)
        ax.add_patch(rect)

    nx.draw(graph_chunk, pos=pos_chunk, with_labels=True, labels=node_labels, node_color=node_colors, node_size=50, font_size=4)
    plt.savefig('graph_chunk_label_same_color.png', dpi=900,bbox_inches='tight')
    plt.close()

    return 0

def _cal_dependencies(graph, labels):
    dependencies = [[] for i in range(len(labels))]
    label_dict = {}
    count_label = 0
    for node in graph.nodes():
        node_label = graph.nodes[node]['label']
        if node_label not in label_dict.keys():
            label_dict[node_label] = count_label
            count_label += 1
        label_chunk = label_dict[node_label]
        node_snap = graph.nodes[node]['snap_id']
        node_id = graph.nodes[node]['orig_id']
        for other_node in graph.nodes:
            other_node_snap = graph.nodes[other_node]['snap_id']
            other_node_id = graph.nodes[other_node]['orig_id']
            if other_node_id == node_id:
                if other_node_snap == node_snap - 1:
                    other_node_label = graph.nodes[other_node]['label']
                    if other_node_label not in label_dict.keys():
                        label_dict[other_node_label] = count_label
                        count_label += 1
                    other_node_chunk = label_dict[other_node_label]
                    if other_node_chunk not in dependencies[label_chunk] and other_node_chunk != label_chunk:
                        dependencies[label_chunk].append(other_node_chunk)
    print(dependencies)




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run Label Propagation.")
    args = vars(parser.parse_args())
    graph_list = _load_graphs(timesteps)
    full_graph, labels = _coarsening(args, graph_list)
    print('Number of labels: ', len(set(labels.values())))
    nx.set_node_attributes(full_graph, labels, "label")
    # print([graph.number_of_nodes() for graph in graph_list])
    # print(labels)
    # _plot_chunk(full_graph, set(labels.values()))
    _cal_dependencies(full_graph, set(labels.values()))
    _plot_chunk_label_same_color(full_graph, set(labels.values()))
    # print(type(graph_list[0]))
