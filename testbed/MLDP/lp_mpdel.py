"""Model class label propagation."""

import random
import torch
import networkx as nx
from tqdm import tqdm
# from community import modularity
# from print_and_read import json_dumper
from .lp_helper import overlap, unit, min_norm, normalized_overlap, overlap_generator, update_node_weight

import sys
sys.path.append("..")
from .Model_evaluation.mlp import MLP_Predictor

class LabelPropagator:
    """
    Label propagation class.
    """
    def __init__(self, graphs, graph, args):
        """
        Setting up the Label Propagator object.
        :param graph: NetworkX object.
        :param args: Arguments object.
        """
        self.args = args
        self.seeding = args['seed']
        self.graph_list = graphs
        self.graph = graph
        self.nodes = [node for node in graph.nodes()]
        self.rounds = args['rounds']
        self.labels = {node: node for node in self.nodes}
        self.label_count = len(set(self.labels.values()))
        self.flag = True
        self.weight_setup(args['weighting'])
        self.node_weight = {node: 1 for node in self.nodes}

        # self.nodes_list_mask = {}
        # for node in self.nodes:
        #     if node not in self.nodes_list_mask.keys():
        #         self.nodes_list_mask[node] = [[] for _ in range(len(self.graph_list))]
        #         snap_id = self.graph.nodes[node]['snap_id'][0]
        #         self.nodes_list_mask[node][snap_id].append(self.graph.nodes[node]['orig_id'])
        #     # print(self.nodes_list_mask[node])
        # self.nodes_list_mask_full = {}
        # for node in self.nodes:
        #     if node not in self.nodes_list_mask_full.keys():
        #         self.nodes_list_mask_full[node] = [node]

        # self.device = torch.device("cuda")
        # self.model_str = MLP_Predictor(in_feature = 2)
        # self.model_str.load_state_dict(torch.load('./MLDP/Model_evaluation/model/str_{}.pt'.format(10)))
        # self.model_str = self.model_str.to(self.device)

        # self.model_tem = MLP_Predictor(in_feature = 2)
        # self.model_tem.load_state_dict(torch.load('./MLDP/Model_evaluation/model/tem_{}.pt'.format(10)))
        # self.model_tem = self.model_tem.to(self.device)

        # self.model_str.eval()
        # self.model_tem.eval()

    def weight_setup(self, weighting):
        """
        Calculating the edge weights.
        :param weighting: Type of edge weights.
        """
        if weighting == "overlap":
            self.edge_weights = overlap_generator(overlap, self.graph)
        elif weighting == "unit":
            self.edge_weights = overlap_generator(unit, self.graph)
        elif weighting == "min_norm":
            self.edge_weights = overlap_generator(min_norm, self.graph)
        else:
            self.edge_weights = overlap_generator(normalized_overlap, self.graph)

    def make_a_pick(self, source, neighbors):
        """
        Choosing a neighbor from a propagation source node.
        :param source: Source node.
        :param neigbors: Neighboring nodes.
        """
        scores = {}
        for neighbor in neighbors:
            neighbor_label = self.labels[neighbor]
            if neighbor_label in scores.keys():
                scores[neighbor_label] = scores[neighbor_label] + self.edge_weights[(neighbor, source)]
            else:
                scores[neighbor_label] = self.edge_weights[(neighbor, source)]
        top = [key for key, val in scores.items() if val == max(scores.values())]
        return random.sample(top, 1)[0]

    def make_a_pick_CT(self, source, neighbors):
        """
        Choosing a neighbor from a propagation source node.
        :param source: Source node.
        :param neigbors: Neighboring nodes.
        """
        scores = {}
        for neighbor in neighbors:
            neighbor_label = self.labels[neighbor]
            if neighbor_label in scores.keys():
                scores[neighbor_label] = scores[neighbor_label] + self.edge_weights[(source, neighbor)]*self.node_weight[neighbor]
            else:
                scores[neighbor_label] = self.edge_weights[(source, neighbor)]*self.node_weight[neighbor]
            # if neighbor_label in scores.keys():
            #     scores[neighbor_label] = scores[neighbor_label] + self.node_weight[neighbor]
            # else:
            #     scores[neighbor_label] = self.node_weight[neighbor]
        top = [key for key, val in scores.items() if val == max(scores.values())]
        # print('scores:', scores)
        return random.sample(top, 1)[0]
        # return min(top)

    def graph_update(self):
        """
        update graph after a round of propagation
        """

        # node weight update
        new_node_weight = {}
        new_graph = nx.Graph()
        new_edge_weight = {}

        labels_list = list(set(self.labels.values()))
        # print('labels_list: ', labels_list)
        node_mask = {label: i for (i, label) in enumerate(labels_list)}
        # print('node_mask: ', node_mask)
        num_nodes = len(labels_list)
        nodes = [i for i in range(num_nodes)]
        new_labels = {nodes[i]: labels_list[i] for i in range(len(nodes))}
        new_graph.add_nodes_from(nodes)

        # update the nodes list in the new coarsened graph
        # update the node weights
        if self.args['method'] == 'cost':
            new_nodes_list_mask = {}
            new_nodes_list_mask_full = {}
            new_nodes_list_mask_full_temp = {}
            new_nodes_attr = [[] for i in range(num_nodes)]
            i = 0
            where_to_i = {}
            self.nodes.sort()
            # print('nodes: ', self.nodes)
            for node in self.nodes:
                label = self.labels[node]
                where = node_mask[label]
                if where not in new_nodes_list_mask.keys():
                    new_nodes_list_mask[where] = [[] for i in range(len(self.graph_list))]
                for t in range(len(self.graph_list)):
                    new_nodes_list_mask[where][t].extend(self.nodes_list_mask[node][t])
                
                if where not in new_nodes_list_mask_full.keys():
                    new_nodes_list_mask_full[where] = []
                    new_nodes_list_mask_full_temp[i] = []
                    # new_nodes_attr[where] = []
                    where_to_i[where] = i
                    i = i+1
                new_nodes_list_mask_full[where].extend(self.nodes_list_mask_full[node])
                new_nodes_attr[where].extend(self.graph.nodes[node]['snap_id'])
                new_nodes_list_mask_full_temp[where_to_i[where]].extend(self.nodes_list_mask_full[node])
            # print(new_nodes_list_mask_full, new_nodes_list_mask_full_temp)
            new_node_weight = update_node_weight(self.graph_list, new_nodes_list_mask, self.model_str, self.model_tem, self.device)
            # print('new nodes attr: ', new_nodes_attr)
            attr_list = {node: new_nodes_attr[node] for node in new_graph.nodes()}
            nx.set_node_attributes(new_graph, attr_list, 'snap_id')

        # update the edge weights
        for edge in self.graph.edges():
            v_1 = edge[0]
            v_2 = edge[1]
            where_1 = node_mask[self.labels[v_1]]
            where_2 = node_mask[self.labels[v_2]]
            if where_1 != where_2:
                if new_graph.has_edge(where_1, where_2):
                    new_graph.edges[where_1, where_2]['type'].append([self.graph.edges[edge]['type']])
                    new_edge_weight[(where_2, where_1)] = new_edge_weight[(where_2, where_1)] + self.edge_weights[(v_2, v_1)]
                    new_edge_weight[(where_1, where_2)] = new_edge_weight[(where_1, where_2)] + self.edge_weights[(v_1, v_2)]
                else:
                    new_graph.add_edge(where_1, where_2, type=[self.graph.edges[edge]['type']])
                    new_edge_weight[(where_2, where_1)] = self.edge_weights[(v_2, v_1)]
                    new_edge_weight[(where_1, where_2)] = self.edge_weights[(v_1, v_2)]
        
        # update
        self.nodes_list_mask = new_nodes_list_mask
        self.nodes_list_mask_full = new_nodes_list_mask_full
        self.nodes_list_mask_full_temp = new_nodes_list_mask_full_temp
        self.node_map_mask = {node: self.labels[node] for node in self.graph.nodes()}
        # self.pre_node_weight = {node: self.node_weight[node] for node in self.graph.nodes()}
        self.graph = new_graph
        self.nodes = [node for node in new_graph.nodes()]
        self.labels = new_labels
        self.edge_weights = new_edge_weight
        self.node_weight = new_node_weight

    def do_a_propagation(self, step):
        """
        Doing a propagation round.
        """
        random.seed(self.seeding)
        random.shuffle(self.nodes)
        # print('original labels: ', self.labels)
        for node in self.nodes:
            neighbors = nx.neighbors(self.graph, node)
            neighbors_temp = nx.neighbors(self.graph, node)
            num_neighbors = sum(1 for _ in neighbors_temp)
            if num_neighbors > 0:
                if self.args['method'] == 'cost':
                    pick = self.make_a_pick(node, neighbors)
                else:
                    pick = self.make_a_pick(node, neighbors)
                self.labels[node] = pick
        # print('updated labels: ', self.labels)
        # if self.args.method == 'cost':
        # if step %2 == 0:
        #     self.graph_update()
        current_label_count = len(set(self.labels.values()))
        if self.label_count == current_label_count:
            self.flag = False
        else:
            self.label_count = current_label_count
        # print(self.node_weight)

    def do_a_series_of_propagations(self):
        """
        Doing propagations until convergence or reaching time budget.
        """
        index = 0
        # while index < self.rounds and self.flag:
        step = 1
        # while index < self.rounds:
        for i in tqdm(range(self.args['rounds']), desc='Coarsening...', leave=False):
            index = index + 1
            print("\nLabel propagation round: {}; Number of labels: {}; Number of nodes: {}\n".format(index, self.label_count, len(self.nodes)))
            self.do_a_propagation(step)
            step += 1
        print("")
        # print("Modularity is: " + str(round(modularity(self.labels, self.graph), 3)) + ".\n")
        # json_dumper(self.labels, self.args['assignment_output'])

    def graph_coarsening(self):
        index = 0
        # while index < self.rounds and self.flag:
        step = 1
        pbar = tqdm(range(self.args['rounds']), leave=False)
        for i in pbar:
            index = index + 1
            # print("\nLabel propagation round: {}; Number of labels: {}; Number of nodes: {}\n".format(index, self.label_count, len(self.nodes)))
            self.do_a_propagation(step)
            pbar.set_description('Coarsening | #Labels: {}'.format(self.label_count))
            step += 1
        # return self.graph, self.nodes_list_mask, 
        return self.labels
