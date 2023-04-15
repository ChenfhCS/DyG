"""Model class label propagation."""
import sys,os
sys.path.append(os.path.dirname(__file__) + os.sep + '../')
import random
import torch
import networkx as nx
from tqdm import tqdm
# from community import modularity
from .print_and_read import json_dumper
from .calculation_helper import overlap, unit, min_norm, normalized_overlap, overlap_generator, update_node_weight

# from cost_evaluator import MLP_Predictor

class LabelPropagator:
    """
    Label propagation class.
    """
    def __init__(self, graph, args):
        """
        Setting up the Label Propagator object.
        :param graph: NetworkX object.
        :param args: Arguments object.
        """
        self.args = args
        self.seeding = args['seed']
        self.graph = graph
        self.nodes = [node for node in graph.nodes()]
        self.rounds = args['rounds']
        self.labels = {node: node for node in self.nodes}
        self.label_count = len(set(self.labels.values()))
        self.flag = True
        self.weight_setup(args['weighting'])
        self.node_weight = {node: 1 for node in self.nodes}

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


    def do_a_propagation(self, step):
        """
        Doing a propagation round.
        """
        random.seed(self.seeding)
        random.shuffle(self.nodes)
        for node in self.nodes:
            neighbors = nx.neighbors(self.graph, node)
            neighbors_temp = nx.neighbors(self.graph, node)
            num_neighbors = sum(1 for _ in neighbors_temp)
            if num_neighbors > 0:
                pick = self.make_a_pick(node, neighbors)
                self.labels[node] = pick
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
        while index < self.rounds:
            index = index + 1
            print("\nLabel propagation round: {}; Number of labels: {}; Number of nodes: {}\n".format(index, self.label_count, len(self.nodes)))
            self.do_a_propagation(step)
            step += 1
        print("")
        json_dumper(self.labels, self.args['assignment_output'])

    def graph_coarsening(self):
        index = 0
        for index in tqdm(range(self.rounds), desc='Coarsening', leave=False):
            # index = index + 1
            # print("\nLabel propagation round: {}; Number of labels: {}; Number of nodes: {}\n".format(index, self.label_count, len(self.nodes)))
            self.do_a_propagation(index + 1)

            # step += 1
        nx.set_node_attributes(self.graph, self.labels, "label")
        return self.graph
