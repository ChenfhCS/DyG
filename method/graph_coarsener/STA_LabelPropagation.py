import argparse

from .new_model import LabelPropagator
from .print_and_read import graph_reader, argument_printer

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
    graph_list, full_graph = graph_reader(args.dataset, args)
    model = LabelPropagator(graph_list, full_graph, args)
    model.do_a_series_of_propagations()

def propagation(args, full_graph):
    args['rounds'] = 30
    args['seed'] = 42
    args['weighting'] = 'unit'
    args['method'] = 'cost'
    # graph_list, full_graph = graph_reader(args, graphs)
    model = LabelPropagator(full_graph, args)
    graph = model.graph_coarsening()
    return graph

if __name__ == '__main__':
    args = parameter_parser()
    argument_printer(args)
    run_coarsening(args)