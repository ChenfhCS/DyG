import torch
import argparse
import numpy as np
import warnings
import networkx as nx
import pandas as pd
import logging
warnings.filterwarnings('ignore')

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = "1"
import sys
sys.path.append("..") 

from tqdm import tqdm
from dataset import AmazonDatasetLoader, EpinionDatasetLoader, MovieDatasetLoader, StackDatasetLoader
from nn import DySAT
from nn import classifier

from simulation.communication_sim import Simulator
# from simulation.Simulator_new import Diana
from MLDP import Partition_DyG, node_partition_balance

class My_Model(torch.nn.Module):
    def __init__(self, args, node_features):
        super(My_Model, self).__init__()
        self.args = args
        self.dgnn = DySAT(args, num_features = node_features)
        self.classifier = classifier(in_feature = 16)

    def forward(self, snapshots, samples):
        str_emb, final_emb = self.dgnn(snapshots)
        outputs = []
        for time, snapshot in enumerate(snapshots):
            emb = final_emb[:, time, :].to(self.args['device'])

            sample = samples[time]
            # get target embeddings
            source_id = sample[:, 0]
            target_id = sample[:, 1]
            source_emb = emb[source_id]
            target_emb = emb[target_id]
            input_emb = source_emb.mul(target_emb)
            outputs.append(self.classifier(input_emb))
        return str_emb, final_emb, outputs

def _get_args():
    parser = argparse.ArgumentParser(description='example settings')
    
    # for experimental configurations
    parser.add_argument('--timesteps', type=int, nargs='?', default=10,
                    help="total time steps used for train, eval and test")
    parser.add_argument('--epochs', type=int, nargs='?', default=200,
                    help="total number of epochs")
    parser.add_argument('--world_size', type=int, default=2,
                        help='method for DGNN training')
    parser.add_argument('--dataset', type=str, default='Amazon',
                        help='method for DGNN training')
    parser.add_argument('--trace', action='store_true')
    parser.set_defaults(trace=False)
    args = vars(parser.parse_args())
    return args

def _get_partitions(snapshots):
    graphs = [snapshot.raw_graph for snapshot in snapshots]
    nodes_list = [torch.tensor([j for j in range(graphs[i].number_of_nodes())]) for i in range(len(snapshots))]
    adjs_list = []
    for i in range(len(snapshots)):
        adj_sp = nx.adjacency_matrix(graphs[i]).tocoo()
        adj = torch.sparse.LongTensor(torch.LongTensor([adj_sp.row.tolist(), adj_sp.col.tolist()]),
                            torch.LongTensor(adj_sp.data.astype(np.int32))).coalesce()
        adjs_list.append(adj)
    # partitioner = Partition_DyG(args, graphs, nodes_list, adjs_list, 2, 128, 128, float(1024*1024*8))
    # total_workload_gcn, total_workload_rnn = partitioner.get_partition()
    partitioner = node_partition_balance(args, graphs, nodes_list, adjs_list, 2)
    total_workload_gcn, total_workload_rnn = partitioner.get_partition()
    # Diana_obj = Diana(args, graphs, nodes_list, adjs_list, args['world_size'], 128*32, 128*32, float(1024*1024*8), logger=None)
    # Diana_obj.partitioning('LDG_base')
    # total_workload_gcn, total_workload_rnn = Diana_obj.get_partition()
    return total_workload_gcn, total_workload_rnn, adjs_list

def _save_log(args, loss_log, acc_log):
    folder_in = os.path.exists('./log/')
    if not folder_in:
        os.makedirs('./log/')
    df_loss=pd.DataFrame(data=loss_log)
    df_loss.to_csv('./log/{}_{}_loss.csv'.format(args['dataset'], args['timesteps']), header=False)
    df_acc=pd.DataFrame(data=acc_log)
    df_acc.to_csv('./log/{}_{}_acc.csv'.format(args['dataset'], args['timesteps']), header=False)

def run_example(args):
    print('dataset: {} timesteps: {}'.format(args['dataset'], args['timesteps']))
    args['rank'] = 0
    args['device'] = torch.device("cuda")
    # args['device'] = 'cpu'
    if args['dataset'] == 'Epinion':
        loader = EpinionDatasetLoader(timesteps = args['timesteps'])
    elif args['dataset'] == 'Amazon':
        loader = AmazonDatasetLoader(timesteps = args['timesteps'])
    elif args['dataset'] == 'Movie':
        loader = MovieDatasetLoader(timesteps = args['timesteps'])
    elif args['dataset'] == 'Stack':
        loader = StackDatasetLoader(timesteps = args['timesteps'])
    else:
        raise ValueError("No such dataset...")
    dataset = loader.get_dataset()
    snapshots = [snapshot for snapshot in dataset]
    model = My_Model(args, node_features = 2).to(args['device'])

    _, partition, adjs = _get_partitions(snapshots)

    loss_func = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=0.001)

    pre_str_emb = []
    cur_str_emb = []
    pre_tem_emb = []
    cur_tem_emb = []
    acc_log = []
    loss_log = []
    print([snapshot.raw_graph.number_of_nodes() for snapshot in snapshots])
    # print('features: ', snapshots[0].x[80:90])
    pbar = tqdm(range(args['epochs']), leave=False)

    if args['trace']:
        import wandb
        wandb.init(project="DGNN_GPU_utilization", entity="fahao", name='Movie_12')
    for epoch in range(args['epochs']):
        model.train()
        loss = 0
        samples = [snapshot.train_samples for snapshot in snapshots]
        structural_outputs, temporal_outputs, outputs = model(snapshots, samples)

        gpu_mem_alloc = torch.cuda.max_memory_allocated() / 1000000 if torch.cuda.is_available() else 0

        if epoch in [10-1, 50-1, 100-1, 150-1]:
            pre_str_emb.append(structural_outputs[:300, -1, :].detach().cpu().tolist())
            pre_tem_emb.append(temporal_outputs[:300, -1, :].detach().cpu().tolist())
        if epoch in [10, 50, 100, 150]:
            cur_str_emb.append(structural_outputs[:300, -1, :].detach().cpu().tolist())
            cur_tem_emb.append(temporal_outputs[:300, -1, :].detach().cpu().tolist())

        # communication simulation
        if epoch == 0:
            simulator = Simulator(structural_outputs, temporal_outputs, partition, adjs)
            # print('communication vertex index: {}{}'.format(simulator.comm_str_index, simulator.comm_tem_index))
        else:
            original_comm_str, original_comm_tem = simulator.count_comm(structural_outputs, temporal_outputs)
            # increment_comm_str, increment_comm_tem = simulator.count_comm_incremental(structural_outputs, temporal_outputs)
            stale_comm_str, stale_comm_tem = simulator.count_comm_stale(structural_outputs, temporal_outputs, epoch)
            print(original_comm_str, original_comm_tem, stale_comm_str, stale_comm_tem)

        for time, snapshot in enumerate(snapshots):
            y = outputs[time]
            label = snapshot.train_labels
            # print('emb size: {}, target_id size: {}'.format(y.size(), label.size()))
            error = loss_func(y.squeeze(dim=-1), label)
            loss += error
        loss = loss/len(snapshots)
        loss_log.append(loss.item())
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        # print('epoch: {} loss: {}'.format(epoch, loss.item()))

        with torch.no_grad():
            model.eval()
            ACC = 0
            samples = [snapshot.test_samples for snapshot in snapshots]
            _, _, outputs = model(snapshots, samples)
            for time, snapshot in enumerate(snapshots):
                y = outputs[time]
                label = snapshot.test_labels.cpu().numpy()
                prob_f1 = []
                prob_auc = []   
                prob_f1.extend(np.argmax(y.detach().cpu().numpy(), axis = 1))
                ACC += sum(prob_f1 == label)/len(label)
            acc_log.append(ACC/len(snapshots))
        print('epoch: {} loss: {:.4f} acc: {:.4f} GPU memory {:.3f}'.format(epoch, loss.item(), ACC/len(snapshots), gpu_mem_alloc))

        # pbar.set_description('epoch: {} loss: {}'.format(epoch, loss.item()))

    _save_log(args, loss_log, acc_log)

    # print('str emb: {} \n  \n tem_emb: {}'.format(print_str_emb, print_tem_emb))
    # log config
    current_path = os.path.abspath(os.path.join(os.getcwd(), ".."))
    log_file = current_path + '/log/testbed_emb_{}_{}.log'.format(args['dataset'], args['timesteps'])
    logging.basicConfig(filename=log_file,
                        filemode='a',
                        format='%(message)s',
                        level=logging.INFO)
    # logging.basicConfig(level=logging.INFO, format='%(message)s')
    logger = logging.getLogger('example')
    logger.info('previous str embedding {}'.format(pre_str_emb))
    logger.info('previous tem embedding {}'.format(pre_tem_emb))
    logger.info('current str embedding {}'.format(cur_str_emb))
    logger.info('current tem embedding {}'.format(cur_tem_emb))
    handlers = logger.handlers[:]
    for handler in handlers:
        logger.removeHandler(handler)
        handler.close()
    # print(print_weight)

if __name__ == '__main__':
    args = _get_args()
    run_example(args)
    # for dataset in ['Amazon', 'Epinion', 'Movie', 'Stack']:
    #     args['dataset'] = dataset
    #     run_example(args)
    
    # for dataset in ['Amazon', 'Epinion', 'Movie', 'Stack']:
    #     args['dataset'] = dataset
    #     args['timesteps'] = 9
    #     run_example(args)

    




