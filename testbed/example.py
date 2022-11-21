import torch
import argparse
import numpy as np
import warnings
import networkx as nx
import pandas as pd
warnings.filterwarnings('ignore')

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

from tqdm import tqdm
from dataset import EpinionDatasetLoader, AmazonDatasetLoader, MovieDatasetLoader, StackDatasetLoader
from nn import DySAT
from nn import classifier

from communication_sim import Simulator
from MLDP import Partition_DyG, node_partition_balance

class My_Model(torch.nn.Module):
    def __init__(self, args, node_features):
        super(My_Model, self).__init__()
        self.args = args
        self.dgnn = DySAT(args, num_features = node_features)
        self.classifier = classifier(in_feature = 128)

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
    parser.add_argument('--epochs', type=int, nargs='?', default=100,
                    help="total number of epochs")
    parser.add_argument('--world_size', type=int, default=2,
                        help='method for DGNN training')
    parser.add_argument('--dataset', type=str, default='Epinion',
                        help='method for DGNN training')
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
    return total_workload_gcn, total_workload_rnn, adjs_list

def _save_log(args, loss_log, acc_log):
    folder_in = os.path.exists('./log/')
    if not folder_in:
        os.makedirs('./log/')
    df_loss=pd.DataFrame(data=loss_log)
    df_loss.to_csv('./log/{}_{}_loss.csv'.format(args['dataset'], args['timesteps']), header=False)
    df_acc=pd.DataFrame(data=acc_log)
    df_acc.to_csv('./log/{}_{}_acc.csv'.format(args['dataset'], args['timesteps']), header=False)

if __name__ == '__main__':
    args = _get_args()
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
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0005)

    print_str_emb = []
    print_tem_emb = []
    acc_log = []
    loss_log = []
    print([snapshot.raw_graph.number_of_nodes() for snapshot in snapshots])
    pbar = tqdm(range(200), leave=False)

    # import wandb
    # wandb.init(project="DGNN_GPU_utilization", entity="fahao", name='Epinion_10')
    for epoch in range(args['epochs']):
        model.train()
        loss = 0
        samples = [snapshot.train_samples for snapshot in snapshots]
        structural_outputs, temporal_outputs, outputs = model(snapshots, samples)

        gpu_mem_alloc = torch.cuda.max_memory_allocated() / 1000000 if torch.cuda.is_available() else 0
        print_str_emb.append(list(structural_outputs[5, 0, :10].detach().cpu().numpy()))
        print_tem_emb.append(list(temporal_outputs[5, 0, :10].detach().cpu().numpy()))

        # communication simulator
        # if epoch == 0:
        #     simulator = Simulator(structural_outputs, temporal_outputs, partition, adjs)
        #     # print('communication vertex index: {}{}'.format(simulator.comm_str_index, simulator.comm_tem_index))
        # else:
        #     original_comm_str, original_comm_tem = simulator.count_comm(structural_outputs, temporal_outputs)
        #     increment_comm_str, increment_comm_tem = simulator.count_comm_incremental(structural_outputs, temporal_outputs)
        #     print(original_comm_str, original_comm_tem, increment_comm_str, increment_comm_tem)

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
        print('epoch: {} loss: {} acc: {} GPU memory {}'.format(epoch, loss.item(), ACC/len(snapshots), gpu_mem_alloc))
        # pbar.set_description('epoch: {} loss: {}'.format(epoch, loss.item()))

    _save_log(args, loss_log, acc_log)
    # print('str emb: {} \n tem_emb: {}'.format(print_str_emb, print_tem_emb))
    # print(print_weight)





