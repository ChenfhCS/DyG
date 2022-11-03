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

from MLDP import Partition_DyG

class My_Model(torch.nn.Module):
    def __init__(self, args, node_features):
        super(My_Model, self).__init__()
        self.args = args
        self.dgnn = DySAT(args, num_features = node_features)
        self.classifier = classifier(in_feature = 128)

    def forward(self, snapshots, samples):
        final_emb = self.dgnn(snapshots)
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
        return outputs

def _get_args():
    parser = argparse.ArgumentParser(description='example settings')
    
    # for experimental configurations
    parser.add_argument('--timesteps', type=int, nargs='?', default=10,
                    help="total time steps used for train, eval and test")
    parser.add_argument('--epochs', type=int, nargs='?', default=100,
                    help="total number of epochs")
    parser.add_argument('--world_size', type=int, default=1,
                        help='method for DGNN training')
    parser.add_argument('--dataset', type=str, default='Epinion_rating',
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
    partitioner = Partition_DyG(args, graphs, nodes_list, adjs_list, 2, 128, 128, float(1024*1024*8))
    total_workload_gcn, total_workload_rnn = partitioner.get_partition()
    return total_workload_gcn[1], total_workload_rnn[1]

def _save_log(args, loss_log, acc_log):
    df_loss=pd.DataFrame(data=loss_log)
    df_loss.to_csv('/home/Distributed_DGNN/experiment_results/{}_{}_loss.csv'.format(args['dataset'], args['timesteps']), header=False)
    df_acc=pd.DataFrame(data=acc_log)
    df_acc.to_csv('/home/Distributed_DGNN/experiment_results/{}_{}_acc.csv'.format(args['dataset'], args['timesteps']), header=False)

if __name__ == '__main__':
    args = _get_args()
    args['rank'] = 0
    args['device'] = torch.device("cuda")
    # args['device'] = 'cpu'
    if args['dataset'] == 'Epinion_rating':
        loader = EpinionDatasetLoader(timesteps = args['timesteps'])
    elif args['dataset'] == 'Amazon_rating':
        loader = AmazonDatasetLoader(timesteps = args['timesteps'])
    elif args['dataset'] == 'Movie_rating':
        loader = MovieDatasetLoader(timesteps = args['timesteps'])
    elif args['dataset'] == 'Stack_overflow':
        loader = StackDatasetLoader(timesteps = args['timesteps'])
    else:
        raise ValueError("No such dataset...")

    dataset = loader.get_dataset()
    snapshots = [snapshot for snapshot in dataset]
    model = My_Model(args, node_features = 2).to(args['device'])

    loss_func = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=0.0005)

    print_weight = []
    acc_log = []
    loss_log = []
    print([snapshot.raw_graph.number_of_nodes() for snapshot in snapshots])
    pbar = tqdm(range(200), leave=False)
    for epoch in range(200):
        model.train()
        loss = 0
        samples = [snapshot.train_samples for snapshot in snapshots]
        outputs = model(snapshots, samples)
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

        model.eval()
        ACC = 0
        samples = [snapshot.test_samples for snapshot in snapshots]
        outputs = model(snapshots, samples)
        for time, snapshot in enumerate(snapshots):
            y = outputs[time]
            label = snapshot.test_labels.cpu().numpy()
            prob_f1 = []
            prob_auc = []   
            prob_f1.extend(np.argmax(y.detach().cpu().numpy(), axis = 1))
            ACC += sum(prob_f1 == label)/len(label)
        acc_log.append(ACC/len(snapshots))
        # print_weight.append(list(model.state_dict()['dgnn.temporal_attn.temporal_layer_0.Q_embedding_weights'][0:10, 10].cpu().numpy()))
        print('epoch: {} loss: {} acc: {}'.format(epoch, loss.item(), ACC/len(snapshots)))
        # pbar.set_description('epoch: {} loss: {}'.format(epoch, loss.item()))

    _save_log(args, loss_log, acc_log)
    # print(print_weight)





