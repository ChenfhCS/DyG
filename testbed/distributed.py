import torch
import argparse
import numpy as np
import networkx as nx
import pandas as pd
import logging
import multiprocessing as mp

import warnings
warnings.filterwarnings('ignore')

import os, sys
os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "12346"
os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"
sys.path.append("..") 

from tqdm import tqdm
from torch.nn.parallel import DistributedDataParallel as DDP
from dataset import AmazonDatasetLoader, EpinionDatasetLoader, MovieDatasetLoader, StackDatasetLoader
from nn import DySAT
from nn import classifier
from partitioner import PSS, PTS, PSS_TS, Diana

Comm_backend = 'gloo'

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
    parser.add_argument('--dataset', type=str, default='Amazon',
                        help='method for DGNN training')
    parser.add_argument('--timesteps', type=int, nargs='?', default=15,
                    help="total time steps used for train, eval and test")
    parser.add_argument('--epochs', type=int, nargs='?', default=500,
                    help="total number of epochs")
    parser.add_argument('--world_size', type=int, default=2,
                        help='method for DGNN training')
    parser.add_argument('--stale', dest='stale', action='store_true')
    parser.set_defaults(stale=False)
    args = vars(parser.parse_args())
    return args

def _get_partitions(args, snapshots):
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
    partitioner = PSS(args, graphs, nodes_list, adjs_list, 2)
    total_workload_gcn, total_workload_rnn = partitioner.get_partition()
    # Diana_obj = Diana(args, graphs, nodes_list, adjs_list, args['world_size'], 128*32, 128*32, float(1024*1024*8), logger=None)
    # Diana_obj.partitioning('LDG_base')
    # total_workload_gcn, total_workload_rnn = Diana_obj.get_partition()
    return total_workload_gcn, total_workload_rnn, adjs_list

def _run_training(args):
    # print hyper-parameters
    print('{} {}'.format(args['rank'], args))

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
    model = DDP(model, process_group=args['dp_group'], device_ids=[args['rank']], find_unused_parameters=True)

    _, partition, adjs = _get_partitions(args, snapshots)

    loss_func = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=0.001)

    acc_log = []
    loss_log = []
    best_acc = 0
    # print('features: ', snapshots[0].x[80:90])
    # pbar = tqdm(range(args['epochs']), leave=False)

    # load data to GPU
    for graph in snapshots:
        graph = graph.to(args['device'])

    for epoch in range(args['epochs']):
        model.train()
        loss = 0
        samples = [snapshot.train_samples for snapshot in snapshots]
        _, _, outputs = model(snapshots, samples)

        gpu_mem_alloc = torch.cuda.max_memory_allocated() / 1000000 if torch.cuda.is_available() else 0

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

        if args['rank'] == 0:
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
                acc = ACC/len(snapshots)
                acc_log.append(acc)
                if best_acc <= acc:
                    best_acc = acc
            print('epoch: {} loss: {:.4f} acc: {:.4f} GPU memory {:.3f}'.format(epoch, loss.item(), acc, gpu_mem_alloc))
    if args['rank'] == 0:
        print('best accuracy: {:.3f}'.format(best_acc))

def _set_env(rank):
    args = _get_args()
    world_size = args['world_size']
    args['rank'] = rank
    args['distributed'] = True

    # init the communication group
    dist_init_method = 'tcp://{master_ip}:{master_port}'.format(
            master_ip='127.0.0.1', master_port='12346')
    torch.distributed.init_process_group(backend = Comm_backend,
                                         init_method = dist_init_method,
                                         world_size = world_size,
                                         rank = rank,
                                        )

    # set device
    local_rank = torch.distributed.get_rank()
    # if args['rank'] == 0:
    #     device = torch.device("cpu")
    # else:
    #     device = torch.device("cuda")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    args['device'] = device

    # init mp group for intermediate data exchange
    group_idx = range(world_size)
    args['mp_group'] = [
        torch.distributed.new_group(
            ranks = [i for i in group_idx[worker:]
            ],
            backend = Comm_backend,
        )
        for worker in range (world_size - 1)
    ]

    # init dp group for gradients synchronization
    args['dp_group'] = torch.distributed.new_group(
                        ranks = list(range(world_size)), 
                        backend = Comm_backend,
                        ) 
    
    # local training
    _run_training(args)

if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')
    args = _get_args()
    folder_in = os.path.exists('./log/')
    if not folder_in:
        os.makedirs('./log/')

    world_size = args['world_size']
    assert torch.cuda.device_count() >= world_size, 'No enough GPU!'
    workers = []
    for rank in range(world_size):
        p = mp.Process(target=_set_env, args=(rank,))
        p.start()
        workers.append(p)
    for p in workers:
        p.join()
    