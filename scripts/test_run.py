import argparse
import logging
import random
import time
import datetime
import numpy as np
import multiprocessing as mp
import psutil
import os, sys
sys.path.append("..") 

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import torch
import torch.distributed
import torch.nn
import torch.nn.parallel
import torch.utils.data
from sklearn.metrics import average_precision_score, roc_auc_score

from Diana.utils import (build_dynamic_graph, load_feat)
from Diana.distributed.partition import partitioner
from Diana.distributed.worker import work
from Diana.distributed.kvstore import KVStoreServer

from torch.utils.data import dataloader
from torch.multiprocessing import reductions
from multiprocessing.reduction import ForkingPickler

default_collate_func = dataloader.default_collate


def default_collate_override(batch):
  dataloader._use_shared_memory = False
  return default_collate_func(batch)

setattr(dataloader, 'default_collate', default_collate_override)

for t in torch._storage_classes:
  if sys.version_info[0] == 2:
    if t in ForkingPickler.dispatch:
        del ForkingPickler.dispatch[t]
  else:
    if t in ForkingPickler._extra_reducers:
        del ForkingPickler._extra_reducers[t]


datasets = ['Amazon', 'Epinion', 'Movie', 'Stack']
models = ['TGCN', 'MPNN-LSTM', 'GC-LSTM']
# checkpoint_path = os.path.join() # save and load model states

def _get_args():
    parser = argparse.ArgumentParser(description='example settings')
    
    # default configurations
    parser.add_argument("--model", choices=models, required=True,
                    help="model architecture" + '|'.join(models))
    parser.add_argument("--dataset", choices=datasets, required=True,
                    help="dataset:" + '|'.join(datasets))
    parser.add_argument('--timesteps', type=int, nargs='?', default=10,
                    help="total time steps used for train, eval and test")
    parser.add_argument('--epochs', type=int, nargs='?', default=200,
                    help="total number of epochs")
    parser.add_argument('--world_size', type=int, default=1,
                        help='method for DGNN training')
    parser.add_argument("--lr", type=float, default=0.0001,
                        help='learning rate')
    parser.add_argument("--seed", type=int, default=42)
    
    # DGC configurations
    parser.add_argument("--partition_method", type=str, default='PSS',
                        help='learning rate')
    parser.add_argument("--experiment", type=str, default='partition',
                        help='experiments')
    
    args = vars(parser.parse_args())
    logging.info(args)
    return args

def _main():
    args = _get_args()
    args['distributed'] = torch.cuda.device_count() > 1
    if args['distributed']:
        args['world_size'] = min(torch.cuda.device_count(), args['world_size'])
    else:
        args['world_size'] = 1

    # build dynamic graph
    dgraph = build_dynamic_graph(args)

    if args['partition_method'] == 'PGC':
        partition_service = partitioner(args, dgraph, args['world_size'], args['partition_method'])
        datas, data_gpu_map = partition_service.partition()
        args['num_nodes'] = partition_service.num_nodes
        print('total nodes: ', np.sum([graph.x.size(0) for graph in dgraph]))
        print('total chunks: ',len(datas))
    elif args['partition_method'] == 'PSS':
        partition_service = partitioner(args, dgraph, args['world_size'], args['partition_method'])
        datas, data_gpu_map = partition_service.partition()
        args['num_nodes'] = partition_service.num_nodes
        print('total nodes: ', np.sum([graph.x.size(0) for graph in dgraph]))
        print('total snapshots: ',len(datas))
    elif args['partition_method'] == 'PTS':
        partition_service = partitioner(args, dgraph, args['world_size'], args['partition_method'])
        datas, data_gpu_map = partition_service.partition()
        args['num_nodes'] = partition_service.num_nodes
        print('total nodes: ', np.sum([graph.x.size(0) for graph in dgraph]))
        print('total sequences: ',len(datas))
    else:
        raise Exception('There is no such an partition method!')
    
    other_args = (datas, 
                  partition_service.spatial_edge_index,
                  partition_service.temporal_edge_index, 
                  partition_service.train_samples_list,
                  partition_service.test_samples_list,
                  partition_service.train_labels_list,
                  partition_service.test_labels_list,
                  data_gpu_map)

    workers = []
    shared_dict = mp.Manager().dict()
    for rank in range(args['world_size']):
        p = mp.Process(target=work, args=(rank, args, other_args, shared_dict))
        p.start()
        workers.append(p)
    for p in workers:
        p.join()

if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')
    _main()



