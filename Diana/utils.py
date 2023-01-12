import logging
import os, sys
import random
import time
from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.distributed

from dgl.utils.shared_mem import create_shared_mem_array, get_shared_mem_array

sys.path.append("..") 
from dataset import AmazonDatasetLoader, EpinionDatasetLoader, MovieDatasetLoader, StackDatasetLoader

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def load_dataset(args):
    '''
    return dataset loader, which consists of a series of snapshot;
    each snapshot is a pytorch data object, consisting of train, val, test samples
    '''
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
    return dataset

def build_dynamic_graph(args):
    '''
    build a dynamic graph object
    '''
    '''
    return dataset loader, which consists of a series of snapshot;
    each snapshot is a pytorch data object, consisting of train, val, test samples
    '''
    if args['dataset'] == 'Epinion':
        dataset = EpinionDatasetLoader(timesteps = args['timesteps'])
    elif args['dataset'] == 'Amazon':
        dataset = AmazonDatasetLoader(timesteps = args['timesteps'])
    elif args['dataset'] == 'Movie':
        dataset = MovieDatasetLoader(timesteps = args['timesteps'])
    elif args['dataset'] == 'Stack':
        dataset = StackDatasetLoader(timesteps = args['timesteps'])
    else:
        raise ValueError("No such dataset...")
    
    dgraph = dataset.get_dataset()
    
    dgraph_to_device = [snapshot.to(args['local_rank']) for snapshot in dgraph]
    return dgraph_to_device


def load_feat(dgraph, shared_memory: bool = False, local_rank: int = 0, 
              local_world_size: int = 1, memmap: bool = False, load_node: bool = True, 
              load_edge: bool = False):
    '''
    load features from the dynamic graph.
    Args:
        dataset: the name of the dataset.
        data_dir: the directory where the dataset is stored.
        shared_memory: whether to use shared memory.
        local_rank: the local rank of the process.
        local_world_size: the local world size of the process.
        memmap (bool): whether to use memmap.
        load_node (bool): whether to load node features.
        load_edge (bool): whether to load edge features.

    Returns:
        node_feats: the node features (tensor list). (None if not available)
        edge_feats: the edge features (tensor list). (None if not available)
    '''
    node_feats_list = list()
    edge_feats_list = list()

    if not shared_memory or (shared_memory and local_rank == 0):
        for snapshot in dgraph:
            if load_node:
                node_feats_list.append(snapshot.x)
            if load_edge:
                node_feats_list.append(snapshot.edge_feats)
    
    # load feats with shared memory
    if shared_memory:
        node_feats_list_shm, edge_feats_list_shm = list(), list()
        if local_rank == 0:
            for timestep, snapshot in enumerate(dgraph):
                if len(node_feats_list) != 0:
                    node_feats_list_shm.append(create_shared_mem_array(
                        'node_feats_{}'.format(timestep), node_feats_list[timestep].shape, 
                        node_feats_list[timestep].dtype))
                    node_feats_list_shm[timestep][:] = node_feats_list[timestep][:]
                if len(edge_feats_list) != 0:
                    edge_feats_list_shm.append(create_shared_mem_array(
                        'edge_feats_{}'.format(timestep), edge_feats_list[timestep].shape, 
                        edge_feats_list[timestep].dtype))
                    edge_feats_list_shm[timestep][:] = edge_feats_list[timestep][:]
            
            node_feats_shape_list = [node_feats.shape for node_feats in node_feats_list]
            edge_feats_shape_list = [edge_feats.shape for edge_feats in edge_feats_list]
            torch.distributed.broadcast_object_list(
                [node_feats_shape_list, edge_feats_shape_list], src = 0)

        elif local_rank != 0:
            shapes = [list(), list()]
            torch.distributed.broadcast_object_list(
                shapes, src=0)
            node_feats_shape_list, edge_feats_shape_list = shapes

            for timestep, snapshot in enumerate(dgraph):
                if len(node_feats_shape_list) != 0:
                    node_feats_list_shm.append(get_shared_mem_array(
                        'node_feats_{}'.format(timestep), node_feats_shape_list[timestep], torch.float32))
                if len(edge_feats_shape_list) != 0:
                    edge_feats_list_shm.append(get_shared_mem_array(
                        'edge_feats_{}'.format(timestep), edge_feats_shape_list[timestep], torch.float32))

        torch.distributed.barrier()
        if len(node_feats_list_shm) != 0:
            logging.info("rank {} node_feats_shm shape {}".format(
                local_rank, node_feats_shape_list))

        if len(edge_feats_list_shm) != 0:
            logging.info("rank {} edge_feats_shm shape {}".format(
                local_rank, edge_feats_shape_list))

        return node_feats_list_shm, edge_feats_list_shm

    return node_feats_list, edge_feats_list

