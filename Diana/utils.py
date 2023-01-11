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

def load_dataset(args):
    '''
    return dataset loader, which consists of a series of snapshot;
    each snapshot is a pytorch data ref, consisting of train, val, test samples
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

def load_feat():
    