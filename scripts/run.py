import argparse
import logging
import math
import os
import random
import time
import numpy as np

import torch
import torch.distributed
import torch.nn
import torch.nn.parallel
import torch.utils.data
from sklearn.metrics import average_precision_score, roc_auc_score

from Diana.utils import (load_dataset)

datasets = ['Amazon', 'Epinion', 'Movie', 'Stack']
models = ['DySAT']

logging.basicConfig(level=logging.DEBUG)
checkpoint_path = os.path.join() # save and load model states

# set random seeds
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

def _get_args():
    parser = argparse.ArgumentParser(description='example settings')
    
    # default configurations
    parser.add_argument('--dataset', type=str, default='Amazon',
                        help='method for DGNN training')
    parser.add_argument('--timesteps', type=int, nargs='?', default=15,
                    help="total time steps used for train, eval and test")
    parser.add_argument('--epochs', type=int, nargs='?', default=200,
                    help="total number of epochs")
    parser.add_argument('--experiments', type=str, nargs='?', required=True,
                    help="experiment type")
    parser.add_argument('--world_size', type=int, default=2,
                        help='method for DGNN training')
    parser.add_argument("--lr", type=float, default=0.0001,
                        help='learning rate')
    
    # optimization configurations

    args = vars(parser.parse_args())
    logging.info(args)
    return args

def _evaluation(dataloader, sampler, model, criterion, device):
    model.eval()
    val_losses = []
    aps = [] # average precision metrics
    aucs = [] # auc metrics

    with torch.no_grad():
        loss = 0
        for target_nodes, ts, eid in dataloader:
            mfgs = sampler.sample(target_nodes, ts)
            mfgs_to_cuda(mfgs, device)
            mfgs = cache.fetch_feature(
                mfgs, eid)
            pred_pos, pred_neg = model(mfgs)

            if args.use_memory:
                # NB: no need to do backward here
                # use one function
                if args.distributed:
                    model.module.memory.update_mem_mail(
                        **model.module.last_updated, edge_feats=cache.target_edge_features,
                        neg_sample_ratio=1)
                else:
                    model.memory.update_mem_mail(
                        **model.last_updated, edge_feats=cache.target_edge_features,
                        neg_sample_ratio=1)

            total_loss += criterion(pred_pos, torch.ones_like(pred_pos))
            total_loss += criterion(pred_neg, torch.zeros_like(pred_neg))
            y_pred = torch.cat([pred_pos, pred_neg], dim=0).sigmoid().cpu()
            y_true = torch.cat(
                [torch.ones(pred_pos.size(0)),
                 torch.zeros(pred_neg.size(0))], dim=0)
            aucs.append(roc_auc_score(y_true, y_pred))
            aps.append(average_precision_score(y_true, y_pred))

        val_losses.append(float(total_loss))

    ap = float(torch.tensor(aps).mean())
    auc_mrr = float(torch.tensor(aucs).mean())
    return ap, auc_mrr

def _train():
    return 0

def _main():
    args = _get_args()

    # set envs
    if args['distributed']:
        args['local_rank'] = int(os.environ['LOCAL_RANK'])
        args['local_world_size'] = int(os.environ['LOCAL_WORLD_SIZE'])
        torch.cuda.set_device(args['local_rank'])
        torch.distributed.init_process_group('nccl')
        args['rank'] = torch.distributed.get_rank()
        args['world_size'] = torch.distributed.get_world_size()
    else:
        args['local_rank'] = 0
        args['local_world_size'] = 1
    
    logging.info("rank: {}, world_size: {}".format(args['rank'], args['world_size']))

    # load dataset
    dataset = load_dataset(args)

    



