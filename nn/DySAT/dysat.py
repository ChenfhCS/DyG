import os
import sys
import copy
import torch
import time
import numpy as np

import torch.nn as nn
import torch.nn.functional as F

# from Model.layers import StructuralAttentionLayer
# from Model.layers import TemporalAttentionLayer

from .layers import GAT_Layer as StructuralAttentionLayer
from .layers import ATT_layer as TemporalAttentionLayer

# from utils import *

class DySAT(nn.Module):
    def __init__(self, args, num_features, workload_GCN = None, workload_RNN = None):
        '''
        Args:
            args: hyperparameters
            num_features: input dimension
            time_steps: total timesteps in dataset
            sample_mask: sample different snapshot graphs
            method: adding time interval information methods
        '''
        super(DySAT, self).__init__()
        # structural_time_steps = args['structural_time_steps']
        # temporal_time_steps = args['temporal_time_steps']
        structural_time_steps = args['timesteps']
        temporal_time_steps = args['timesteps']
        args['window'] = -1
        self.args = args

        # for local training
        self.num_time_steps = args['timesteps']

        self.rank = args['rank']
        self.device = args['device']
        
        # self.workload_GCN = workload_GCN
        # self.workload_RNN = workload_RNN
        # self.local_workload_GCN = workload_GCN[self.rank]
        # self.local_workload_RNN = workload_RNN[self.rank]

        # if args['window'] < 0:
        #     self.structural_time_steps = structural_time_steps # training graph per 'num_time_steps'
        # else:
        #     self.structural_time_steps = min(structural_time_steps, args['window'] + 1)
        self.temporal_time_steps = temporal_time_steps
        self.num_features = num_features

        # network parameters
        self.structural_head_config = [8]
        self.structural_layer_config = [128]
        self.temporal_head_config = [8]
        self.temporal_layer_config = [128]
        self.spatial_drop = 0.1
        self.temporal_drop = 0.5
        self.out_feats = 128
        self.residual = True
        self.interval_ratio = 0

        self.structural_out_pre = []

        self.n_hidden = self.temporal_layer_config[-1]
        # self.method = method

        # construct layers
        self.structural_attn, self.temporal_attn = self.build_model()

    def forward(self, graphs, gate = None, distribute = None):
        structural_out = []
        for t in range(0, self.num_time_steps):
            graphs[t] = graphs[t].to(self.device)
            structural_out.append(self.structural_attn(graphs[t]))
        structural_outputs = [g[:,None,:] for g in structural_out] # list of [Ni, 1, F]

        # padding outputs along with Ni
        maximum_node_num = structural_outputs[-1].shape[0]
        out_dim = structural_outputs[-1].shape[-1]
        structural_outputs_padded = []
        for out in structural_outputs:
            zero_padding = torch.zeros(maximum_node_num-out.shape[0], 1, out_dim).to(out.device)
            padded = torch.cat((out, zero_padding), dim=0)
            structural_outputs_padded.append(padded)
        structural_outputs_padded = torch.cat(structural_outputs_padded, dim=1) # [N, T, F]
        # print([out.size() for out in structural_outputs])
        # print('structure input size: ', structural_outputs_padded.size())
        # if len(self.structural_out_pre) < 1:
        #     for t in range(structural_outputs_padded.size(1)):
        #         temp = structural_outputs_padded[self.args['workload_gcn'][t], t, :].clone()
        #         self.structural_out_pre.append(temp)
        
        # for t in range(structural_outputs_padded.size(1)):
        #     temp = structural_outputs_padded[self.args['workload_gcn'][t], t, :].clone()
        #     structural_outputs_padded[self.args['workload_gcn'][t], t, :] = self.structural_out_pre[t]
        #     self.structural_out_pre[t] = temp.clone()
        #         # temp = structural_outputs_padded[self.args['workload_gcn'][t], :, :]
        #         # self.structural_out_pre.append(temp)
        
        # Temporal Attention forward
        temporal_out = self.temporal_attn(structural_outputs_padded)
        
        # print('node embeddings: ', temporal_out[10, -1, :])
        return temporal_out

    # construct model
    def build_model(self):
        input_dim = self.num_features
        # 1: Structural Attention Layers
        structural_attention_layers = nn.Sequential()
        for i in range(len(self.structural_layer_config)):
            layer = StructuralAttentionLayer(args=self.args,
                                             input_dim=input_dim,
                                             output_dim=self.structural_layer_config[i],
                                             n_heads=self.structural_head_config[i],
                                             attn_drop=self.spatial_drop,
                                             ffd_drop=self.spatial_drop,
                                             residual=self.residual)
            structural_attention_layers.add_module(name="structural_layer_{}".format(i), module=layer)
            input_dim = self.structural_layer_config[i]

        # 2: Temporal Attention Layers
        input_dim = self.structural_layer_config[-1]
        temporal_attention_layers = nn.Sequential()
        for i in range(len(self.temporal_layer_config)):
            layer = TemporalAttentionLayer(self.args,
                                           method=0,
                                           input_dim=input_dim,
                                           n_heads=self.temporal_head_config[i],
                                           num_time_steps=self.temporal_time_steps,
                                           attn_drop=self.temporal_drop,
                                           residual=self.residual,
                                           interval_ratio = self.interval_ratio)
            temporal_attention_layers.add_module(name="temporal_layer_{}".format(i), module=layer)
            input_dim = self.temporal_layer_config[i]
        return structural_attention_layers, temporal_attention_layers