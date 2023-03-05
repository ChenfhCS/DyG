import os
import sys
import copy
import torch
import time
import json
import pickle

import boto3
from concurrent.futures import ThreadPoolExecutor, as_completed, wait

import torch.nn as nn
import torch.nn.functional as F

# from Model.layers import StructuralAttentionLayer
# from Model.layers import TemporalAttentionLayer

from .layers import GATLayer as StructuralAttentionLayer
from .layers import ATT_layer as TemporalAttentionLayer

# from utils import *

lambda_client = boto3.client('lambda')
LAMBDA_FUNCTION_NAME = 'layer_forward'
LAMBDA_POOL_SIZE = 30
lambda_pool = ThreadPoolExecutor(max_workers = LAMBDA_POOL_SIZE)

def invoke_lambda(payload):
    response =  lambda_client.invoke(
        FunctionName='layer_forward',
        InvocationType='RequestResponse', # 同步调用
        Payload=json.dumps(payload),
    )
    result = json.loads(response['Payload'].read().decode('utf-8'))

    try:
        if  result['info'] == 'complete':
            out = torch.load('/home/ubuntu/mnt/efs/outputs/structural_out_{}.pt'.format(payload['index']))
            return {
                'index': result['index'],
                'out': out
            }
    except KeyError:
        raise Exception('There is an error in lambda instance {}. The details are as follows:\n {}'.format(payload['index'], result))

def parallel_lambda(payloads):
    futures = []
    for payload in payloads:
        future = lambda_pool.submit(invoke_lambda, payload)
        futures.append(future)
    
    results = [future.result() for future in as_completed(futures)]
    return [r for _, r in sorted(zip([res['index'] for res in results], [res['out'] for res in results]))]

def _tensor_distance(tensor_A, tensor_B):
    sub_c = torch.sub(tensor_A, tensor_B)
    sq_sub_c = sub_c**2
    sum_sq_sub_c = torch.sum(sq_sub_c, dim=2)
    distance = torch.sqrt(sum_sq_sub_c)
    return distance

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
        self.threshold = 0

        self.str_checkpoint_ini = False
        self.tem_checkpoint_ini = False
        
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
        self.structural_head_config = [4]
        self.structural_layer_config = [16]
        self.temporal_head_config = [4]
        self.temporal_layer_config = [16]
        self.spatial_drop = 0.5
        self.temporal_drop = 0.5
        self.out_feats = 16
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
            # graphs[t] = graphs[t].to(self.device)
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
        
        # spatial embedding reuse
        if self.args['stale'] == True:
            self.threshold = self.args['threshold']
            if self.str_checkpoint_ini == False:
                print('initialize spatial embedding checkpoints!')
                self.emb_str_checkpoint = structural_outputs_padded.detach().clone()
                self.str_checkpoint_ini = True
            else:
                distance_str = _tensor_distance(self.emb_str_checkpoint, structural_outputs_padded)  #[N, T]
                for t in range(structural_outputs_padded.size(1)):
                    str_avg_distance = torch.mean(distance_str[:, t])
                    str_max_distance = torch.max(distance_str[:, t])
                    str_normalized_distance = distance_str[:, t]/str_max_distance
                    str_threshold = str_avg_distance*self.threshold
                    smaller_distance_str = torch.nonzero(str_normalized_distance <= self.args['threshold'], as_tuple=False).view(-1)
                    greater_distance_str = torch.nonzero(str_normalized_distance > self.args['threshold'], as_tuple=False).view(-1)
                    # smaller_distance_str = torch.nonzero(distance_str[:, t] <= str_threshold, as_tuple=False).view(-1)
                    # greater_distance_str = torch.nonzero(distance_str[:, t] > str_threshold, as_tuple=False).view(-1)
                    structural_outputs_padded[smaller_distance_str, t, :] = self.emb_str_checkpoint[smaller_distance_str, t, :].detach().clone()
                    self.emb_str_checkpoint[greater_distance_str, t, :] = structural_outputs_padded[greater_distance_str, t, :].detach().clone()

        temporal_out = self.temporal_attn(structural_outputs_padded)
        
        # temporal embedding reuse
        if self.args['stale'] == True:
            if self.tem_checkpoint_ini == False:
                print('initialize temporal embedding checkpoints!')
                self.emb_tem_checkpoint = temporal_out.detach().clone()
                self.tem_checkpoint_ini = True
            else:
                distance_tem = _tensor_distance(self.emb_tem_checkpoint, temporal_out)  #[N, T]
                for t in range(temporal_out.size(1)):
                    tem_avg_distance = torch.mean(distance_tem[:, t])
                    tem_max_distance = torch.max(distance_tem[:, t])
                    tem_normalized_distance = distance_tem[:, t]/tem_max_distance
                    tem_threshold = tem_avg_distance*self.threshold
                    smaller_distance_tem = torch.nonzero(tem_normalized_distance <= self.args['threshold'], as_tuple=False).view(-1)
                    greater_distance_tem = torch.nonzero(tem_normalized_distance > self.args['threshold'], as_tuple=False).view(-1)
                    # smaller_distance_tem = torch.nonzero(distance_tem[:, t] <= tem_threshold, as_tuple=False).view(-1)
                    # greater_distance_tem = torch.nonzero(distance_tem[:, t] > tem_threshold, as_tuple=False).view(-1)
                    temporal_out[smaller_distance_tem, t, :] = self.emb_tem_checkpoint[smaller_distance_tem, t, :].detach().clone()
                    self.emb_tem_checkpoint[greater_distance_tem, t, :] = temporal_out[greater_distance_tem, t, :].detach().clone()

        return structural_outputs_padded, temporal_out

    def forward_lambda(self, graphs, gate = None, distribute = None):
        # 打包每个graph为payload，同时指定flag和layer参数
        payloads = []
        layer_path = '/home/ubuntu/mnt/efs/layers/layer.pt'
        torch.save(self.structural_attn.state_dict(), layer_path, pickle_protocol=2, _use_new_zipfile_serialization=False)
        for i in range(len(graphs)):
            payload = {
                'flag': 'structural',
                'layer_addr': '/mnt/efs/layers/layer.pt',
                'graph_x_addr': '/mnt/efs/graphs/graph_x_{}.pt'.format(i),
                'graph_edge_addr': '/mnt/efs/graphs/graph_edge_{}.pt'.format(i),
                'index': i
            }
            payloads.append(payload)
        print('time to write payloads: ', time.time() - time_start)
        time_start = time.time()
        results = parallel_lambda(payloads)
        print('time to launch lambda instances: ', time.time() - time_start)

        structural_outputs = [g[:,None,:] for g in results] # list of [Ni, 1, F]

        # padding outputs along with Ni
        maximum_node_num = structural_outputs[-1].shape[0]
        out_dim = structural_outputs[-1].shape[-1]
        structural_outputs_padded = []
        for out in structural_outputs:
            zero_padding = torch.zeros(maximum_node_num-out.shape[0], 1, out_dim).to(out.device)
            padded = torch.cat((out, zero_padding), dim=0)
            structural_outputs_padded.append(padded)
        structural_outputs_padded = torch.cat(structural_outputs_padded, dim=1) # [N, T, F]

        # split structural_outputs_padded into multiple pieces and adopt lambda

        temporal_out = self.temporal_attn(structural_outputs_padded)
        
        return structural_outputs_padded, temporal_out

    # construct model
    def build_model(self):
        input_dim = self.num_features

        structural_layer = StructuralAttentionLayer(args=self.args,
                                             input_dim=input_dim,
                                             output_dim=self.structural_layer_config[0],
                                             n_heads=self.structural_head_config[0],
                                             attn_drop=self.spatial_drop,
                                             ffd_drop=self.spatial_drop,
                                             residual=self.residual)

        input_dim = self.structural_layer_config[-1]
        temporal_layer = TemporalAttentionLayer(self.args,
                                           method=0,
                                           input_dim=input_dim,
                                           n_heads=self.temporal_head_config[0],
                                           num_time_steps=self.temporal_time_steps,
                                           attn_drop=self.temporal_drop,
                                           residual=self.residual,
                                           interval_ratio = self.interval_ratio)
        return structural_layer, temporal_layer
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