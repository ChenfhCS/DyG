import copy
import torch
import time
import os, sys
sys.path.append("..") 

import torch.nn as nn
import torch.nn.functional as F

from Diana.distributed.utils import (push_all_tensors, pull_all_tensors)


class GATLayer(nn.Module):
    def __init__(self, 
                args,
                input_dim, 
                output_dim, 
                n_heads, 
                attn_drop, 
                ffd_drop,
                residual):
        super(GATLayer, self).__init__()
        self.args = args
        self.in_dim = input_dim
        self.out_dim = output_dim // n_heads
        self.num_heads = n_heads
        self.attn_drop = nn.Dropout(attn_drop)
        self.ffd_drop = nn.Dropout(ffd_drop)
        self.residual = residual
        if self.residual:
            self.lin_residual = nn.Linear(input_dim, n_heads * self.out_dim, bias=False)
        self.LeakyReLU = nn.LeakyReLU(negative_slope=0.2)

        # define linear layers and attention parameters
        self.W = nn.Linear(self.in_dim, self.out_dim * self.num_heads)
        self.a_l = nn.Parameter(torch.empty(size=(1, self.num_heads, self.out_dim)))
        self.a_r = nn.Parameter(torch.empty(size=(1, self.num_heads, self.out_dim)))
        nn.init.xavier_uniform_(self.W.weight)
        nn.init.xavier_uniform_(self.a_l)
        nn.init.xavier_uniform_(self.a_r)
    
    def forward(self, graph, device):
        x = graph.x
        edge_index = graph.edge_index

        # apply linear transformation
        x = self.W(x).view(-1, self.num_heads, self.out_dim)  # [N, heads, out_dim]

        if edge_index.numel() == 0:
            # no edges in the graph
            return x.view(-1, self.num_heads * self.out_dim)

        # attention
        alpha_l = (x * self.a_l).sum(-1).squeeze()
        alpha_r = (x * self.a_r).sum(-1).squeeze()
        alpha = alpha_l[edge_index[0]] + alpha_r[edge_index[1]]
        alpha = self.LeakyReLU(alpha)

        # softmax
        alpha = alpha - alpha.max(dim = 1, keepdim=True)[0]
        alpha = alpha.exp()
        alpha_sum = torch.zeros((x.size(0), self.num_heads)).to(alpha.device)
        alpha_sum = alpha_sum.scatter_add_(0, edge_index[0].unsqueeze(1).repeat(1, self.num_heads), alpha)
        alpha = alpha / alpha_sum[edge_index[0]]  # [E, H]

        # scatter
        out = torch.zeros((x.size(0), self.num_heads * self.out_dim)).to(alpha.device)
        out = out.scatter_add_(0, edge_index[0].unsqueeze(1).repeat(1, self.num_heads * self.out_dim), 
                                (x[edge_index[1]] * alpha.unsqueeze(-1)).reshape(-1, self.num_heads*self.out_dim))
        out = out + x.reshape(-1, self.num_heads*self.out_dim)
        return out

class ATT_layer(nn.Module):
    '''
    inputs may have different time-series lengths
    Method 1: Pad zero to make all inputs have the same lengths
    '''
    def __init__(self,
                args,
                method,
                # sample_masks,
                input_dim,
                n_heads,
                num_time_steps,
                attn_drop,
                residual,
                interval_ratio):
        super(ATT_layer, self).__init__()
        self.n_heads = n_heads
        self.num_time_steps = num_time_steps
        self.residual = residual
        # self.sample_masks = sample_masks
        self.method = method
        self.interval_ratio = interval_ratio
        self.input_dim = input_dim
        self.args = args

        # define weights
        self.position_embeddings = nn.Parameter(torch.Tensor(num_time_steps, input_dim))
        self.Q_embedding_weights = nn.Parameter(torch.Tensor(input_dim, input_dim))
        self.K_embedding_weights = nn.Parameter(torch.Tensor(input_dim, input_dim))
        self.V_embedding_weights = nn.Parameter(torch.Tensor(input_dim, input_dim))
        # ff
        self.lin = nn.Linear(input_dim, input_dim, bias=True)
        # dropout 
        self.attn_dp = nn.Dropout(attn_drop)
        self.xavier_init()


    def forward(self, inputs):
        """In:  attn_outputs (of StructuralAttentionLayer at each snapshot):= [N, T, F]"""
        # 1: Add position embeddings to input
        position_inputs = torch.arange(0,self.num_time_steps).reshape(1, -1).repeat(inputs.shape[0], 1).long().to(inputs.device)
        temporal_inputs = inputs + self.position_embeddings[position_inputs] # [N, T, F]

        # 2: Query, Key based multi-head self attention.
        q = torch.tensordot(temporal_inputs, self.Q_embedding_weights, dims=([2],[0])) # [N, T, F]
        k = torch.tensordot(temporal_inputs, self.K_embedding_weights, dims=([2],[0])) # [N, T, F]
        v = torch.tensordot(temporal_inputs, self.V_embedding_weights, dims=([2],[0])) # [N, T, F]

        # 3: Split, concat and scale.
        split_size = int(q.shape[-1]/self.n_heads)
        q_ = torch.cat(torch.split(q, split_size_or_sections=split_size, dim=2), dim=0) # [hN, T, F/h]
        k_ = torch.cat(torch.split(k, split_size_or_sections=split_size, dim=2), dim=0) # [hN, T, F/h]
        v_ = torch.cat(torch.split(v, split_size_or_sections=split_size, dim=2), dim=0) # [hN, T, F/h]
        
        outputs = torch.matmul(q_, k_.permute(0,2,1)) # [hN, T, T]
        outputs = outputs / (self.num_time_steps ** 0.5)
        # 4: Masked (causal) softmax to compute attention weights.
        diag_val = torch.ones_like(outputs[0])
        tril = torch.tril(diag_val)
        masks = tril[None, :, :].repeat(outputs.shape[0], 1, 1) # [h*N, T, T]
        padding = torch.ones_like(masks) * (-2**32+1)
        outputs = torch.where(masks==0, padding, outputs)
        outputs = F.softmax(outputs, dim=2)
        self.attn_wts_all = outputs # [h*N, T, T]
                
        # 5: Dropout on attention weights.
        if self.training:
            outputs = self.attn_dp(outputs)
        # print('attention map: ', self.attn_wts_all[0,:,:])
        outputs = torch.matmul(outputs, v_)  # [hN, T, F/h]
        outputs = torch.cat(torch.split(outputs, split_size_or_sections=int(outputs.shape[0]/self.n_heads), dim=0), dim=2) # [N, T, F]
        
        # 6: Feedforward and residual
        outputs = self.feedforward(outputs)
        if self.residual:
            outputs = outputs + temporal_inputs
        return outputs

    def feedforward(self, inputs):
        outputs = F.relu(self.lin(inputs))
        return outputs + inputs

    def xavier_init(self):
        nn.init.xavier_uniform_(self.position_embeddings)
        nn.init.xavier_uniform_(self.Q_embedding_weights)
        nn.init.xavier_uniform_(self.K_embedding_weights)
        nn.init.xavier_uniform_(self.V_embedding_weights)

class DySAT(nn.Module):
    def __init__(self, args, num_features):
        '''
        Args:
            args: hyperparameters
            num_features: input dimension
            time_steps: total timesteps in dataset
            sample_mask: sample different snapshot graphs
            method: adding time interval information methods
        '''
        super(DySAT, self).__init__()
        structural_time_steps = args['timesteps']
        temporal_time_steps = args['timesteps']
        args['window'] = -1
        self.args = args

        # for local training
        self.num_time_steps = args['timesteps']

        # self.rank = args['rank']
        # self.device = args['device']

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

        self.n_hidden = self.temporal_layer_config[-1]

        # construct layers
        self.structural_attn, self.temporal_attn = self.build_model()

    def forward(self, graphs):
        # time_start = time.time()
        structural_out = []
        spatial_comm_time = 0
        spatial_comp_time = 0
        temporal_comm_time = 0
        temporal_comp_time = 0

        # # get remote spatial neighbors
        # start_time = time.time()
        # remote_spatial_emb = pull_all_tensors(self.args['kvstore_client'], layer=0, graphs=graphs, pull_type='spatial')
        # spatial_comm_time += time.time() - start_time
        # self.args['logger'].info("rank: {} pulls remote spatial embeddings of layer {} from the kvstore server! time: {}".format(self.args['rank'], 0, time.time() - start_time))

        # structure encoder computation
        for i, graph in enumerate(graphs):
            start_time = time.time()
            num_local_node = graph.local_node_index.size(0)
            remote_spatial_neighbors = graph.remote_spatial_neighbors
            # if len(remote_spatial_neighbors) > 0:
            #     graph.x[num_local_node:] = remote_spatial_emb[i]

            start_time = time.time()
            structural_out.append(self.structural_attn(graph.to(self.args['device']), self.args['device']))
            spatial_comp_time += time.time() - start_time
            graph.to('cpu')

        # # spatial embeddings update to server
        # push_all_tensors(self.args['kvstore_client'], layer=1, graphs=graphs, values=structural_out, push_type='temporal')
        # self.args['logger'].info("rank: {} pushes node embeddings of layer {} to the kvstore server! time: {}".format(self.args['rank'], 1, time.time() - start_time))
        # torch.distributed.barrier()

        # # get remote temporal neighnors
        # start_time = time.time()
        # remote_temporal_emb = pull_all_tensors(self.args['kvstore_client'], layer=1, graphs=graphs, pull_type='temporal')
        # temporal_comm_time += time.time() - start_time
        # self.args['logger'].info("rank: {} pulls remote temporal embeddings of layer {} from the kvstore server! time: {}".format(self.args['rank'], 1, time.time() - start_time))

        # # form inputs of the time encoder (a wrong way: concatenate remote embeddings directly)
        # for i, graph in enumerate(graphs):
        #     start_time = time.time()
        #     local_emb = structural_out[i]
        #     num_local_node = graph.local_node_index.size(0)
        #     remote_temporal_neighbors = graph.remote_temporal_neighbors






        self.args['logger'].info("rank: {} structure encoder computation time {:.3f} communication time {:.3f}".format(self.args['rank'], spatial_comp_time, spatial_comm_time))
        self.args['logger'].info("rank: {} time encoder computation time {:.3f} communication time {:.3f}".format(self.args['rank'], temporal_comp_time, temporal_comm_time))



        # for t in range(0, self.num_time_steps):
        #     # graphs[t] = graphs[t].to(self.device)
        #     structural_out.append(self.structural_attn(graphs[t]))
        # # print('time to compute structural outputs: ', time.time() - time_start)
        # structural_outputs = [g[:,None,:] for g in structural_out] # list of [Ni, 1, F]

        # # padding outputs along with Ni
        # maximum_node_num = structural_outputs[-1].shape[0]
        # out_dim = structural_outputs[-1].shape[-1]
        # structural_outputs_padded = []
        # for out in structural_outputs:
        #     zero_padding = torch.zeros(maximum_node_num-out.shape[0], 1, out_dim).to(out.device)
        #     padded = torch.cat((out, zero_padding), dim=0)
        #     structural_outputs_padded.append(padded)
        # structural_outputs_padded = torch.cat(structural_outputs_padded, dim=1) # [N, T, F]

        # temporal_out = self.temporal_attn(structural_outputs_padded)

        return structural_out

    # construct model
    def build_model(self):
        input_dim = self.num_features

        structure_encoder_layer = GATLayer(args=self.args,
                                             input_dim=input_dim,
                                             output_dim=self.structural_layer_config[0],
                                             n_heads=self.structural_head_config[0],
                                             attn_drop=self.spatial_drop,
                                             ffd_drop=self.spatial_drop,
                                             residual=self.residual)

        input_dim = self.structural_layer_config[-1]
        time_encoder_layer = ATT_layer(self.args,
                                           method=0,
                                           input_dim=input_dim,
                                           n_heads=self.temporal_head_config[0],
                                           num_time_steps=self.temporal_time_steps,
                                           attn_drop=self.temporal_drop,
                                           residual=self.residual,
                                           interval_ratio = self.interval_ratio)
        return structure_encoder_layer, time_encoder_layer