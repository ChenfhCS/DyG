import torch
import torch.nn as nn
import torch.nn.functional as F
import time

import copy
from torch_scatter import scatter
from torch_geometric.utils import softmax

class GAT_Layer(nn.Module):
    def __init__(self, 
                args,
                input_dim, 
                output_dim, 
                n_heads, 
                attn_drop, 
                ffd_drop,
                residual):
        super(GAT_Layer, self).__init__()
        self.args = args
        self.out_dim = output_dim // n_heads
        self.n_heads = n_heads
        self.act = nn.ELU()

        self.lin = nn.Linear(input_dim, n_heads * self.out_dim, bias=False)
        self.att_l = nn.Parameter(torch.Tensor(1, n_heads, self.out_dim))
        self.att_r = nn.Parameter(torch.Tensor(1, n_heads, self.out_dim))

        nn.init.xavier_uniform_(self.lin.weight)
        nn.init.xavier_uniform_(self.att_l)
        nn.init.xavier_uniform_(self.att_r)
        
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)

        self.attn_drop = nn.Dropout(attn_drop)
        self.ffd_drop = nn.Dropout(ffd_drop)

        self.residual = residual
        if self.residual:
            self.lin_residual = nn.Linear(input_dim, n_heads * self.out_dim, bias=False)
    
    def forward(self, graph):
        # graph = copy.deepcopy(graph)
        edge_index = graph.edge_index
        # edge_weight = graph.edge_weight.reshape(-1, 1)
        H, C = self.n_heads, self.out_dim
        x = self.lin(graph.x).view(-1, H, C) # [N, heads, out_dim
        # attention
        alpha_l = (x * self.att_l).sum(dim=-1).squeeze() # [N, heads]
        alpha_r = (x * self.att_r).sum(dim=-1).squeeze()
        alpha_l = alpha_l[edge_index[0]] # [num_edges, heads]
        alpha_r = alpha_r[edge_index[1]]
        alpha = alpha_r + alpha_l
        # alpha = edge_weight * alpha
        alpha = self.leaky_relu(alpha)
        coefficients = softmax(alpha, edge_index[1]) # [num_edges, heads]

        # dropout
        if self.training:
            coefficients = self.attn_drop(coefficients)
            x = self.ffd_drop(x)
        x_j = x[edge_index[0]]  # [num_edges, heads, out_dim]
        out = scatter(x_j * coefficients[:, :, None], edge_index[1], 0, x, reduce='sum')
        out = self.act(out)
        # output    
        # x.scatter_(0, index, x_j * coefficients[:, :, None], reduce="add")
        # out = self.act(scatter(x_j * coefficients[:, :, None], edge_index[1], dim=0, reduce="sum")) # ???????????????edge_index???????????????????????????????????????????????????
        out = x.reshape(-1, self.n_heads*self.out_dim) #[num_nodes, output_dim]

        if self.residual:
            out = out + self.lin_residual(graph.x)
        return out
        # graph.x = out
        # return graph


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

    #     # define weights
    #     self.position_embeddings = nn.Parameter(torch.Tensor(self.num_time_steps, input_dim))
    #     # self.time_weights = nn.Parameter(torch.Tensor(num_time_steps, num_time_steps))
    #     self.Q_embedding_weights = nn.Parameter(torch.Tensor(input_dim, input_dim))
    #     self.K_embedding_weights = nn.Parameter(torch.Tensor(input_dim, input_dim))
    #     self.V_embedding_weights = nn.Parameter(torch.Tensor(input_dim, input_dim))
    #     # ff
    #     self.lin = nn.Linear(input_dim, input_dim, bias=True)
    #     # dropout
    #     self.attn_dp = nn.Dropout(attn_drop)
    #     self.xavier_init()

    # def forward(self, inputs):
    #     """In:  attn_outputs (of StructuralAttentionLayer at each snapshot):= [N, T, F]"""
    #     # 1: Add position embeddings to input
    #     # position_inputs = torch.arange(0,self.num_time_steps).reshape(1, -1).repeat(inputs.shape[0], 1).long().to(inputs.device)
    #     start_time = time.time()

    #     position_inputs = torch.arange(0,inputs.shape[1]).reshape(1, -1).repeat(inputs.shape[0], 1).long().to(inputs.device)
    #     temporal_inputs = inputs + self.position_embeddings[position_inputs] # [N, T, F]

    #     # position_temp = torch.tensor([[j for i in range(self.input_dim)]for j in range (self.num_time_steps)])
    #     # extend_tensor = torch.zeros(inputs.shape[0], position_temp.shape[0], position_temp.shape[1])
    #     # position_extend = position_temp.expand_as(extend_tensor).float().to(inputs.device)
    #     # print(position_extend.type())
    #     # print(self.position_embeddings.type())
    #     # temporal_inputs = inputs + torch.tensordot(position_extend, self.position_embeddings, dims=([2],[0]))
    #     # temporal_inputs = inputs

    #     # 2: Query, Key based multi-head self attention.
    #     q = torch.tensordot(temporal_inputs, self.Q_embedding_weights, dims=([2],[0])) # [N, T, F]
    #     k = torch.tensordot(temporal_inputs, self.K_embedding_weights, dims=([2],[0])) # [N, T, F]
    #     v = torch.tensordot(temporal_inputs, self.V_embedding_weights, dims=([2],[0])) # [N, T, F]

    #     # 3: Split, concat and scale.
    #     split_size = int(q.shape[-1]/self.n_heads)
    #     q_ = torch.cat(torch.split(q, split_size_or_sections=split_size, dim=2), dim=0) # [hN, T, F/h]
    #     k_ = torch.cat(torch.split(k, split_size_or_sections=split_size, dim=2), dim=0) # [hN, T, F/h]
    #     v_ = torch.cat(torch.split(v, split_size_or_sections=split_size, dim=2), dim=0) # [hN, T, F/h]

    #     outputs = torch.matmul(q_, k_.permute(0,2,1)) # [hN, T, T]
    #     outputs = outputs / (self.num_time_steps ** 0.5)

    #     # reduced time interval information
    #     # sample_masks =  self.sample_masks[:self.num_time_steps]
    #     time_interval = [i for i in range(outputs.size(1))]
    #     Time_interval = torch.ones_like(outputs[0])
    #     Time_numpy = Time_interval.cpu().numpy()
    #     for i in range (Time_numpy.shape[0]):
    #         for j in range (Time_numpy.shape[1]):
    #             if i > j:
    #                 Time_numpy[i,j] = float(time_interval[i] - time_interval[j])
    #                 # # Additional 3 (method 3): do not work!
    #                 # Time_interval[i][j] = Time_interval[i][j]*self.ti
    #     # print(Time_numpy)
    #     Time_information = torch.tensor(Time_numpy)
    #     Time_information = torch.tril(Time_information)[None, :, :].repeat(outputs.shape[0], 1, 1)


    #     # 4: Masked (causal) softmax to compute attention weights.
    #     diag_val = torch.ones_like(outputs[0])
    #     tril = torch.tril(diag_val)
    #     masks = tril[None, :, :].repeat(outputs.shape[0], 1, 1) # [h*N, T, T]
    #     padding = torch.ones_like(masks) * (-2**32+1)
    #     outputs = torch.where(masks==0, padding, outputs)

    #     # iterval mask
    #     if self.interval_ratio != 0:
    #         outputs = torch.where(torch.gt(Time_information.cuda(), torch.tensor(self.interval_ratio).cuda()), padding, outputs)  # [h*N, T, T]
    #         Time_information.cpu()
    #     # print(torch.gt(Time_information.cuda(), torch.tensor(self.interval_ratio).cuda()))
    #     # print(outputs.detach().cpu().numpy())

    #     outputs = F.softmax(outputs, dim=2)

    #     self.attn_wts_all = outputs # [h*N, T, T]

    #     # 5: Dropout on attention weights.
    #     if self.training:
    #         outputs = self.attn_dp(outputs)
    #     outputs = torch.matmul(outputs, v_)  # [hN, T, F/h]
    #     outputs = torch.cat(torch.split(outputs, split_size_or_sections=int(outputs.shape[0]/self.n_heads), dim=0), dim=2) # [N, T, F]

    #     # 6: Feedforward and residual
    #     outputs = self.feedforward(outputs)
    #     if self.residual:
    #         outputs = outputs + temporal_inputs
        
    #     # self.args['att_time'] = time.time() - start_time
    #     return outputs

    # def feedforward(self, inputs):
    #     outputs = F.relu(self.lin(inputs))
    #     return outputs + inputs

    # def xavier_init(self):
    #     nn.init.xavier_uniform_(self.position_embeddings)
    #     nn.init.xavier_uniform_(self.Q_embedding_weights)
    #     nn.init.xavier_uniform_(self.K_embedding_weights)
    #     nn.init.xavier_uniform_(self.V_embedding_weights)