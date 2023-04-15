import torch
import torch.nn as nn
import torch.nn.functional as F

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