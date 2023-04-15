import torch
import time
from torch.nn import Parameter
from torch_geometric.nn import ChebConv
from torch_geometric.nn.inits import glorot, zeros

from Diana.distributed.utils import (pull_tensors, push_tensors)

class GCLSTM(torch.nn.Module):
    r"""An implementation of the the Integrated Graph Convolutional Long Short Term
    Memory Cell. For details see this paper: `"GC-LSTM: Graph Convolution Embedded LSTM
    for Dynamic Link Prediction." <https://arxiv.org/abs/1812.04206>`_
    Args:
        in_channels (int): Number of input features.
        out_channels (int): Number of output features.
        K (int): Chebyshev filter size :math:`K`.
        normalization (str, optional): The normalization scheme for the graph
            Laplacian (default: :obj:`"sym"`):
            1. :obj:`None`: No normalization
            :math:`\mathbf{L} = \mathbf{D} - \mathbf{A}`
            2. :obj:`"sym"`: Symmetric normalization
            :math:`\mathbf{L} = \mathbf{I} - \mathbf{D}^{-1/2} \mathbf{A}
            \mathbf{D}^{-1/2}`
            3. :obj:`"rw"`: Random-walk normalization
            :math:`\mathbf{L} = \mathbf{I} - \mathbf{D}^{-1} \mathbf{A}`
            You need to pass :obj:`lambda_max` to the :meth:`forward` method of
            this operator in case the normalization is non-symmetric.
            :obj:`\lambda_max` should be a :class:`torch.Tensor` of size
            :obj:`[num_graphs]` in a mini-batch scenario and a
            scalar/zero-dimensional tensor when operating on single graphs.
            You can pre-compute :obj:`lambda_max` via the
            :class:`torch_geometric.transforms.LaplacianLambdaMax` transform.
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        K: int,
        normalization: str = "sym",
        bias: bool = True,
    ):
        super(GCLSTM, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.K = K
        self.normalization = normalization
        self.bias = bias
        self._create_parameters_and_layers()
        self._set_parameters()

    def _create_input_gate_parameters_and_layers(self):

        self.conv_i = ChebConv(
            in_channels=self.out_channels,
            out_channels=self.out_channels,
            K=self.K,
            normalization=self.normalization,
            bias=self.bias,
        )

        self.W_i = Parameter(torch.Tensor(self.in_channels, self.out_channels))
        self.b_i = Parameter(torch.Tensor(1, self.out_channels))

    def _create_forget_gate_parameters_and_layers(self):

        self.conv_f = ChebConv(
            in_channels=self.out_channels,
            out_channels=self.out_channels,
            K=self.K,
            normalization=self.normalization,
            bias=self.bias,
        )

        self.W_f = Parameter(torch.Tensor(self.in_channels, self.out_channels))
        self.b_f = Parameter(torch.Tensor(1, self.out_channels))

    def _create_cell_state_parameters_and_layers(self):

        self.conv_c = ChebConv(
            in_channels=self.out_channels,
            out_channels=self.out_channels,
            K=self.K,
            normalization=self.normalization,
            bias=self.bias,
        )

        self.W_c = Parameter(torch.Tensor(self.in_channels, self.out_channels))
        self.b_c = Parameter(torch.Tensor(1, self.out_channels))

    def _create_output_gate_parameters_and_layers(self):

        self.conv_o = ChebConv(
            in_channels=self.out_channels,
            out_channels=self.out_channels,
            K=self.K,
            normalization=self.normalization,
            bias=self.bias,
        )

        self.W_o = Parameter(torch.Tensor(self.in_channels, self.out_channels))
        self.b_o = Parameter(torch.Tensor(1, self.out_channels))

    def _create_parameters_and_layers(self):
        self._create_input_gate_parameters_and_layers()
        self._create_forget_gate_parameters_and_layers()
        self._create_cell_state_parameters_and_layers()
        self._create_output_gate_parameters_and_layers()

    def _set_parameters(self):
        glorot(self.W_i)
        glorot(self.W_f)
        glorot(self.W_c)
        glorot(self.W_o)
        zeros(self.b_i)
        zeros(self.b_f)
        zeros(self.b_c)
        zeros(self.b_o)

    def _set_hidden_state(self, X, H):
        if H is None:
            H = torch.zeros(X.shape[0], self.out_channels).to(X.device)
        return H

    def _set_cell_state(self, X, C):
        if C is None:
            C = torch.zeros(X.shape[0], self.out_channels).to(X.device)
        return C

    def _calculate_input_gate(self, X, edge_index, edge_weight, H, C, lambda_max):
        I = torch.matmul(X, self.W_i)
        I = I + self.conv_i(H, edge_index, edge_weight, lambda_max=lambda_max)
        I = I + self.b_i
        I = torch.sigmoid(I)
        return I

    def _calculate_forget_gate(self, X, edge_index, edge_weight, H, C, lambda_max):
        F = torch.matmul(X, self.W_f)
        F = F + self.conv_f(H, edge_index, edge_weight, lambda_max=lambda_max)
        F = F + self.b_f
        F = torch.sigmoid(F)
        return F

    def _calculate_cell_state(self, X, edge_index, edge_weight, H, C, I, F, lambda_max):
        T = torch.matmul(X, self.W_c)
        T = T + self.conv_c(H, edge_index, edge_weight, lambda_max=lambda_max)
        T = T + self.b_c
        T = torch.tanh(T)
        C = F * C + I * T
        return C

    def _calculate_output_gate(self, X, edge_index, edge_weight, H, C, lambda_max):
        O = torch.matmul(X, self.W_o)
        O = O + self.conv_o(H, edge_index, edge_weight, lambda_max=lambda_max)
        O = O + self.b_o
        O = torch.sigmoid(O)
        return O

    def _calculate_hidden_state(self, O, C):
        H = O * torch.tanh(C)
        return H

    # # test workload variance
    # def forward(
    #     self,
    #     args,
    #     X: torch.FloatTensor,
    #     edge_index: torch.LongTensor,
    #     edge_weight: torch.FloatTensor = None,
    #     H: torch.FloatTensor = None,
    #     C: torch.FloatTensor = None,
    #     lambda_max: torch.Tensor = None,
    # ) -> torch.FloatTensor:
    #     """
    #     Making a forward pass. If edge weights are not present the forward pass
    #     defaults to an unweighted graph. If the hidden state and cell state
    #     matrices are not present when the forward pass is called these are
    #     initialized with zeros.
    #     Arg types:
    #         * **args* - include kvstore_client and other metric
    #         * **graph** *(PyTorch Geometirc Data)* - graph
    #         * **X** *(PyTorch Float Tensor)* - Node features.
    #         * **edge_index** *(PyTorch Long Tensor)* - Graph edge indices.
    #         * **edge_weight** *(PyTorch Long Tensor, optional)* - Edge weight vector.
    #         * **H** *(PyTorch Float Tensor, optional)* - Hidden state matrix for all nodes.
    #         * **C** *(PyTorch Float Tensor, optional)* - Cell state matrix for all nodes.
    #         * **lambda_max** *(PyTorch Tensor, optional but mandatory if normalization is not sym)* - Largest eigenvalue of Laplacian.
    #     Return types:
    #         * **H** *(PyTorch Float Tensor)* - Hidden state matrix for all nodes.
    #         * **C** *(PyTorch Float Tensor)* - Cell state matrix for all nodes.
    #     """

    #     H = self._set_hidden_state(X, H)
    #     C = self._set_cell_state(X, C)

    #     start_time = time.time()
    #     I = self._calculate_input_gate(X, edge_index, edge_weight, H, C, lambda_max)  # GCN-1, one spatial communication
    #     F = self._calculate_forget_gate(X, edge_index, edge_weight, H, C, lambda_max) # GCN-1
    #     O = self._calculate_output_gate(X, edge_index, edge_weight, H, C, lambda_max) # GCN-1
    #     C = self._calculate_cell_state(X, edge_index, edge_weight, H, C, I, F, lambda_max) # GCN-2
    #     args['spatial_comp_time'] += time.time() - start_time
    #     start_time = time.time()
    #     H = self._calculate_hidden_state(O, C)  # LSTM-1, one temporal communication
    #     args['temporal_comp_time'] += time.time() - start_time
    #     return H


    def forward_stale(
        self,
        X: torch.FloatTensor,
        edge_index: torch.LongTensor,
        edge_weight: torch.FloatTensor = None,
        H: torch.FloatTensor = None,
        C: torch.FloatTensor = None,
        lambda_max: torch.Tensor = None,
    ) -> torch.FloatTensor:
        """
        Making a forward pass. If edge weights are not present the forward pass
        defaults to an unweighted graph. If the hidden state and cell state
        matrices are not present when the forward pass is called these are
        initialized with zeros.
        Arg types:
            * **args* - include kvstore_client and other metric
            * **graph** *(PyTorch Geometirc Data)* - graph
            * **X** *(PyTorch Float Tensor)* - Node features.
            * **edge_index** *(PyTorch Long Tensor)* - Graph edge indices.
            * **edge_weight** *(PyTorch Long Tensor, optional)* - Edge weight vector.
            * **H** *(PyTorch Float Tensor, optional)* - Hidden state matrix for all nodes.
            * **C** *(PyTorch Float Tensor, optional)* - Cell state matrix for all nodes.
            * **lambda_max** *(PyTorch Tensor, optional but mandatory if normalization is not sym)* - Largest eigenvalue of Laplacian.
        Return types:
            * **H** *(PyTorch Float Tensor)* - Hidden state matrix for all nodes.
            * **C** *(PyTorch Float Tensor)* - Cell state matrix for all nodes.
        """
        
        X = X
        edge_index = edge_index

        H = self._set_hidden_state(X, H)
        C = self._set_cell_state(X, C)

        I = self._calculate_input_gate(X, edge_index, edge_weight, H, C, lambda_max)  # GCN-1, one spatial communication
        F = self._calculate_forget_gate(X, edge_index, edge_weight, H, C, lambda_max) # GCN-1
        O = self._calculate_output_gate(X, edge_index, edge_weight, H, C, lambda_max) # GCN-1
        C = self._calculate_cell_state(X, edge_index, edge_weight, H, C, I, F, lambda_max) # GCN-2
        H = self._calculate_hidden_state(O, C)  # LSTM-1, one temporal communication

        return H

    def forward(
        self,
        args,
        X: torch.FloatTensor,
        edge_index: torch.LongTensor,
        edge_weight: torch.FloatTensor = None,
        H: torch.FloatTensor = None,
        C: torch.FloatTensor = None,
        lambda_max: torch.Tensor = None,
    ) -> torch.FloatTensor:
        """
        Making a forward pass. If edge weights are not present the forward pass
        defaults to an unweighted graph. If the hidden state and cell state
        matrices are not present when the forward pass is called these are
        initialized with zeros.
        Arg types:
            * **args* - include kvstore_client and other metric
            * **graph** *(PyTorch Geometirc Data)* - graph
            * **X** *(PyTorch Float Tensor)* - Node features.
            * **edge_index** *(PyTorch Long Tensor)* - Graph edge indices.
            * **edge_weight** *(PyTorch Long Tensor, optional)* - Edge weight vector.
            * **H** *(PyTorch Float Tensor, optional)* - Hidden state matrix for all nodes.
            * **C** *(PyTorch Float Tensor, optional)* - Cell state matrix for all nodes.
            * **lambda_max** *(PyTorch Tensor, optional but mandatory if normalization is not sym)* - Largest eigenvalue of Laplacian.
        Return types:
            * **H** *(PyTorch Float Tensor)* - Hidden state matrix for all nodes.
            * **C** *(PyTorch Float Tensor)* - Cell state matrix for all nodes.
        """
        
        start_time = time.time()
        X = X.to(args['device'])
        edge_index = edge_index.to(args['device'])
        args['data_loading_time'] += time.time() - start_time

        start_time = time.time()
        H = self._set_hidden_state(X, H)
        C = self._set_cell_state(X, C)
        I = self._calculate_input_gate(X, edge_index, edge_weight, H, C, lambda_max)  # GCN-1
        F = self._calculate_forget_gate(X, edge_index, edge_weight, H, C, lambda_max) # GCN-1
        C = self._calculate_cell_state(X, edge_index, edge_weight, H, C, I, F, lambda_max) # GCN-2
        O = self._calculate_output_gate(X, edge_index, edge_weight, H, C, lambda_max) # GCN-1
        args['spatial_comp_time'] += time.time() - start_time
        start_time = time.time()
        H = self._calculate_hidden_state(O, C)  # LSTM-1
        args['temporal_comp_time'] += time.time() - start_time

        del X, edge_index, C, I, F, O
        torch.cuda.empty_cache()
        return H

    def forward_partition(
        self,
        args,
        graphs,
        # X: torch.FloatTensor,
        # edge_index: torch.LongTensor,
        edge_weight: torch.FloatTensor = None,
        H: torch.FloatTensor = None,
        C: torch.FloatTensor = None,
        lambda_max: torch.Tensor = None,
    ) -> torch.FloatTensor:
        """
        Making a forward pass. If edge weights are not present the forward pass
        defaults to an unweighted graph. If the hidden state and cell state
        matrices are not present when the forward pass is called these are
        initialized with zeros.
        Arg types:
            * **args* - include kvstore_client and other metric
            * **graph** *(PyTorch Geometirc Data)* - graph
            * **X** *(PyTorch Float Tensor)* - Node features.
            * **edge_index** *(PyTorch Long Tensor)* - Graph edge indices.
            * **edge_weight** *(PyTorch Long Tensor, optional)* - Edge weight vector.
            * **H** *(PyTorch Float Tensor, optional)* - Hidden state matrix for all nodes.
            * **C** *(PyTorch Float Tensor, optional)* - Cell state matrix for all nodes.
            * **lambda_max** *(PyTorch Tensor, optional but mandatory if normalization is not sym)* - Largest eigenvalue of Laplacian.
        Return types:
            * **H** *(PyTorch Float Tensor)* - Hidden state matrix for all nodes.
            * **C** *(PyTorch Float Tensor)* - Cell state matrix for all nodes.
        """

        X_list = [graph.x for graph in graphs]
        edge_index_list = [graph.edge_index for graph in graphs]

        H_ini_list = []
        C_ini_list = []
        C_list = []
        I_list = []
        F_list = []
        C_list = []
        O_list = []
        H_list = []

        for i, x in enumerate(X_list):
            H_ini_list.append(self._set_hidden_state(X_list[i], H))
            C_ini_list.append(self._set_cell_state(X_list[i], C))
        
        # aggregate remote spatial neighbors
        start_time = time.time()
        need_nodes = args['spatial_nodes_in_other_gpu']
        remote_spatial_emb = pull_tensors(args['kvstore_client'], layer=0, keys=need_nodes)
        args['spatial_comm_time'] += time.time() - start_time

        # computation GCN-1
        for i, x in enumerate(X_list):
            start_time = time.time()
            I = self._calculate_input_gate(X_list[i], edge_index_list[i], edge_weight, H_ini_list[i], C_ini_list[i], lambda_max)  # GCN-1
            F = self._calculate_forget_gate(X_list[i], edge_index_list[i], edge_weight, H_ini_list[i], C_ini_list[i], lambda_max) # GCN-1
            O = self._calculate_output_gate(X_list[i], edge_index_list[i], edge_weight, H_ini_list[i], C_ini_list[i], lambda_max) # GCN-1
            args['spatial_comp_time'] += time.time() - start_time
            I_list.append(I)
            F_list.append(F)
            O_list.append(O)
        
        # send remote spatial neighbors
        start_time = time.time()
        send_nodes = args['spatial_nodes_for_other_gpu']
        push_tensors(args['kvstore_client'], layer=1, keys=send_nodes, values=torch.cat(I_list, dim=0))
        push_tensors(args['kvstore_client'], layer=1, keys=send_nodes, values=torch.cat(F_list, dim=0))
        push_tensors(args['kvstore_client'], layer=1, keys=send_nodes, values=torch.cat(O_list, dim=0))
        args['spatial_comm_time'] += time.time() - start_time
        torch.distributed.barrier()

        # aggregate remote spatial neighbors
        start_time = time.time()
        need_nodes = args['spatial_nodes_in_other_gpu']
        remote_spatial_emb = pull_tensors(args['kvstore_client'], layer=1, keys=need_nodes)
        args['spatial_comm_time'] += time.time() - start_time

        # computation GCN-2
        for i, x in enumerate(X_list):
            start_time = time.time()
            C = self._calculate_cell_state(X_list[i], edge_index_list[i], edge_weight, H_ini_list[i], C_ini_list[i], I_list[i], F_list[i], lambda_max) # GCN-2
            args['spatial_comp_time'] += time.time() - start_time
            C_list.append(C)
        
        # send remote temporal neighbors
        start_time = time.time()
        send_nodes = args['temporal_nodes_for_other_gpu']
        push_tensors(args['kvstore_client'], layer=2, keys=send_nodes, values=torch.cat(C_list, dim=0))
        args['temporal_comm_time'] += time.time() - start_time
        torch.distributed.barrier()

        # aggregate remote temporal neighbors
        start_time = time.time()
        need_nodes = args['temporal_nodes_in_other_gpu']
        remote_spatial_emb = pull_tensors(args['kvstore_client'], layer=2, keys=need_nodes)
        args['temporal_comm_time'] += time.time() - start_time

        # computation LSTM-1
        for i, x in enumerate(X_list):
            H_list.append(self._calculate_hidden_state(O_list[i], C_list[i]))
        
        return H_list




        


        