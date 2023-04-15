import torch
import time
from torch_geometric.nn import GCNConv

from Diana.distributed.utils import (pull_tensors, push_tensors)

class TGCN(torch.nn.Module):
    r"""An implementation of the Temporal Graph Convolutional Gated Recurrent Cell.
    For details see this paper: `"T-GCN: A Temporal Graph ConvolutionalNetwork for
    Traffic Prediction." <https://arxiv.org/abs/1811.05320>`_
    Args:
        in_channels (int): Number of input features.
        out_channels (int): Number of output features.
        improved (bool): Stronger self loops. Default is False.
        cached (bool): Caching the message weights. Default is False.
        add_self_loops (bool): Adding self-loops for smoothing. Default is True.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        improved: bool = False,
        cached: bool = False,
        add_self_loops: bool = True,
    ):
        super(TGCN, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.improved = improved
        self.cached = cached
        self.add_self_loops = add_self_loops

        self._create_parameters_and_layers()

    def _create_update_gate_parameters_and_layers(self):

        self.conv_z = GCNConv(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            improved=self.improved,
            cached=self.cached,
            add_self_loops=self.add_self_loops,
        )

        self.linear_z = torch.nn.Linear(2 * self.out_channels, self.out_channels)

    def _create_reset_gate_parameters_and_layers(self):

        self.conv_r = GCNConv(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            improved=self.improved,
            cached=self.cached,
            add_self_loops=self.add_self_loops,
        )

        self.linear_r = torch.nn.Linear(2 * self.out_channels, self.out_channels)

    def _create_candidate_state_parameters_and_layers(self):

        self.conv_h = GCNConv(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            improved=self.improved,
            cached=self.cached,
            add_self_loops=self.add_self_loops,
        )

        self.linear_h = torch.nn.Linear(2 * self.out_channels, self.out_channels)

    def _create_parameters_and_layers(self):
        self._create_update_gate_parameters_and_layers()
        self._create_reset_gate_parameters_and_layers()
        self._create_candidate_state_parameters_and_layers()

    def _set_hidden_state(self, X, H):
        if H is None:
            H = torch.zeros(X.shape[0], self.out_channels).to(X.device)
        return H

    def _calculate_update_gate(self, X, edge_index, edge_weight, H):
        Z = torch.cat([self.conv_z(X, edge_index, edge_weight), H], axis=1)
        Z = self.linear_z(Z)
        Z = torch.sigmoid(Z)
        return Z

    def _calculate_reset_gate(self, X, edge_index, edge_weight, H):
        R = torch.cat([self.conv_r(X, edge_index, edge_weight), H], axis=1)
        R = self.linear_r(R)
        R = torch.sigmoid(R)
        return R

    def _calculate_candidate_state(self, X, edge_index, edge_weight, H, R):
        H_tilde = torch.cat([self.conv_h(X, edge_index, edge_weight), H * R], axis=1)
        H_tilde = self.linear_h(H_tilde)
        H_tilde = torch.tanh(H_tilde)
        return H_tilde

    def _calculate_hidden_state(self, Z, H, H_tilde):
        H = Z * H + (1 - Z) * H_tilde
        return H

    # # test workload variance
    # def forward(
    #     self,
    #     args,
    #     X: torch.FloatTensor,
    #     edge_index: torch.LongTensor,
    #     edge_weight: torch.FloatTensor = None,
    #     H: torch.FloatTensor = None,
    # ) -> torch.FloatTensor:
    #     """
    #     Making a forward pass. If edge weights are not present the forward pass
    #     defaults to an unweighted graph. If the hidden state matrix is not present
    #     when the forward pass is called it is initialized with zeros.
    #     Arg types:
    #         * **args* - include kvstore_client and other metric
    #         * **graph** *(PyTorch Geometirc Data)* - graph
    #         * **X** *(PyTorch Float Tensor)* - Node features.
    #         * **edge_index** *(PyTorch Long Tensor)* - Graph edge indices.
    #         * **edge_weight** *(PyTorch Long Tensor, optional)* - Edge weight vector.
    #         * **H** *(PyTorch Float Tensor, optional)* - Hidden state matrix for all nodes.
    #     Return types:
    #         * **H** *(PyTorch Float Tensor)* - Hidden state matrix for all nodes.
    #     """

    #     H = self._set_hidden_state(X, H)
    #     start_time = time.time()
    #     Z = self._calculate_update_gate(X, edge_index, edge_weight, H) # parallel GCN-1
    #     R = self._calculate_reset_gate(X, edge_index, edge_weight, H) # parallel GCN-2
    #     H_tilde = self._calculate_candidate_state(X, edge_index, edge_weight, H, R) # paralel GCN-3
    #     args['spatial_comp_time'] += time.time() - start_time
    #     start_time = time.time()
    #     H = self._calculate_hidden_state(Z, H, H_tilde) # GRU
    #     args['temporal_comp_time'] += time.time() - start_time
    #     return H

    # test stale aggregation
    def forward(
        self,
        X: torch.FloatTensor,
        edge_index: torch.LongTensor,
        edge_weight: torch.FloatTensor = None,
        H: torch.FloatTensor = None,
    ) -> torch.FloatTensor:
        """
        Making a forward pass. If edge weights are not present the forward pass
        defaults to an unweighted graph. If the hidden state matrix is not present
        when the forward pass is called it is initialized with zeros.
        Arg types:
            * **args* - include kvstore_client and other metric
            * **graph** *(PyTorch Geometirc Data)* - graph
            * **X** *(PyTorch Float Tensor)* - Node features.
            * **edge_index** *(PyTorch Long Tensor)* - Graph edge indices.
            * **edge_weight** *(PyTorch Long Tensor, optional)* - Edge weight vector.
            * **H** *(PyTorch Float Tensor, optional)* - Hidden state matrix for all nodes.
        Return types:
            * **H** *(PyTorch Float Tensor)* - Hidden state matrix for all nodes.
        """

        H = self._set_hidden_state(X, H)
        Z = self._calculate_update_gate(X, edge_index, edge_weight, H) # parallel GCN-1
        R = self._calculate_reset_gate(X, edge_index, edge_weight, H) # parallel GCN-2
        H_tilde = self._calculate_candidate_state(X, edge_index, edge_weight, H, R) # paralel GCN-3
        H = self._calculate_hidden_state(Z, H, H_tilde) # GRU
        return H


    def forward_partition(
        self,
        args,
        graphs,
        # X: torch.FloatTensor,
        # edge_index: torch.LongTensor,
        edge_weight: torch.FloatTensor = None,
        H: torch.FloatTensor = None,
    ) -> torch.FloatTensor:
        """
        Making a forward pass. If edge weights are not present the forward pass
        defaults to an unweighted graph. If the hidden state matrix is not present
        when the forward pass is called it is initialized with zeros.
        Arg types:
            * **args* - include kvstore_client and other metric
            * **graph** *(PyTorch Geometirc Data)* - graph
            * **X** *(PyTorch Float Tensor)* - Node features.
            * **edge_index** *(PyTorch Long Tensor)* - Graph edge indices.
            * **edge_weight** *(PyTorch Long Tensor, optional)* - Edge weight vector.
            * **H** *(PyTorch Float Tensor, optional)* - Hidden state matrix for all nodes.
        Return types:
            * **H** *(PyTorch Float Tensor)* - Hidden state matrix for all nodes.
        """
        H_ini_list = []
        R_list = []
        Z_list = []
        H_tilde_list = []
        H_list = []

        # # fusion
        # if len(graphs) > 10:
        #     X_list = []
        #     edge_index_list = []
        #     num_fused_graph = 3
        #     graphs_per_fuion = len(graphs)//num_fused_graph + min(1, len(graphs)%num_fused_graph)
        #     for i in range(num_fused_graph):
        #         x = torch.cat([graph.x for graph in graphs[i*graphs_per_fuion: min(len(graphs)-1, (i+1)*graphs_per_fuion)]], dim=0)
        #         edge = torch.cat([graph.edge_index for graph in graphs[i*graphs_per_fuion: min(len(graphs)-1, (i+1)*graphs_per_fuion)]], dim=1)
        #         X_list.append(x)
        #         edge_index_list.append(edge)
        # else:
        X_list = [graph.x for graph in graphs]
        edge_index_list = [graph.edge_index for graph in graphs]
        
        # aggregate remote spatial neighbors
        start_time = time.time()
        need_nodes = args['spatial_nodes_in_other_gpu']
        remote_spatial_emb = pull_tensors(args['kvstore_client'], layer=0, keys=need_nodes)
        args['spatial_comm_time'] += time.time() - start_time

        # computation in the structure encoder layer
        for i, x in enumerate(X_list):
            start_time = time.time()
            H_ini = self._set_hidden_state(X_list[i], H)
            Z = self._calculate_update_gate(X_list[i], edge_index_list[i], edge_weight, H_ini)
            R = self._calculate_reset_gate(X_list[i], edge_index_list[i], edge_weight, H_ini)
            args['spatial_comp_time'] += time.time() - start_time
            H_ini_list.append(H_ini)
            Z_list.append(Z)
            R_list.append(R)

        # send remote spatial neighbors
        start_time = time.time()
        send_nodes = args['spatial_nodes_for_other_gpu']
        push_tensors(args['kvstore_client'], layer=1, keys=send_nodes, values=torch.cat(H_ini_list, dim=0))
        push_tensors(args['kvstore_client'], layer=1, keys=send_nodes, values=torch.cat(R_list, dim=0))
        args['spatial_comm_time'] += time.time() - start_time
        torch.distributed.barrier()

        # aggregate remote spatial neighbors
        start_time = time.time()
        need_nodes = args['spatial_nodes_in_other_gpu']
        remote_spatial_emb = pull_tensors(args['kvstore_client'], layer=1, keys=need_nodes)
        args['spatial_comm_time'] += time.time() - start_time

        for i, x in enumerate(X_list):
            start_time = time.time()
            H_tilde = self._calculate_candidate_state(X_list[i], edge_index_list[i], edge_weight, H_ini_list[i], R_list[i])
            args['spatial_comp_time'] += time.time() - start_time
            H_tilde_list.append(H_tilde)

        # send remote spatial neighbors
        start_time = time.time()
        send_nodes = args['temporal_nodes_for_other_gpu']
        push_tensors(args['kvstore_client'], layer=2, keys=send_nodes, values=torch.cat(H_tilde_list, dim=0))
        args['temporal_comm_time'] += time.time() - start_time
        torch.distributed.barrier()

        # aggregate remote spatial neighbors
        start_time = time.time()
        need_nodes = args['temporal_nodes_in_other_gpu']
        remote_spatial_emb = pull_tensors(args['kvstore_client'], layer=2, keys=need_nodes)
        args['temporal_comm_time'] += time.time() - start_time

        for i, x in enumerate(X_list):
            start_time = time.time()
            H = self._calculate_hidden_state(Z_list[i], H_ini_list[i], H_tilde_list[i])
            args['temporal_comp_time'] += time.time() - start_time
            H_list.append(H)

        # H = self._set_hidden_state(X, H)
        # Z = self._calculate_update_gate(X, edge_index, edge_weight, H)
        # R = self._calculate_reset_gate(X, edge_index, edge_weight, H)
        # H_tilde = self._calculate_candidate_state(X, edge_index, edge_weight, H, R)
        # H = self._calculate_hidden_state(Z, H, H_tilde)
        return H_list


class TGCN2(torch.nn.Module):
    r"""An implementation THAT SUPPORTS BATCHES of the Temporal Graph Convolutional Gated Recurrent Cell.
    For details see this paper: `"T-GCN: A Temporal Graph ConvolutionalNetwork for
    Traffic Prediction." <https://arxiv.org/abs/1811.05320>`_
    Args:
        in_channels (int): Number of input features.
        out_channels (int): Number of output features.
        batch_size (int): Size of the batch.
        improved (bool): Stronger self loops. Default is False.
        cached (bool): Caching the message weights. Default is False.
        add_self_loops (bool): Adding self-loops for smoothing. Default is True.
    """

    def __init__(self, in_channels: int, out_channels: int, 
                 batch_size: int,  # this entry is unnecessary, kept only for backward compatibility
                 improved: bool = False, cached: bool = False, 
                 add_self_loops: bool = True):
        super(TGCN2, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.improved = improved
        self.cached = cached
        self.add_self_loops = add_self_loops
        self.batch_size = batch_size  # not needed
        self._create_parameters_and_layers()

    def _create_update_gate_parameters_and_layers(self):
        self.conv_z = GCNConv(in_channels=self.in_channels,  out_channels=self.out_channels, improved=self.improved,
                              cached=self.cached, add_self_loops=self.add_self_loops )
        self.linear_z = torch.nn.Linear(2 * self.out_channels, self.out_channels)

    def _create_reset_gate_parameters_and_layers(self):
        self.conv_r = GCNConv(in_channels=self.in_channels, out_channels=self.out_channels, improved=self.improved,
                              cached=self.cached, add_self_loops=self.add_self_loops )
        self.linear_r = torch.nn.Linear(2 * self.out_channels, self.out_channels)

    def _create_candidate_state_parameters_and_layers(self):
        self.conv_h = GCNConv(in_channels=self.in_channels, out_channels=self.out_channels, improved=self.improved,
                              cached=self.cached, add_self_loops=self.add_self_loops )
        self.linear_h = torch.nn.Linear(2 * self.out_channels, self.out_channels)

    def _create_parameters_and_layers(self):
        self._create_update_gate_parameters_and_layers()
        self._create_reset_gate_parameters_and_layers()
        self._create_candidate_state_parameters_and_layers()

    def _set_hidden_state(self, X, H):
        if H is None:
            # can infer batch_size from X.shape, because X is [B, N, F]
            H = torch.zeros(X.shape[0], X.shape[1], self.out_channels).to(X.device) #(b, 207, 32)
        return H

    def _calculate_update_gate(self, X, edge_index, edge_weight, H):
        Z = torch.cat([self.conv_z(X, edge_index, edge_weight), H], axis=2) # (b, 207, 64)
        Z = self.linear_z(Z) # (b, 207, 32)
        Z = torch.sigmoid(Z)

        return Z

    def _calculate_reset_gate(self, X, edge_index, edge_weight, H):
        R = torch.cat([self.conv_r(X, edge_index, edge_weight), H], axis=2) # (b, 207, 64)
        R = self.linear_r(R) # (b, 207, 32)
        R = torch.sigmoid(R)

        return R

    def _calculate_candidate_state(self, X, edge_index, edge_weight, H, R):
        H_tilde = torch.cat([self.conv_h(X, edge_index, edge_weight), H * R], axis=2) # (b, 207, 64)
        H_tilde = self.linear_h(H_tilde) # (b, 207, 32)
        H_tilde = torch.tanh(H_tilde)

        return H_tilde

    def _calculate_hidden_state(self, Z, H, H_tilde):
        H = Z * H + (1 - Z) * H_tilde   # # (b, 207, 32)
        return H

    def forward(self,X: torch.FloatTensor, edge_index: torch.LongTensor, edge_weight: torch.FloatTensor = None,
                H: torch.FloatTensor = None ) -> torch.FloatTensor:
        """
        Making a forward pass. If edge weights are not present the forward pass
        defaults to an unweighted graph. If the hidden state matrix is not present
        when the forward pass is called it is initialized with zeros.
        Arg types:
            * **X** *(PyTorch Float Tensor)* - Node features.
            * **edge_index** *(PyTorch Long Tensor)* - Graph edge indices.
            * **edge_weight** *(PyTorch Long Tensor, optional)* - Edge weight vector.
            * **H** *(PyTorch Float Tensor, optional)* - Hidden state matrix for all nodes.
        Return types:
            * **H** *(PyTorch Float Tensor)* - Hidden state matrix for all nodes.
        """
        H = self._set_hidden_state(X, H)
        Z = self._calculate_update_gate(X, edge_index, edge_weight, H)
        R = self._calculate_reset_gate(X, edge_index, edge_weight, H)
        H_tilde = self._calculate_candidate_state(X, edge_index, edge_weight, H, R)
        H = self._calculate_hidden_state(Z, H, H_tilde) # (b, 207, 32)
        return H