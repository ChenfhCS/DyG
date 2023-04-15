import torch
import torch.nn as nn
import torch.nn.functional as F
import time

from torch_geometric.nn import GCNConv

from Diana.distributed.utils import (pull_all_tensors, push_all_tensors,
                                     pull_tensors, push_tensors)

class MPNNLSTM(nn.Module):
    r"""An implementation of the Message Passing Neural Network with Long Short Term Memory.
    For details see this paper: `"Transfer Graph Neural Networks for Pandemic Forecasting." <https://arxiv.org/abs/2009.08388>`_
    Args:
        in_channels (int): Number of input features.
        hidden_size (int): Dimension of hidden representations.
        num_nodes (int): Number of nodes in the network.
        window (int): Number of past samples included in the input.
        dropout (float): Dropout rate.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_size: int,
        window: int,
        dropout: float,
        num_nodes: int = None,
    ):
        super(MPNNLSTM, self).__init__()

        self.window = window
        self.num_nodes = num_nodes
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.in_channels = in_channels

        self._create_parameters_and_layers()

    def _create_parameters_and_layers(self):

        self._convolution_1 = GCNConv(self.in_channels, self.hidden_size)
        self._convolution_2 = GCNConv(self.hidden_size, self.hidden_size)

        self._batch_norm_1 = nn.BatchNorm1d(self.hidden_size)
        self._batch_norm_2 = nn.BatchNorm1d(self.hidden_size)

        self._recurrent_1 = nn.LSTM(2 * self.hidden_size, self.hidden_size, 1)
        self._recurrent_2 = nn.LSTM(self.hidden_size, self.hidden_size, 1)

    def _graph_convolution_1(self, X, edge_index, edge_weight):
        X = F.relu(self._convolution_1(X, edge_index, edge_weight))
        # X = self._batch_norm_1(X)
        X = F.dropout(X, p=self.dropout, training=self.training)
        return X

    def _graph_convolution_2(self, X, edge_index, edge_weight):
        X = F.relu(self._convolution_2(X, edge_index, edge_weight))
        # X = self._batch_norm_2(X)
        X = F.dropout(X, p=self.dropout, training=self.training)
        return X

    # # # test workload variance
    # def forward(
    #     self,
    #     args,
    #     X: torch.FloatTensor,
    #     edge_index: torch.LongTensor,
    #     edge_weight: torch.FloatTensor = None,
    #     H = None,
    # ) -> torch.FloatTensor:
    #     """
    #     Making a forward pass through the whole architecture.
    #     Arg types:
    #         * **args* - include kvstore_client and other metric
    #         * **graph** *(PyTorch Geometirc Data)* - graph
    #         * **X** *(PyTorch FloatTensor)* - Node features.
    #         * **edge_index** *(PyTorch LongTensor)* - Graph edge indices.
    #         * **edge_weight** *(PyTorch LongTensor, optional)* - Edge weight vector.
    #     Return types:
    #         *  **H** *(PyTorch FloatTensor)* - The hidden representation of size 2*nhid+in_channels+window-1 for each node.
    #     """
    #     R = list()
    #     num_nodes = X.size(0)
    
    #     S = X.view(-1, self.window, num_nodes, self.in_channels)
    #     S = torch.transpose(S, 1, 2)
    #     S = S.reshape(-1, self.window, self.in_channels)
    #     O = [S[:, 0, :]]

    #     for l in range(1, self.window):
    #         O.append(S[:, l, self.in_channels - 1].unsqueeze(1))

    #     S = torch.cat(O, dim=1)
        
    #     start_time = time.time()
    #     X = self._graph_convolution_1(X, edge_index, edge_weight)  # GCN-1, one spatial communication
    #     args['spatial_comp_time'] += time.time() - start_time
    #     R.append(X)

    #     start_time = time.time()
    #     X = self._graph_convolution_2(X, edge_index, edge_weight)  # GCN-2, one spatial communication
    #     args['spatial_comp_time'] += time.time() - start_time
    #     R.append(X)

    #     X = torch.cat(R, dim=1)

    #     X = X.view(-1, self.window, num_nodes, X.size(1))
    #     X = torch.transpose(X, 0, 1)
    #     X = X.contiguous().view(self.window, -1, X.size(3))

    #     start_time = time.time()
    #     X, (H_1, _) = self._recurrent_1(X)  # LSTM-1, one temporal communication
    #     X, (H_2, _) = self._recurrent_2(X)  # LSTM-2, one temporal communication
    #     args['temporal_comp_time'] += time.time() - start_time

    #     H = torch.cat([H_1[0, :, :], H_2[0, :, :], S], dim=1)
    #     return H


    # test stale aggregation
    def forward_stale(
        self,
        X: torch.FloatTensor,
        edge_index: torch.LongTensor,
        edge_weight: torch.FloatTensor = None,
        H = None,
    ) -> torch.FloatTensor:
        """
        Making a forward pass through the whole architecture.
        Arg types:
            * **args* - include kvstore_client and other metric
            * **graph** *(PyTorch Geometirc Data)* - graph
            * **X** *(PyTorch FloatTensor)* - Node features.
            * **edge_index** *(PyTorch LongTensor)* - Graph edge indices.
            * **edge_weight** *(PyTorch LongTensor, optional)* - Edge weight vector.
        Return types:
            *  **H** *(PyTorch FloatTensor)* - The hidden representation of size 2*nhid+in_channels+window-1 for each node.
        """
        R = list()
        num_nodes = X.size(0)
    
        S = X.view(-1, self.window, num_nodes, self.in_channels)
        S = torch.transpose(S, 1, 2)
        S = S.reshape(-1, self.window, self.in_channels)
        O = [S[:, 0, :]]

        for l in range(1, self.window):
            O.append(S[:, l, self.in_channels - 1].unsqueeze(1))

        S = torch.cat(O, dim=1)
        
        X = self._graph_convolution_1(X, edge_index, edge_weight)  # GCN-1, one spatial communication
        R.append(X)

        X = self._graph_convolution_2(X, edge_index, edge_weight)  # GCN-2, one spatial communication
        R.append(X)

        X = torch.cat(R, dim=1)

        X = X.view(-1, self.window, num_nodes, X.size(1))
        X = torch.transpose(X, 0, 1)
        X = X.contiguous().view(self.window, -1, X.size(3))

        X, (H_1, _) = self._recurrent_1(X)  # LSTM-1, one temporal communication
        X, (H_2, _) = self._recurrent_2(X)  # LSTM-2, one temporal communication

        H = torch.cat([H_1[0, :, :], H_2[0, :, :], S], dim=1)
        return H

    def forward(
        self,
        X: torch.FloatTensor,
        edge_index: torch.LongTensor,
        edge_weight: torch.FloatTensor = None,
        H = None,
    ) -> torch.FloatTensor:
        """
        Making a forward pass through the whole architecture.
        Arg types:
            * **args* - include kvstore_client and other metric
            * **graph** *(PyTorch Geometirc Data)* - graph
            * **X** *(PyTorch FloatTensor)* - Node features.
            * **edge_index** *(PyTorch LongTensor)* - Graph edge indices.
            * **edge_weight** *(PyTorch LongTensor, optional)* - Edge weight vector.
        Return types:
            *  **H** *(PyTorch FloatTensor)* - The hidden representation of size 2*nhid+in_channels+window-1 for each node.
        """
        R = list()
        num_nodes = X.size(0)
        
        S = X.view(-1, self.window, num_nodes, self.in_channels)
        S = torch.transpose(S, 1, 2)
        S = S.reshape(-1, self.window, self.in_channels)
        O = [S[:, 0, :]]

        for l in range(1, self.window):
            O.append(S[:, l, self.in_channels - 1].unsqueeze(1))

        S = torch.cat(O, dim=1)
        
        X = self._graph_convolution_1(X, edge_index, edge_weight)  # GCN-1, one spatial communication
        R.append(X)

        X = self._graph_convolution_2(X, edge_index, edge_weight)  # GCN-2, one spatial communication
        R.append(X)

        X = torch.cat(R, dim=1)

        X = X.view(-1, self.window, num_nodes, X.size(1))
        X = torch.transpose(X, 0, 1)
        X = X.contiguous().view(self.window, -1, X.size(3))

        X, (H_1, _) = self._recurrent_1(X)  # LSTM-1, one temporal communication
        X, (H_2, _) = self._recurrent_2(X)  # LSTM-2, one temporal communication

        H = torch.cat([H_1[0, :, :], H_2[0, :, :], S], dim=1)
        return H

    def forward_partition(
        self,
        args,
        graphs,
        # X: torch.FloatTensor,
        # edge_index: torch.LongTensor,
        edge_weight: torch.FloatTensor = None,
    ) -> torch.FloatTensor:
        """
        Making a forward pass through the whole architecture.
        Arg types:
            * **args* - include kvstore_client and other metric
            * **graph** *(PyTorch Geometirc Data)* - graph
            * **X** *(PyTorch FloatTensor)* - Node features.
            * **edge_index** *(PyTorch LongTensor)* - Graph edge indices.
            * **edge_weight** *(PyTorch LongTensor, optional)* - Edge weight vector.
        Return types:
            *  **H** *(PyTorch FloatTensor)* - The hidden representation of size 2*nhid+in_channels+window-1 for each node.
        """

        S_list = []
        R_list = [list() for i in range(len(graphs))]
        H_1_list = []
        H_2_list = []
        H_list = []


        X_list = [graph.x for graph in graphs]
        edge_index_list = [graph.edge_index for graph in graphs]

        for i, x in enumerate(X_list):
            S = X_list[i].view(-1, self.window, x.size(0), self.in_channels)
            S = torch.transpose(S, 1, 2)
            S = S.reshape(-1, self.window, self.in_channels)

            O = [S[:, 0, :]]
            for l in range(1, self.window):
                O.append(S[:, l, self.in_channels - 1].unsqueeze(1))
            S = torch.cat(O, dim=1)
            S_list.append(S)

        # aggregate remote spatial neighbors
        start_time = time.time()
        need_nodes = args['spatial_nodes_in_other_gpu']
        remote_spatial_emb = pull_tensors(args['kvstore_client'], layer=0, keys=need_nodes)
        args['spatial_comm_time'] += time.time() - start_time

        # computation in the structure encoder layer
        for i, x in enumerate(X_list):
            start_time = time.time()
            X_list[i] = self._graph_convolution_1(X_list[i], edge_index_list[i], edge_weight)
            args['spatial_comp_time'] += time.time() - start_time
            R_list[i].append(X_list[i])

        # send remote spatial neighbors
        start_time = time.time()
        send_nodes = args['spatial_nodes_for_other_gpu']
        push_tensors(args['kvstore_client'], layer=1, keys=send_nodes, values=torch.cat(X_list, dim=0))
        args['spatial_comm_time'] += time.time() - start_time
        torch.distributed.barrier()

        # aggregate remote spatial neighbors
        start_time = time.time()
        need_nodes = args['spatial_nodes_in_other_gpu']
        remote_spatial_emb = pull_tensors(args['kvstore_client'], layer=1, keys=need_nodes)
        args['spatial_comm_time'] += time.time() - start_time

        # computation in the structure encoder layer
        for i, x in enumerate(X_list):
            start_time = time.time()
            X_list[i] = self._graph_convolution_2(X_list[i], edge_index_list[i], edge_weight)
            args['spatial_comp_time'] += time.time() - start_time
            R_list[i].append(X_list[i])

        # send remote temporal neighbors
        start_time = time.time()
        send_nodes = args['temporal_nodes_for_other_gpu']
        push_tensors(args['kvstore_client'], layer=2, keys=send_nodes, values=torch.cat(X_list, dim=0))
        args['temporal_comm_time'] += time.time() - start_time
        torch.distributed.barrier()

        for i, x in enumerate(X_list):
            X_list[i] = torch.cat(R_list[i], dim=1)
            X_list[i] = X_list[i].view(-1, self.window,  x.size(0), X_list[i].size(1))
            X_list[i] = torch.transpose(X_list[i], 0, 1)
            X_list[i] = X_list[i].contiguous().view(self.window, -1, X_list[i].size(3))

        # aggregate remote temporal neighbors
        start_time = time.time()
        need_nodes = args['temporal_nodes_in_other_gpu']
        remote_spatial_emb = pull_tensors(args['kvstore_client'], layer=2, keys=need_nodes)
        args['temporal_comm_time'] += time.time() - start_time

        # computation of the time encoder layer
        for i, x in enumerate(X_list):
            start_time = time.time()
            X_list[i], (H_1, _) = self._recurrent_1(X_list[i])
            args['temporal_comp_time'] += time.time() - start_time
            H_1_list.append(H_1)
        
        # send remote temporal neighbors
        start_time = time.time()
        send_nodes = args['temporal_nodes_for_other_gpu']
        push_tensors(args['kvstore_client'], layer=3, keys=send_nodes, values=torch.cat([H_1.squeeze(0) for H_1 in H_1_list], dim=0))
        args['temporal_comm_time'] += time.time() - start_time
        torch.distributed.barrier()

        # aggregate remote temporal neighbors
        start_time = time.time()
        need_nodes = args['temporal_nodes_in_other_gpu']
        remote_spatial_emb = pull_tensors(args['kvstore_client'], layer=3, keys=need_nodes)
        args['temporal_comm_time'] += time.time() - start_time

        # computation of the time encoder layer
        for i, x in enumerate(X_list):
            start_time = time.time()
            X_list[i], (H_2, _) = self._recurrent_2(X_list[i])
            args['temporal_comp_time'] += time.time() - start_time
            H_2_list.append(H_2)
        
        for i, x in enumerate(X_list):
            H_list.append(torch.cat([H_1_list[i][0, :, :], H_2_list[i][0, :, :], S_list[i]], dim=1))

        return H_list
