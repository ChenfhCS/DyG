import torch
import torch.distributed.rpc as rpc
import os
# 设置环境变量
 # @staticmethod
    # def _rref(self):
    #     return rpc.RRef(self)
class KVStoreServer:
    """
    Simple key-value store server that stores tensors.
    """
    _data = []
    def __init__(self, model_type, rank=None):
        self.rank = rank
        # self._data = []
        self.initialization(model_type)
    

    def initialization(self, model_type):
        if model_type == 'MPNN-LSTM':
            self._data.append({})  # original from gcn layer
            self._data.append({})  # activations from gcn layer
            self._data.append({})  # activations from lstm layer
            self._data.append({})  # activations from lstm layer
        if model_type == 'TGCN':
            self._data.append({})  # original node features
            self._data.append({})  # activations from gcn layer
            self._data.append({})  # activations from lstm layer
        if model_type == 'GC-LSTM':
            self._data.append({})  # original node features
            self._data.append({})  # activations from gcn layer
            self._data.append({})  # activations from lstm layer
        else:
            return 0

    def push(self, layer, nodes, tensors):
        """
        Push a tensor to the server.
        Args:
            key: layer index
            node: node index
            tensor: node emb
        """
        self._data[layer].update(dict(zip(nodes.tolist(), tensors)))

    def pull(self, layer, nodes):
        """
        Pull a tensor from the server.
        Args:
            key: layer index
            node: node index
            tensor: node emb
        Returns:
            The tensor associated with the key, or None if the key is not found.
        """
        return torch.cat([self._data[layer][key.item()].unsqueeze(0) for key in nodes], dim=0)

    def get_data(self):
        return self._data

    def get_rank(self):
        return self.rank

    def shutdown(self):
        """
        Shutdown the RPC server.
        """
        rpc.shutdown()


class KVStoreClient:
    """
    Simple key-value store client that stores tensors.
    """

    def __init__(self, server_name, module_name):
        self._server_name = server_name
        self._module_name = module_name

    def push(self, layer, nodes, tensors):
        """
        Push a tensor to the server.
        Args:
            key: The key associated with the tensor.
            tensor: The tensor to push.
        """
        # self._server.rpc_sync().push(key, node, tensor)
        rpc.rpc_sync(self._server_name, self._module_name.push, args=(layer, nodes, tensors))

    def pull(self, key, node):
        """
        Pull a tensor from the server.
        Args:
            key: The key of the tensor to pull.
        Returns:
            The tensor associated with the key, or None if the key is not found.
        """
        # return self._server.rpc_sync().pull(key, node)
        mes = rpc.rpc_sync(self._server_name, self._module_name.pull, args=(key, node))
        return mes

    def get_rank(self):
        """
        Pull a tensor from the server.
        Args:
            key: The key of the tensor to pull.
        Returns:
            The tensor associated with the key, or None if the key is not found.
        """
        # return self._server.rpc_sync().pull(key, node)
        mes = rpc.rpc_sync(self._server_name, self._module_name.get_rank, args=())
        return mes

    def get_data(self):
        mes = rpc.rpc_sync(self._server_name, self._module_name.get_data, args=())
        return mes

    def shutdown(self):
        """
        Shutdown the RPC client.
        """
        rpc.shutdown()