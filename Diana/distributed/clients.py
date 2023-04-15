import logging
import time
import datetime
import psutil
import math
import torch
import numpy as np
import os, sys
sys.path.append("..") 

import torch.distributed.rpc as rpc
import torch.nn.functional as F

from Diana.nn import (DySAT, TGCN, MPNNLSTM, GCLSTM, Classifier)
from Diana.utils import (set_seed, embedding_distance)
from Diana.distributed.kvstore import (KVStoreServer, KVStoreClient)
from Diana.distributed.utils import (get_remote_neighbors, get_local_belong_remote_neighbors,
                                     push_all_tensors, push_tensors)

def LocalTimeFormatter(sec, what):
    local_time = datetime.datetime.now() + datetime.timedelta(hours=9)
    return local_time.timetuple()

class RecurrentGCN(torch.nn.Module):
    def __init__(self, node_features, model):
        super(RecurrentGCN, self).__init__()
        if model == 'MPNN-LSTM':
            self.recurrent = MPNNLSTM(in_channels=2, hidden_size=32, window=1, dropout=0.5)
            self.classifier = Classifier(input_dim=66, hidden_dim=128)
        elif model == 'TGCN':
            self.recurrent = TGCN(in_channels=2, out_channels=32)
            self.classifier = Classifier(input_dim=32, hidden_dim=128)
        elif model == 'GC-LSTM':
            self.recurrent = GCLSTM(in_channels=2, out_channels=32, K=3)
            self.classifier = Classifier(input_dim=32, hidden_dim=128)
        else:
            raise Exception('There is no such an model!')

    def forward(self, args, graphs, samples_list, cache_out = None):
        pred_y = []
        outs = []
        for i in range(len(graphs)):
            samples = samples_list[i]
            out = self.recurrent(args, graphs[i].x.to(args['device']), graphs[i].edge_index.to(args['device']))
            outs.append(out)
            
            # get target embeddings
            source_id = samples[:, 0]
            target_id = samples[:, 1]
            source_emb = out[source_id]
            target_emb = out[target_id]
            input_emb = source_emb.mul(target_emb)
            pred_y.append(self.classifier(input_emb))

        return pred_y, outs


def _train(args, local_data):
    # define metric
    args['spatial_comm_time'] = 0
    args['spatial_comp_time'] = 0
    args['temporal_comm_time'] = 0
    args['temporal_comp_time'] = 0

    # push locals (original features) belong remote neighbors to kvstore_server
    start_time = time.time()
    push_all_tensors(kvstore_client=args['kvstore_client'], layer=0, graphs=local_data, 
                     values=[data.x for data in local_data], push_type='spatial')
    push_tensors(args['kvstore_client'], layer=0, keys=args['spatial_nodes_for_other_gpu'], values=torch.cat([data.x for data in local_data], dim=0))
    args['spatial_comm_time'] += time.time() - start_time
    torch.distributed.barrier()

    # training loop
    scale=0.1 # for stale aggregation
    model = RecurrentGCN(node_features=2, model=args['model']).to(args['device'])
    loss_func = torch.nn.CrossEntropyLoss()
    # loss_func = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args['lr'], weight_decay=0.01)
    best_acc = 0
    caching_content = []
    saved_embs = 0
    loss_max = 0
    loss_pre = 0
    total_embs = 0
    for epoch in range(150):
        # train
        model.train()
        loss = 0
        if args['stale']:
            # update caching
            if epoch == 0:
                outputs, embs = model(args, local_data, args['train_samples_list'])
                caching_content = [embs[i].detach().clone() for i in range(len(embs))]
            else:
                outputs, embs = model(args, local_data, args['train_samples_list'], caching_content)
                distances = [embedding_distance(caching_content[i], embs[i]) for i in range(len(embs))]
                for i in range(len(embs)):
                    distance = distances[i]
                    max_distance = torch.max(distance)
                    # static threshold
                    threshold = scale*max_distance
                    # dynamic threshold
                    # if epoch > 20:
                    #     threshold = 1/(1+math.exp((loss_max - loss_pre)/loss_max)) * max_distance
                    # else:
                    #     threshold = 0
                    smaller_distance_str = torch.nonzero(distance < threshold, as_tuple=False).view(-1)
                    greater_distance_str = torch.nonzero(distance >= threshold, as_tuple=False).view(-1)
                    caching_content[i][greater_distance_str, :] = embs[i][greater_distance_str, :].detach().clone()
                    saved_embs += smaller_distance_str.size(0)
                    total_embs += embs[i].size(0)
        else:
            outputs, _ = model(args, local_data, args['train_samples_list'])
        pred_y = torch.cat(outputs, dim=0)
        labels = torch.cat(args['train_labels_list'], dim=0).to(args['device'])
        loss = loss_func(pred_y.squeeze(dim=-1), labels)
        if loss_max <= loss.item():
            loss_max = loss.item()
        loss_pre = loss.item()
        loss.backward()
        optimizer.step()
        gpu_mem_alloc = torch.cuda.max_memory_allocated() / 1000000 if torch.cuda.is_available() else 0
        optimizer.zero_grad()

        # test
        with torch.no_grad():
            model.eval()
            ACC = 0
            outputs, _ = model(args, local_data, args['test_samples_list'])
            for i, _ in enumerate(local_data):
                y = outputs[i]
                label =args['test_labels_list'][i].cpu().numpy()
                prob_f1 = []
                prob_f1.extend(np.argmax(y.detach().cpu().numpy(), axis = 1))
                ACC += sum(prob_f1 == label)/len(label)
            acc = ACC/len(local_data)
            if best_acc <= acc:
                best_acc = acc
        print('epoch: {} loss: {:.4f} acc: {:.4f} GPU memory {:.3f}'.format(epoch, loss.item(), acc, gpu_mem_alloc))

    args['logger'].info("rank: {} structure encoder computation time {:.3f} communication time {:.3f}".format(args['rank'], 
                                                                                                            args['spatial_comp_time'], args['spatial_comm_time']))
    args['logger'].info("rank: {} time encoder computation time {:.3f} communication time {:.3f}".format(args['rank'], 
                                                                                                        args['temporal_comp_time'], args['temporal_comm_time']))
    args['logger'].info("model: {} threshold {} best accuracy {:.3f}, total embeddings {} saved embeddings {}".format(args['model'], scale, best_acc, total_embs, saved_embs))

def work(rank, args, other_args, shared_dict):
    # environment settings
    current_path = os.path.abspath(os.path.join(os.getcwd(), "../"))
    log_file = current_path + '/log/test_example_work_{}.log'.format(rank)
    logging.basicConfig(filename=log_file, format='%(asctime)s %(levelname)s: %(message)s', level=logging.DEBUG)
    logging.Formatter.converter = LocalTimeFormatter
    logger = logging.getLogger('example')
    set_seed(args['seed'])
    args['device'] = torch.device("cuda", rank)
    # args['device'] = 'cpu'
    logging.info(f"Rank {rank} started with args: {args}")

    # get data information
    args['all_datas'] = other_args[0]
    args['spatial_edge_index'] = other_args[1]
    args['temporal_edge_index'] = other_args[2]
    args['train_samples_list'] = other_args[3]
    args['test_samples_list'] = other_args[4]
    args['train_labels_list'] = other_args[5]
    args['test_labels_list'] = other_args[6]
    args['data_gpu_map'] = other_args[7]

    # set envs
    if args['distributed']:
        # args['local_world_size'] = int(os.environ['LOCAL_WORLD_SIZE'])
        # args['world_size'] = args['world_size']
        torch.cuda.set_device(rank)
        # torch.distributed.init_process_group('nccl')
        dist_init_method = 'tcp://{master_ip}:{master_port}'.format(
            master_ip='127.0.0.1', master_port='12352')
        torch.distributed.init_process_group(
            'gloo', timeout=datetime.timedelta(seconds=36000),
            world_size = args['world_size'], rank = rank,
            init_method = dist_init_method)
        args['world_size'] = torch.distributed.get_world_size()

        # register kvstore and rpc environment
        if rank == 0:
            rpc.init_rpc(
            name=f"server_{rank}",
            rank=rank,
            world_size=args['world_size'],
            rpc_backend_options=rpc.TensorPipeRpcBackendOptions(
                init_method=dist_init_method)
            )
            # rpc.shutdown()
            kv_server = KVStoreServer(args['model'], rank)
            shared_dict['kv_server'] = kv_server
        else:
            rpc.init_rpc(
                name=f"worker_{rank}",
                rank=rank,
                world_size=args['world_size'],
                rpc_backend_options=rpc.TensorPipeRpcBackendOptions(
                init_method=dist_init_method)
            )
        torch.distributed.barrier()
        kv_server = shared_dict['kv_server']
        kvstore_client = KVStoreClient(server_name='server_0', module_name=kv_server)
    else:
        args['local_world_size'] = 1
        args['world_size'] = 1

    args['kvstore_client'] = kvstore_client
    args['logger'] = logger
    args['rank'] = rank
    
    data_gpu_map = args['data_gpu_map']
    args['local_data'] = data_gpu_map[rank]
    all_local_nodes = [data.local_node_index for data in args['local_data']]
    args['all_local_nodes'] = torch.cat(all_local_nodes, dim=0)

    args['spatial_nodes_in_other_gpu'] = get_remote_neighbors(args['all_local_nodes'], args['num_nodes'], args['spatial_edge_index'])
    args['temporal_nodes_in_other_gpu'] = get_remote_neighbors(args['all_local_nodes'], args['num_nodes'], args['temporal_edge_index'])
    args['spatial_nodes_for_other_gpu'] = get_local_belong_remote_neighbors(args['all_local_nodes'], args['spatial_edge_index'])
    args['temporal_nodes_for_other_gpu'] = get_local_belong_remote_neighbors(args['all_local_nodes'], args['temporal_edge_index'])

    logging.info("rank: {} get {} chunks".format(rank, len(data_gpu_map[rank])))

    start_time = time.time()
    _train(args, args['local_data'])
    print(f'training time cost {time.time() - start_time}')

    logging.info('End!\n')
    rpc.shutdown()