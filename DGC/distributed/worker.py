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

from DGC.nn import (DySAT, TGCN, MPNNLSTM, GCLSTM, Classifier)
from DGC.utils import (set_seed, embedding_distance)
from DGC.distributed.kvstore import (KVStoreServer, KVStoreClient)
from DGC.distributed.utils import (get_remote_neighbors, get_local_belong_remote_neighbors,
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
            if i == 0:
                out = self.recurrent(graphs[i].x.to(args['device']), graphs[i].edge_index.to(args['device']))
                outs.append(out)
            else:
                # pad zeros
                padding_zeros = torch.zeros(graphs[i].x.size(0) - graphs[i-1].x.size(0), out[i-1].size(0), dtype=torch.float).to(outs[i-1].device)
                if cache_out != None:
                    H = torch.cat([cache_out[i-1], padding_zeros], dim=0)
                else:
                    H = torch.cat([outs[i-1], padding_zeros], dim=0)
                out = self.recurrent(X=graphs[i].x.to(args['device']), edge_index=graphs[i].edge_index.to(args['device']), H=H)
                outs.append(out)

            # get target embeddings
            source_id = samples[:, 0]
            target_id = samples[:, 1]
            source_emb = out[source_id]
            target_emb = out[target_id]
            input_emb = source_emb.mul(target_emb)
            pred_y.append(self.classifier(input_emb))

        return pred_y, outs

def _run_partition(args, local_data):
    # define metric
    args['spatial_comm_time'] = 0
    args['spatial_comp_time'] = 0
    args['temporal_comm_time'] = 0
    args['temporal_comp_time'] = 0

    args['device'] = 'cpu'

    # push local belong remote neighbors to kvstore_server
    start_time = time.time()
    push_tensors(args['kvstore_client'], layer=0, keys=args['spatial_nodes_for_other_gpu'], values=torch.cat([data.x for data in local_data], dim=0))
    logging.info("rank: {} pushes node embeddings of layer {} to the kvstore server! time: {}".format(args['rank'], 0, time.time() - start_time))
    args['spatial_comm_time'] += time.time() - start_time
    torch.distributed.barrier()

    # define model
    start_time = time.time()
    # model = DySAT(args, num_features=2).to(args['device'])
    if args['model'] == 'MPNN-LSTM':
        model = MPNNLSTM(in_channels=2, hidden_size=16, window=1, dropout=0.5).to(args['device'])
    elif args['model'] == 'TGCN':
        model = TGCN(in_channels=2, out_channels=16).to(args['device'])
    elif args['model'] == 'GC-LSTM':
        model = GCLSTM(in_channels=2, out_channels=32, K=3).to(args['device'])
    else:
        raise Exception('There is no such an model!')
    logging.info("rank: {} loads model to device {}! time: {}".format(args['rank'], args['device'], time.time() - start_time))
    mem = psutil.virtual_memory().percent
    args['logger'].info("memory usage: {}".format(mem))

    # define optimizer
    # loss_func = torch.nn.CrossEntropyLoss()
    loss_func = torch.nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    # test communication
    for epoch in range(10):
        outs = model.forward_partition(args=args, graphs=local_data)

    args['logger'].info("rank: {} structure encoder computation time {:.3f} communication time {:.3f}".format(args['rank'], 
                                                                                                                    args['spatial_comp_time'], args['spatial_comm_time']))
    args['logger'].info("rank: {} time encoder computation time {:.3f} communication time {:.3f}".format(args['rank'], 
                                                                                                                args['temporal_comp_time'], args['temporal_comm_time']))


def _run_fusion(args, local_data):
    # define metric
    args['spatial_comm_time'] = 0
    args['spatial_comp_time'] = 0
    args['temporal_comm_time'] = 0
    args['temporal_comp_time'] = 0

    args['device'] = torch.device("cuda", 1)

    # push local belong remote neighbors to kvstore_server
    start_time = time.time()
    push_tensors(args['kvstore_client'], layer=0, keys=args['spatial_nodes_for_other_gpu'], values=torch.cat([data.x for data in local_data], dim=0))
    logging.info("rank: {} pushes node embeddings of layer {} to the kvstore server! time: {}".format(args['rank'], 0, time.time() - start_time))
    args['spatial_comm_time'] += time.time() - start_time
    torch.distributed.barrier()

    # define model
    start_time = time.time()
    # model = DySAT(args, num_features=2).to(args['device'])
    if args['model'] == 'MPNN-LSTM':
        model = MPNNLSTM(in_channels=2, hidden_size=16, window=1, dropout=0.5).to(args['device'])
    elif args['model'] == 'TGCN':
        model = TGCN(in_channels=2, out_channels=16).to(args['device'])
    elif args['model'] == 'GC-LSTM':
        model = GCLSTM(in_channels=2, out_channels=32, K=3).to(args['device'])
    else:
        raise Exception('There is no such an model!')
    logging.info("rank: {} loads model to device {}! time: {}".format(args['rank'], args['device'], time.time() - start_time))
    mem = psutil.virtual_memory().percent
    args['logger'].info("memory usage: {}".format(mem))

    # test fusion
    X_list = []
    edge_index_list = []
    if len(local_data) > 500:
        num_fused_graph = 500
        graphs_per_fuion = len(local_data)//num_fused_graph + min(1, len(local_data)%num_fused_graph)
        for i in range(num_fused_graph):
            x_temp = [torch.zeros(graph.x.size(0), 2, dtype=torch.float32) for graph in local_data[i*graphs_per_fuion: min(len(local_data)-1, (i+1)*graphs_per_fuion)]]
            if len(x_temp) > 0:
                x = torch.cat(x_temp, dim=0)
                edge = torch.cat([graph.edge_index for graph in local_data[i*graphs_per_fuion: min(len(local_data)-1, (i+1)*graphs_per_fuion)]], dim=1)
                X_list.append(x)
                edge_index_list.append(edge)
    else:
        x = torch.cat([torch.zeros(graph.x.size(0), 2, dtype=torch.float32) for graph in local_data], dim=0)
        edge = torch.cat([graph.edge_index for graph in local_data], dim=1)
        X_list.append(x)
        edge_index_list.append(edge)
    # X_list = [graph.x for graph in local_data]
    # edge_index_list = [graph.edge_index for graph in local_data]
    import wandb
    run = wandb.init(
        project=f"DGNN_GPU_utilization_{args['rank']}", entity="fahao", name='{}_wo_fusion'.format(args['dataset']), reinit=True,
        settings=wandb.Settings(
        _stats_sample_rate_seconds=0.1,
        _stats_samples_to_average=1,
        ))
    args['data_loading_time'] = 0
    X_list_new = []
    edge_index_list_new = []
    for epoch in range(100):
        X_list_new.extend(X_list)
        edge_index_list_new.extend(edge_index_list)
    for i, x in enumerate(X_list_new):
        out = model(args=args, X=X_list_new[i], edge_index=edge_index_list_new[i])
    run.finish()
    args['logger'].info("rank: {} structure encoder computation time {:.3f} communication time {:.3f}".format(args['rank'], 
                                                                                                        args['spatial_comp_time'], args['spatial_comm_time']))
    args['logger'].info("rank: {} time encoder computation time {:.3f} communication time {:.3f}".format(args['rank'], 
                                                                                                        args['temporal_comp_time'], args['temporal_comm_time']))


def _run_stale(args, local_data):
    # define metric
    args['spatial_comm_time'] = 0
    args['spatial_comp_time'] = 0
    args['temporal_comm_time'] = 0
    args['temporal_comp_time'] = 0

    args['device'] = torch.device("cuda", 1)

    # push local belong remote neighbors to kvstore_server
    start_time = time.time()
    # push_all_tensors(kvstore_client=args['kvstore_client'], layer=0, graphs=local_data, 
    #                  values=[data.x for data in local_data], push_type='spatial')
    push_tensors(args['kvstore_client'], layer=0, keys=args['spatial_nodes_for_other_gpu'], values=torch.cat([data.x for data in local_data], dim=0))
    # logging.info("rank: {} pushes node embeddings of layer {} to the kvstore server! time: {}".format(rank, 0, time.time() - start_time))
    args['spatial_comm_time'] += time.time() - start_time
    torch.distributed.barrier()

    # test stale aggregation, 使用pss方法分图，更方便进行训练
    models = ['TGCN','GC-LSTM','MPNN-LSTM']
    for model_name in models:
        for scale in [0.3, 0.5, 0.7]:
            model = RecurrentGCN(node_features=2, model=model_name).to(args['device'])
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
                # update caching
                if epoch == 0:
                    outputs, embs = model(args, local_data, args['train_samples_list'])
                    caching_content = [embs[i].detach().clone() for i in range(len(embs))]
                else:
                    outputs, embs = model(args, local_data, args['train_samples_list'], caching_content)
                    distances = [embedding_distance(caching_content[i], embs[i]) for i in range(len(embs))]
                    for i in range(len(embs)):
                        distance = distances[i]
                        avg_distance = torch.mean(distance)
                        max_distance = torch.max(distance)
                        normalized_distance = distance/max_distance
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

                # print(outputs)
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
            args['logger'].info("model: {} threshold {} best accuracy {:.3f}, total embeddings {} saved embeddings {}".format(model_name, scale, best_acc, total_embs, saved_embs))
    

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
            # kv_server = shared_dict['kv_server']
            # shared_dict['kv_server'].rank = 1
        else:
            rpc.init_rpc(
                name=f"worker_{rank}",
                rank=rank,
                world_size=args['world_size'],
                rpc_backend_options=rpc.TensorPipeRpcBackendOptions(
                init_method=dist_init_method)
            )
            # kv_server = rpc.rpc_sync("server_0", KVStoreServer, args=('DySAT', rank))
        # kvstore_server_pub= args['kvstore_server']
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
    if args['experiment'] == 'partition':
        _run_partition(args, args['local_data'])
    elif args['experiment'] == 'fusion':
        _run_fusion(args, args['local_data'])
    elif args['experiment'] == 'stale':
        _run_stale(args, args['local_data'])
    else:
        raise Exception('There is no such an experiment!')

    logging.info('End!\n')
    rpc.shutdown()