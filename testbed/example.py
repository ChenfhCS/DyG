import torch
import argparse
import numpy as np
import warnings
import networkx as nx
import pandas as pd
import logging
import time
warnings.filterwarnings('ignore')

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = "1"
import sys
sys.path.append("..") 

from tqdm import tqdm
from dataset import AmazonDatasetLoader, EpinionDatasetLoader, MovieDatasetLoader, StackDatasetLoader
from nn import DySAT
from nn import classifier

from simulation.communication_sim import Simulator
# from simulation.Simulator_new import Diana
from MLDP import Partition_DyG, node_partition_balance

import boto3
import json
from concurrent.futures import ThreadPoolExecutor, as_completed, wait

def _warmup_lambda(lambda_client, function_name, num_warmup_invocations):
    for i in range(num_warmup_invocations):
        response = lambda_client.invoke(
            FunctionName=function_name,
            InvocationType='RequestResponse',
            Payload=json.dumps({'warmup': True})
        )

class My_Model(torch.nn.Module):
    def __init__(self, args, node_features):
        super(My_Model, self).__init__()
        self.args = args
        self.dgnn = DySAT(args, num_features = node_features)
        self.classifier = classifier(in_feature = 16)

    def forward(self, snapshots, samples):
        if self.args['testbed'] == 'cpu' or self.args['testbed'] == 'gpu':
            str_emb, final_emb = self.dgnn.forward(snapshots)
        elif self.args['testbed'] == 'lambda':
            str_emb, final_emb = self.dgnn.forward_lambda(snapshots)
        else:
            raise Exception('There is no such an device type to support!')
        outputs = []
        for t, snapshot in enumerate(snapshots):
            emb = final_emb[:, t, :].to(self.args['device'])

            sample = samples[t]
            # get target embeddings
            source_id = sample[:, 0]
            target_id = sample[:, 1]
            source_emb = emb[source_id]
            target_emb = emb[target_id]
            input_emb = source_emb.mul(target_emb)
            outputs.append(self.classifier(input_emb))
        return str_emb, final_emb, outputs

def _get_args():
    parser = argparse.ArgumentParser(description='example settings')
    
    # for experimental configurations
    parser.add_argument('--dataset', type=str, default='Amazon',
                        help='method for DGNN training')
    parser.add_argument('--timesteps', type=int, nargs='?', default=15,
                    help="total time steps used for train, eval and test")
    parser.add_argument('--epochs', type=int, nargs='?', default=50,
                    help="total number of epochs")
    parser.add_argument('--experiments', type=str, nargs='?', required=True,
                    help="experiment type")
    parser.add_argument('--world_size', type=int, default=2,
                        help='method for DGNN training')
    parser.add_argument('--testbed', type=str, default='cpu',
                        help='training testbed')
    args = vars(parser.parse_args())
    return args

def _get_partitions(snapshots):
    graphs = [snapshot.raw_graph for snapshot in snapshots]
    nodes_list = [torch.tensor([j for j in range(graphs[i].number_of_nodes())]) for i in range(len(snapshots))]
    adjs_list = []
    for i in range(len(snapshots)):
        adj_sp = nx.adjacency_matrix(graphs[i]).tocoo()
        adj = torch.sparse.LongTensor(torch.LongTensor([adj_sp.row.tolist(), adj_sp.col.tolist()]),
                            torch.LongTensor(adj_sp.data.astype(np.int32))).coalesce()
        adjs_list.append(adj)
    # partitioner = Partition_DyG(args, graphs, nodes_list, adjs_list, 2, 128, 128, float(1024*1024*8))
    # total_workload_gcn, total_workload_rnn = partitioner.get_partition()
    partitioner = node_partition_balance(args, graphs, nodes_list, adjs_list, 2)
    total_workload_gcn, total_workload_rnn = partitioner.get_partition()
    # Diana_obj = Diana(args, graphs, nodes_list, adjs_list, args['world_size'], 128*32, 128*32, float(1024*1024*8), logger=None)
    # Diana_obj.partitioning('LDG_base')
    # total_workload_gcn, total_workload_rnn = Diana_obj.get_partition()
    return total_workload_gcn, total_workload_rnn, adjs_list

def _save_log(args, loss_log, acc_log):
    folder_in = os.path.exists('./log/')
    if not folder_in:
        os.makedirs('./log/')
    df_loss=pd.DataFrame(data=loss_log)
    df_loss.to_csv('./log/{}_{}_loss.csv'.format(args['dataset'], args['timesteps']), header=False)
    df_acc=pd.DataFrame(data=acc_log)
    df_acc.to_csv('./log/{}_{}_acc.csv'.format(args['dataset'], args['timesteps']), header=False)

def run_example(args, logger):
    # print hyper-parameters
    args['rank'] = 0
    args['stale'] = False
    if args['testbed'] == 'cpu' or args['testbed'] == 'lambda':
        args['device'] = torch.device("cpu")
    elif args['testbed'] == 'gpu':
        args['device'] = torch.device("cuda")
    print(args)
    if args['dataset'] == 'Epinion':
        loader = EpinionDatasetLoader(timesteps = args['timesteps'])
    elif args['dataset'] == 'Amazon':
        loader = AmazonDatasetLoader(timesteps = args['timesteps'])
    elif args['dataset'] == 'Movie':
        loader = MovieDatasetLoader(timesteps = args['timesteps'])
    elif args['dataset'] == 'Stack':
        loader = StackDatasetLoader(timesteps = args['timesteps'])
    else:
        raise ValueError("No such dataset...")
    dataset = loader.get_dataset()
    snapshots = [snapshot for snapshot in dataset]
    for graph in snapshots:
        graph = graph.to(args['device'])
    model = My_Model(args, node_features = 2).to(args['device'])

    loss_func = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=0.001)

    acc_log = []
    loss_log = []
    best_acc = 0

    # write graph data to efs for lambda processing
    if args['testbed'] == 'lambda':
        for i, graph in enumerate(snapshots):
            graph_x_path = '/home/ubuntu/mnt/efs/graphs/graph_x_{}.pt'.format(i)
            graph_edge_path = '/home/ubuntu/mnt/efs/graphs/graph_edge_{}.pt'.format(i)
            torch.save(graph.x, graph_x_path, pickle_protocol=2, _use_new_zipfile_serialization=False)
            torch.save(graph.edge_index, graph_edge_path, pickle_protocol=2, _use_new_zipfile_serialization=False)
    
        # warm-up lambda instances
        lambda_client = boto3.client('lambda')
        function_name = 'layer_forward'
        pool_size = 30
        _warmup_lambda(lambda_client, function_name, pool_size)
        print('{} lambda invocations have been warmed up!'.format(pool_size))

    time_cost = []
    for epoch in range(args['epochs']):
        time_start = time.time()
        model.train()
        loss = 0
        samples = [snapshot.train_samples for snapshot in snapshots]
        _, _, outputs = model(snapshots, samples)
        gpu_mem_alloc = torch.cuda.max_memory_allocated() / 1000000 if torch.cuda.is_available() else 0

        for t, snapshot in enumerate(snapshots):
            y = outputs[t]
            label = snapshot.train_labels
            # print('emb size: {}, target_id size: {}'.format(y.size(), label.size()))
            error = loss_func(y.squeeze(dim=-1), label)
            loss += error
        loss = loss/len(snapshots)
        loss_log.append(loss.item())
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        with torch.no_grad():
            model.eval()
            ACC = 0
            samples = [snapshot.test_samples for snapshot in snapshots]
            _, _, outputs = model(snapshots, samples)
            for t, snapshot in enumerate(snapshots):
                y = outputs[t]
                label = snapshot.test_labels.cpu().numpy()
                prob_f1 = []
                prob_auc = []   
                prob_f1.extend(np.argmax(y.detach().cpu().numpy(), axis = 1))
                ACC += sum(prob_f1 == label)/len(label)
            acc = ACC/len(snapshots)
            acc_log.append(acc)
            if best_acc <= acc:
                best_acc = acc
        print('epoch: {} loss: {:.4f} acc: {:.4f} GPU memory {:.3f}'.format(epoch, loss.item(), acc, gpu_mem_alloc))

        time_end = time.time()
        time_cost.append(time_end - time_start)
    print('best accuracy: {:.3f} | total time: {:.3f} | average epoch time: {:.3f}'.format(best_acc, np.sum(time_cost[3:]), np.mean(time_cost[3:])))
    logger.info('device: {} | {} | T: {} | accuracy: {:.3f} | total time: {:.3f} | average epoch time: {:.3f}'.format(args['device'], args['dataset'], args['timesteps'], 
                                                                            best_acc,  np.sum(time_cost[3:]), np.mean(time_cost[3:])))

def run_experiment_stale_aggregation_comm(args):
    # print hyper-parameters
    print(args)
    # print('dataset: {} timesteps: {}'.format(args['dataset'], args['timesteps']))
    # print('Node lists: ', [snapshot.raw_graph.number_of_nodes() for snapshot in snapshots])

    args['rank'] = 0
    args['device'] = torch.device("cuda")
    # args['device'] = 'cpu'
    if args['dataset'] == 'Epinion':
        loader = EpinionDatasetLoader(timesteps = args['timesteps'])
    elif args['dataset'] == 'Amazon':
        loader = AmazonDatasetLoader(timesteps = args['timesteps'])
    elif args['dataset'] == 'Movie':
        loader = MovieDatasetLoader(timesteps = args['timesteps'])
    elif args['dataset'] == 'Stack':
        loader = StackDatasetLoader(timesteps = args['timesteps'])
    else:
        raise ValueError("No such dataset...")
    dataset = loader.get_dataset()
    snapshots = [snapshot for snapshot in dataset]
    for graph in snapshots:
        graph = graph.to(args['device'])
    model = My_Model(args, node_features = 2).to(args['device'])

    _, partition, adjs = _get_partitions(snapshots)

    loss_func = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=0.001)

    acc_log = []
    loss_log = []
    best_acc = 0
    # print('features: ', snapshots[0].x[80:90])
    # pbar = tqdm(range(args['epochs']), leave=False)

    for epoch in range(args['epochs']):
        model.train()
        loss = 0
        samples = [snapshot.train_samples for snapshot in snapshots]
        structural_outputs, temporal_outputs, outputs = model(snapshots, samples)

        gpu_mem_alloc = torch.cuda.max_memory_allocated() / 1000000 if torch.cuda.is_available() else 0

        # if epoch in [10-1, 50-1, 100-1, 150-1]:
        #     pre_str_emb.append(structural_outputs[:300, -1, :].detach().cpu().tolist())
        #     pre_tem_emb.append(temporal_outputs[:300, -1, :].detach().cpu().tolist())
        # if epoch in [10, 50, 100, 150]:
        #     cur_str_emb.append(structural_outputs[:300, -1, :].detach().cpu().tolist())
        #     cur_tem_emb.append(temporal_outputs[:300, -1, :].detach().cpu().tolist())

        # communication simulation
        # if epoch == 0:
        #     simulator = Simulator(structural_outputs, temporal_outputs, partition, adjs)
        # else:
        #     original_comm_str, original_comm_tem = simulator.count_comm(structural_outputs, temporal_outputs)
        #     stale_comm_str, stale_comm_tem = simulator.count_comm_stale(structural_outputs, temporal_outputs, epoch)
        #     reduced_str_comm.append((original_comm_str-stale_comm_str)/original_comm_str)
        #     reduced_tem_comm.append((original_comm_tem-stale_comm_tem)/original_comm_tem)

        for time, snapshot in enumerate(snapshots):
            y = outputs[time]
            label = snapshot.train_labels
            # print('emb size: {}, target_id size: {}'.format(y.size(), label.size()))
            error = loss_func(y.squeeze(dim=-1), label)
            loss += error
        loss = loss/len(snapshots)
        loss_log.append(loss.item())
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        # print('epoch: {} loss: {}'.format(epoch, loss.item()))

        with torch.no_grad():
            model.eval()
            ACC = 0
            samples = [snapshot.test_samples for snapshot in snapshots]
            _, _, outputs = model(snapshots, samples)
            for time, snapshot in enumerate(snapshots):
                y = outputs[time]
                label = snapshot.test_labels.cpu().numpy()
                prob_f1 = []
                prob_auc = []   
                prob_f1.extend(np.argmax(y.detach().cpu().numpy(), axis = 1))
                ACC += sum(prob_f1 == label)/len(label)
            acc = ACC/len(snapshots)
            acc_log.append(acc)
            if best_acc <= acc:
                best_acc = acc
        print('epoch: {} loss: {:.4f} acc: {:.4f} GPU memory {:.3f}'.format(epoch, loss.item(), acc, gpu_mem_alloc))
    print('best accuracy: {:.3f}'.format(best_acc))
    
def run_experiment_stale_aggregation(args, logger, thresholds):
    args['rank'] = 0
    args['device'] = torch.device("cuda")
    # args['device'] = 'cpu'
    if args['dataset'] == 'Epinion':
        loader = EpinionDatasetLoader(timesteps = args['timesteps'])
    elif args['dataset'] == 'Amazon':
        loader = AmazonDatasetLoader(timesteps = args['timesteps'])
    elif args['dataset'] == 'Movie':
        loader = MovieDatasetLoader(timesteps = args['timesteps'])
    elif args['dataset'] == 'Stack':
        loader = StackDatasetLoader(timesteps = args['timesteps'])
    else:
        raise ValueError("No such dataset...")
    dataset = loader.get_dataset()
    snapshots = [snapshot for snapshot in dataset]

    for graph in snapshots:
        graph = graph.to(args['device'])
    for threshold in thresholds:
        args['threshold'] = threshold
        if args['threshold'] == 0:
            args['stale'] = False
        else:
            args['stale'] = True
        print(args)

        _, partition, adjs = _get_partitions(snapshots)

        best_acc_list = []
        total_str_reduced_comm = []
        total_tem_reduced_comm = []
        for i in range(5):
            args['threshold'] = threshold
            model = My_Model(args, node_features = 2).to(args['device'])
            loss_func = torch.nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=0.001)
            acc_log = []
            loss_log = []
            best_acc = 0
            reduced_str_comm = []
            reduced_tem_comm = []
            for epoch in range(args['epochs']):
                # if epoch != 0 and epoch%50 == 0:
                #     args['threshold'] += 0.5
                #     print('current threshold ', args['threshold'])
                model.train()
                loss = 0
                samples = [snapshot.train_samples for snapshot in snapshots]
                structural_outputs, temporal_outputs, outputs = model(snapshots, samples, epoch)
                # communication simulation
                if epoch == 0:
                    simulator = Simulator(structural_outputs, temporal_outputs, partition, adjs)
                else:
                    original_comm_str, original_comm_tem = simulator.count_comm(structural_outputs, temporal_outputs)
                    stale_comm_str, stale_comm_tem, reduced_str, reduced_tem = simulator.count_comm_stale(structural_outputs, temporal_outputs, epoch, args['threshold'])
                    print('reduced spatial communication {}  and temporal communication {}'.format(np.mean(reduced_str), np.mean(reduced_tem)))
                    # reduced_str_comm.append((original_comm_str-stale_comm_str)/original_comm_str)
                    # reduced_tem_comm.append((original_comm_tem-stale_comm_tem)/original_comm_tem)

                gpu_mem_alloc = torch.cuda.max_memory_allocated() / 1000000 if torch.cuda.is_available() else 0

                for time, snapshot in enumerate(snapshots):
                    y = outputs[time]
                    label = snapshot.train_labels
                    # print('emb size: {}, target_id size: {}'.format(y.size(), label.size()))
                    error = loss_func(y.squeeze(dim=-1), label)
                    loss += error
                loss = loss/len(snapshots)
                loss_log.append(loss.item())
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                # print('epoch: {} loss: {}'.format(epoch, loss.item()))

                with torch.no_grad():
                    model.eval()
                    ACC = 0
                    samples = [snapshot.test_samples for snapshot in snapshots]
                    _, _, outputs = model(snapshots, samples)
                    for time, snapshot in enumerate(snapshots):
                        y = outputs[time]
                        label = snapshot.test_labels.cpu().numpy()
                        prob_f1 = []
                        prob_auc = []   
                        prob_f1.extend(np.argmax(y.detach().cpu().numpy(), axis = 1))
                        ACC += sum(prob_f1 == label)/len(label)
                    acc = ACC/len(snapshots)
                    acc_log.append(acc)
                    if best_acc <= acc:
                        best_acc = acc
                print('epoch: {} loss: {:.4f} acc: {:.4f} GPU memory {:.3f}'.format(epoch, loss.item(), acc, gpu_mem_alloc))
            total_str_reduced_comm.append(np.mean(reduced_str))
            total_tem_reduced_comm.append(np.mean(reduced_tem))
            best_acc_list.append(best_acc)
        print('best accuracy: {:.3f} error: {:.3f} str comm: {:.3f} tem comm: {:.3f}'.format(np.mean(best_acc_list[3:]), np.std(best_acc_list[3:]), np.mean(total_str_reduced_comm), np.mean(total_tem_reduced_comm)))
        logger.info('{} | T: {} | threshold: {} | accuracy: {:.3f} | error: {:.3f} | str reduced: {:.3f} | tem reduced: {:.3f}'.format(args['dataset'], args['timesteps'], args['threshold'], np.mean(best_acc_list[3:]), np.std(best_acc_list[3:]), np.mean(total_str_reduced_comm), np.mean(total_tem_reduced_comm)))

def run_experiment_fusion(args):
    print(args)

    args['rank'] = 0
    args['device'] = torch.device("cuda")
    args['stale'] = False

    # args['device'] = 'cpu'
    if args['dataset'] == 'Epinion':
        loader = EpinionDatasetLoader(timesteps = args['timesteps'])
    elif args['dataset'] == 'Amazon':
        loader = AmazonDatasetLoader(timesteps = args['timesteps'])
    elif args['dataset'] == 'Movie':
        loader = MovieDatasetLoader(timesteps = args['timesteps'])
    elif args['dataset'] == 'Stack':
        loader = StackDatasetLoader(timesteps = args['timesteps'])
    else:
        raise ValueError("No such dataset...")
    dataset = loader.get_dataset()
    snapshots = [snapshot for snapshot in dataset]

    for graph in snapshots:
        graph = graph.to(args['device'])

    model = My_Model(args, node_features = 2).to(args['device'])
    loss_func = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=0.001)
    acc_log = []
    loss_log = []
    best_acc = 0

    import wandb
    run = wandb.init(project="DGNN_GPU_utilization", entity="fahao", name=args['dataset'], reinit=True)
    for epoch in range(args['epochs']):
        model.train()
        loss = 0
        samples = [snapshot.train_samples for snapshot in snapshots]
        _, _, outputs = model(snapshots, samples, epoch)
        gpu_mem_alloc = torch.cuda.max_memory_allocated() / 1000000 if torch.cuda.is_available() else 0
        for time, snapshot in enumerate(snapshots):
            y = outputs[time]
            label = snapshot.train_labels
            # print('emb size: {}, target_id size: {}'.format(y.size(), label.size()))
            error = loss_func(y.squeeze(dim=-1), label)
            loss += error
        loss = loss/len(snapshots)
        loss_log.append(loss.item())
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        # print('epoch: {} loss: {}'.format(epoch, loss.item()))

        with torch.no_grad():
            model.eval()
            ACC = 0
            samples = [snapshot.test_samples for snapshot in snapshots]
            _, _, outputs = model(snapshots, samples)
            for time, snapshot in enumerate(snapshots):
                y = outputs[time]
                label = snapshot.test_labels.cpu().numpy()
                prob_f1 = []
                prob_auc = []   
                prob_f1.extend(np.argmax(y.detach().cpu().numpy(), axis = 1))
                ACC += sum(prob_f1 == label)/len(label)
            acc = ACC/len(snapshots)
            acc_log.append(acc)
            if best_acc <= acc:
                best_acc = acc
        print('epoch: {} loss: {:.4f} acc: {:.4f} GPU memory {:.3f}'.format(epoch, loss.item(), acc, gpu_mem_alloc))
    print('best accuracy: {:.3f} '.format(best_acc))
    run.finish()

if __name__ == '__main__':
    args = _get_args()
    folder_in = os.path.exists('../log/')
    if not folder_in:
        os.makedirs('../log/')

    if args['experiments'] == 'stale_acc':
        experiment_thresholds = [0, 0.1, 0.3, 0.5, 0.9]
        experiments_datasets = ['Amazon', 'Epinion', 'Movie', 'Stack']
        for dataset in experiments_datasets:
            args['dataset'] = dataset
            # log config
            current_path = os.path.abspath(os.path.join(os.getcwd(), ".."))
            log_file = current_path + '/log/example_experiment_stale_aggregation.log'
            logging.basicConfig(filename=log_file,
                                filemode='a',
                                format='%(message)s',
                                level=logging.INFO)
            # logging.basicConfig(level=logging.INFO, format='%(message)s')
            logger = logging.getLogger('example')
            run_experiment_stale_aggregation(args, logger, experiment_thresholds)
    
    elif args['experiments'] == 'stale_comm':
        # run_example(args)
        reduced_communication_str = []
        reduced_communication_tem = []
        experiments_datasets = ['Amazon', 'Epinion', 'Movie', 'Stack']
        for dataset in experiments_datasets:
            args['dataset'] = dataset
            str_reduce, tem_reduce = run_experiment_stale_aggregation_comm(args)
            reduced_communication_str.append(str_reduce)
            reduced_communication_tem.append(tem_reduce)
        print('average reduced spatial communication: {} | reduced temporal communication: {}'.format(reduced_communication_str, reduced_communication_tem))

    elif args['experiments'] == 'fusion_gpu':
        experiments_datasets = ['Amazon', 'Epinion', 'Movie', 'Stack']
        for dataset in experiments_datasets:
            args['dataset'] = dataset
            run_experiment_fusion(args)
    
    elif args['experiments'] == 'default':
        current_path = os.path.abspath(os.path.join(os.getcwd(), ".."))
        log_file = current_path + '/log/example.log'
        logging.basicConfig(filename=log_file,
                                filemode='a',
                                format='%(message)s',
                                level=logging.INFO)
        # logging.basicConfig(level=logging.INFO, format='%(message)s')
        logger = logging.getLogger('example')
        logger.info('----------------------------------------------------------')
        experiments_datasets = ['Amazon']
        for dataset in experiments_datasets:
            args['dataset'] = dataset
            run_example(args, logger)

    else:
        raise Exception('There is no such an experiment type!')




    




