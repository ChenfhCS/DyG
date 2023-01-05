import torch
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "1"
device = torch.device('cuda')

def count_comm_vollum():
    return 0

def _tensor_distance(tensor_A, tensor_B):
    sub_c = torch.sub(tensor_A, tensor_B)
    sq_sub_c = sub_c**2
    sum_sq_sub_c = torch.sum(sq_sub_c, dim=2)
    distance = torch.sqrt(sum_sq_sub_c)
    return distance

class Simulator():
    def __init__(self, origin_emb_str, origin_emb_tem, partition, adjs) -> None:
        super(Simulator, self).__init__()
        self.emb_str_checkpoint = origin_emb_str
        self.emb_tem_checkpoint = origin_emb_tem
        self.partition = partition # [m][t][n]->list of tensor
        self.adjs = adjs # for str communication counting

        self.max_num_v = self.partition[0][-1].size(0)
        self.num_partitions = len(self.partition)
        self.num_snapshots = len(self.partition[0])

        self.N = origin_emb_str.size(0)
        self.T = origin_emb_str.size(1)
        self.F = origin_emb_str.size(2)

        self.comm_str_index = torch.ones(self.max_num_v, self.num_snapshots, dtype=torch.bool)
        self.comm_tem_index = torch.ones(self.max_num_v, self.num_snapshots, dtype=torch.bool)
        # self._get_comm_v_index()

    def count_comm_incremental(self, emb_str, emb_tem):
        str_need_comm = torch.zeros(self.max_num_v, self.num_snapshots, emb_str.size(2), dtype=torch.bool)
        tem_need_comm = torch.zeros(self.max_num_v, self.num_snapshots, emb_tem.size(2), dtype=torch.bool)

        str_count = 0
        tem_count = 0

        # compare_str = emb_str - self.emb_str_checkpoint
        # compare_tem = emb_tem - self.emb_tem_checkpoint
        compare_str = torch.sub(emb_str, self.emb_str_checkpoint)
        compare_tem = torch.sub(emb_tem, self.emb_tem_checkpoint)

        # print('emb: {} checkpoint: {} compare: {}'.format(emb_str[:, 0, :], self.emb_str_checkpoint[:, 0, :], compare_str[:, 0, :]))
        # print(compare_tem)
        for time in range(self.num_snapshots):
            str_send = torch.nonzero(self.comm_str_index[:, time] == True, as_tuple=False).view(-1)
            tem_send = torch.nonzero(self.comm_tem_index[:, time] == True, as_tuple=False).view(-1)
            str_need_comm[str_send, time, :] = torch.ones(str_send.size(0), emb_str.size(2), dtype=torch.bool)
            tem_need_comm[tem_send, time, :] = torch.ones(tem_send.size(0), emb_str.size(2), dtype=torch.bool)

        unchange_str_count = 0
        unchange_tem_count = 0
        for time in range(self.num_snapshots):
            for feat in range(emb_str.size(2)):
                # print(compare_str[time][:, feat].size())
                # all_v = compare_str[self.comm_str_index[:, time], time, feat] # no change and need to send
                str_idx = torch.nonzero(compare_str[:, time, feat] == 0, as_tuple=False).view(-1)
                unchange_str_count += str_idx.size(0)
                str_need_comm[str_idx, time, feat] = torch.zeros(str_idx.size(0), dtype=torch.bool)
                # print('unchanged valume: ', compare_str[str_idx, time, feat].size())

            # print('unchanged node: {} emb: {} checked emb: {}'.format(str_idx, emb_str[str_idx, time, :], self.emb_str_checkpoint[str_idx, time, :]))

            for feat in range(emb_tem.size(2)):
                # all_v = compare_tem[self.comm_tem_index[:, time], time, feat]
                tem_idx = torch.nonzero(compare_tem[:, time, feat] == 0, as_tuple=False).view(-1)
                unchange_tem_count += tem_idx.size(0)
                tem_need_comm[tem_idx, time, feat] = torch.zeros(tem_idx.size(0), dtype=torch.bool)

        for time in range(self.num_snapshots):
            for feat in range(emb_str.size(2)):
                change_emb = torch.nonzero(str_need_comm[:, time, feat] == True, as_tuple=False).view(-1)
                str_count += change_emb.size(0)
                self.emb_str_checkpoint[change_emb, time, feat] = emb_str[change_emb, time, feat]
            for feat in range(emb_tem.size(2)):
                change_emb = torch.nonzero(tem_need_comm[:, time, feat] == True, as_tuple=False).view(-1)
                tem_count += change_emb.size(0)
                self.emb_tem_checkpoint[change_emb, time, feat] = emb_tem[change_emb, time, feat]

        return str_count, tem_count
    
    def count_comm(self, emb_str, emb_tem):
        """
        para:
        emb_str: a list of [NxF] tensor
        emb_tem: a [NxTxF] tensor
        """
        str_count = 0
        tem_count = 0
        for time in range(self.num_snapshots):
            count = torch.nonzero(self.comm_str_index[:, time] == True, as_tuple=False).view(-1)
            str_count += count.size(0)*emb_str.size(2)
            # print(type(self.comm_tem_index), type(emb_tem))
            count = torch.nonzero(self.comm_tem_index[:, time] == True, as_tuple=False).view(-1)
            tem_count += count.size(0)*emb_tem.size(2)
        return str_count, tem_count

    def count_comm_stale(self, emb_str, emb_tem, epoch, theta):

        '''
        paras:
            emb_str: vertices spatial embeddings in current epoch
            emb_tem: vertices temporal embeddings in current epoch
        '''
        str_need_not_comm = torch.zeros(self.N, self.T, dtype=torch.bool) # [N,T]
        tem_need_not_comm = torch.zeros(self.N, self.T, dtype=torch.bool)
        for time in range(self.T):
            str_not_send = torch.nonzero(self.comm_str_index[:, time] == False, as_tuple=False).view(-1)
            tem_not_send = torch.nonzero(self.comm_tem_index[:, time] == False, as_tuple=False).view(-1)
            str_need_not_comm[str_not_send, time] = torch.ones(str_not_send.size(0), dtype=torch.bool)
            tem_need_not_comm[tem_not_send, time] = torch.ones(tem_not_send.size(0), dtype=torch.bool)

        # step 1: calculate the L2 distance between current embedding and original embeddings (the latest sent)
        # Each embedding has the dimension of F
        comm_str_vollume = 0
        comm_tem_vollume = 0
        whether_send_str = torch.zeros(self.N, self.T, dtype=torch.bool)
        whether_send_tem = torch.zeros(self.N, self.T, dtype=torch.bool)
        distance_str = _tensor_distance(self.emb_str_checkpoint, emb_str)
        distance_tem = _tensor_distance(self.emb_tem_checkpoint, emb_tem)

        reduced_str = []
        reduced_tem = []
        for time in range(self.T):
            str_not_send_num = torch.nonzero(str_need_not_comm[:, time] == True, as_tuple=False).view(-1)
            tem_not_send_num = torch.nonzero(tem_need_not_comm[:, time] == True, as_tuple=False).view(-1)
            str_not_send_mask = str_need_not_comm[:, time].to(device)
            tem_not_send_mask = tem_need_not_comm[:, time].to(device)
            str_send_mask = ~str_need_not_comm[:, time].to(device)
            tem_send_mask = ~tem_need_not_comm[:, time].to(device)

            # print('original spatial distances ', distance_str[:, time].tolist())
            # print('original temporal distances ', distance_tem[:, time].tolist())
            # process the spatial and temporal embeddings
            str_values, str_indices = distance_str[:, time].topk(2, largest=True, sorted=True)
            tem_values, tem_indices = distance_tem[:, time].topk(2, largest=True, sorted=True)
            distance_str[str_indices, time] = torch.zeros(str_indices.size(0), dtype=torch.float32).to(device)
            distance_tem[tem_indices, time] = torch.zeros(tem_indices.size(0), dtype=torch.float32).to(device)

            # # calculate the average distance, method 1: average distance
            # distance_str[str_not_send_mask, time] = torch.zeros(str_not_send_num.size(0), dtype=torch.float32).to(device)
            # distance_tem[tem_not_send_mask, time] = torch.zeros(tem_not_send_num.size(0), dtype=torch.float32).to(device)
            # str_avg_distance = torch.mean(distance_str[str_send_mask, time])
            # tem_avg_distance = torch.mean(distance_tem[tem_send_mask, time])
            # str_threshold = str_avg_distance*theta
            # tem_threshold = tem_avg_distance*theta
            # greater_distance_str = torch.nonzero(distance_str[str_send_mask, time] >= str_threshold, as_tuple=False).view(-1)
            # greater_distance_tem = torch.nonzero(distance_tem[tem_send_mask, time] >= tem_threshold, as_tuple=False).view(-1)
            # smaller_distance_str = torch.nonzero(distance_str[str_send_mask, time] < str_threshold, as_tuple=False).view(-1)
            # smaller_distance_tem = torch.nonzero(distance_tem[tem_send_mask, time] < tem_threshold, as_tuple=False).view(-1)

            # method 2: normalized distance
            str_max_distance = torch.max(distance_str[:, time])
            tem_max_distance = torch.max(distance_tem[:, time])
            str_normalized_distance = distance_str[:, time]/str_max_distance
            tem_normalized_distance = distance_str[:, time]/tem_max_distance
            # str_normalized_distance[str_not_send_mask] = torch.zeros(str_not_send_num.size(0), dtype=torch.float32).to(device)
            # tem_normalized_distance[tem_not_send_mask] = torch.zeros(tem_not_send_num.size(0), dtype=torch.float32).to(device)
            greater_distance_str = torch.nonzero(str_normalized_distance[:] >= theta/100, as_tuple=False).view(-1)
            smaller_distance_str = torch.nonzero(str_normalized_distance[:] < theta/100, as_tuple=False).view(-1)
            greater_distance_tem = torch.nonzero(tem_normalized_distance[:] >= theta/100, as_tuple=False).view(-1)
            smaller_distance_tem = torch.nonzero(tem_normalized_distance[:] < theta/100, as_tuple=False).view(-1)
    
            print('need to send str embeddings {}, reduced communication {}'.format(str_normalized_distance.size(0), smaller_distance_tem.size(0)))
            reduced_str.append(smaller_distance_str.size(0)/str_send_mask.size(0))
            reduced_tem.append(smaller_distance_tem.size(0)/tem_send_mask.size(0))
            whether_send_str[greater_distance_str, time] = torch.ones(greater_distance_str.size(0), dtype=torch.bool)
            whether_send_tem[greater_distance_tem, time] = torch.ones(greater_distance_tem.size(0), dtype=torch.bool)
            comm_str_vollume += greater_distance_str.size(0)
            comm_tem_vollume += greater_distance_tem.size(0)

        
        # update latest sent embeddings
        for time in range(self.T):
            self.emb_str_checkpoint[whether_send_str[:, time], time] = emb_str[whether_send_str[:, time], time]
            self.emb_tem_checkpoint[whether_send_tem[:, time], time] = emb_tem[whether_send_tem[:, time], time]
        
        return comm_str_vollume*self.F, comm_tem_vollume*self.F, reduced_str, reduced_tem

    # def count_comm_volume(self, emb_str, emb_tem):
    #     comm_volume_str = 0
    #     comm_volume_tem = 0

    def _get_comm_v_index(self):
        self.comm_str_index = torch.zeros(self.max_num_v, self.num_snapshots, dtype=torch.bool)
        self.comm_tem_index = torch.zeros(self.max_num_v, self.num_snapshots, dtype=torch.bool)

        # get str comm index
        for device_id in range(self.num_partitions):
            for time in range(self.num_snapshots):
                adj = self.adjs[time].clone()
                local_node_mask = self.partition[device_id][time]
                remote_node_mask = ~self.partition[device_id][time]
                edge_source = adj._indices()[0]
                edge_target = adj._indices()[1]

                # send
                edge_source_remote_mask = remote_node_mask[edge_source] # check each source node whether it belongs to other devices
                need_send_nodes = edge_target[edge_source_remote_mask] # get the target nodes with the source nodes belong to other devices
                send_node_local = local_node_mask[need_send_nodes] # check whether the send nodes in local?
                send = torch.nonzero(send_node_local == True, as_tuple=False).squeeze().view(-1) # only the send nodes are in local
                # print('send: ', send)
                self.comm_str_index[send, time] = torch.ones(send.size(0), dtype=torch.bool)

        # get tem comm index
        Req = [[torch.zeros(self.max_num_v, dtype=torch.bool) for time in range(self.num_snapshots)] for m in range(self.num_partitions)]
        for m in range(self.num_partitions):
            # compute the required node list
            for time in range(self.num_snapshots):
                where_need_comp = torch.nonzero(self.partition[m][time] == True, as_tuple=False).view(-1)
                if where_need_comp!= torch.Size([]):
                    for k in range(self.num_snapshots)[0:time-1]:
                        idx = torch.tensor([i for i in range(Req[m][k].size(0))])
                        need_nodes_mask = self.partition[m][time][idx]
                        where_need = torch.nonzero(need_nodes_mask == True, as_tuple=False).view(-1)
                        # print(where_need)
                        if (where_need.size(0) > 0):
                            Req[m][k][where_need] = torch.ones(where_need.size(0), dtype=torch.bool)
            # remove already owned nodes
            for time in range(self.num_snapshots):
                where_have_nodes = torch.nonzero(self.partition[m][time] == True, as_tuple=False).view(-1)
                # print(where_have_nodes)
                if where_have_nodes!= torch.Size([]):
                    Req[m][time][where_have_nodes] = torch.zeros(where_have_nodes.size(0), dtype=torch.bool)
        # Compute the number of nodes need to be sent
        for m in range(self.num_partitions):
            for time in range(self.num_snapshots):
                others_need = torch.zeros(self.max_num_v, dtype=torch.bool)
                for k in range(self.num_partitions):
                    where_other_need = torch.nonzero(Req[k][time] == True, as_tuple=False).view(-1)
                    others_need[where_other_need] = torch.ones(where_other_need.size(0), dtype=torch.bool)
                where_have = torch.nonzero(self.partition[m][time] == True, as_tuple=False).view(-1)
                send_mask = others_need[where_have]
                send = torch.nonzero(send_mask == True, as_tuple=False).view(-1)
                self.comm_tem_index[send, time] = torch.ones(send.size(0), dtype=torch.bool)
        
        str_count = 0
        tem_count = 0
        for time in range(self.num_snapshots):
            str_count += torch.nonzero(self.comm_str_index[:, time] == True, as_tuple=False).view(-1).size(0)
            tem_count += torch.nonzero(self.comm_tem_index[:, time] == True, as_tuple=False).view(-1).size(0)
        print('need to send nodes {} {}'.format(str_count, tem_count))


if __name__ == '__main__':
    # test tensor distance calculation
    tensor_A = torch.ones(2, 2, 2, dtype=torch.float32)
    tensor_B = torch.zeros(2, 2, 2, dtype=torch.float32)
    distance = _tensor_distance(tensor_A, tensor_B)
    print(distance)
