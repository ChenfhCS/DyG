import torch

def count_comm_vollum():
    return 0

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

        self._get_comm_v_index()

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



