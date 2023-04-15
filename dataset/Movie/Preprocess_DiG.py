import os
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import numpy as np

from collections import defaultdict
from datetime import datetime, timedelta
from tqdm import tqdm


# Set up folder paths and check if folder exists
current_path = os.getcwd()
folder_in = os.path.exists(current_path + '/data/snapshots/')
if not folder_in:
    os.makedirs(current_path + '/data/snapshots/')

# Set up file paths and load data from file
file_path = current_path + '/rec-movielens-ratings.edges'
save_graph_path = current_path + f'/data/snapshots/'

def _load_data():
    # Initialize variables
    links = []
    ts = []
    ctr = 0
    node_cnt = 0
    node_idx = {}

    with open(file_path) as f:
        lines = f.read().splitlines()
        pbar = tqdm(lines, desc='Loading', leave=False)
        for l in pbar:
            # Ignore comments in the input file
            if l[0] == '%':
                continue

            # Parse edge information from file
            x, y, e, t = map(float, l.split(','))
            timestamp = datetime.fromtimestamp(t)
            ts.append(timestamp)

            ctr += 1

            # Create node mappings
            if x not in node_idx:
                node_idx[x] = node_cnt
                node_cnt += 1

            if y not in node_idx:
                node_idx[y] = node_cnt
                node_cnt += 1

            # Append edge information to links list
            links.append((node_idx[x],node_idx[y], timestamp))

    print ("Min ts", min(ts), "max ts", max(ts))
    print ("Total time span: {} days".format((max(ts) - min(ts)).days))
    links.sort(key =lambda x: x[2])
    return ts, links

# Define a function to remap node ids
def _remap(snapshot):
    node_mapping = {n: i for i, n in enumerate(sorted(snapshot.nodes()))}
    snapshot_remap = nx.relabel_nodes(snapshot, node_mapping, copy=True)
    return snapshot_remap

# Define a function to save graph to file
def _save_graph(snapshot, snapshot_id):
    path = save_graph_path + f'snapshot_{snapshot_id}.gpickle'
    snspshot_remap = _remap(snapshot)
    nx.write_gpickle(snspshot_remap, path)

# Generate snapshots according to links
def _generate_snapshots(slice_days, ts, links):
    START_DATE = min(ts) + timedelta(0)
    # END_DATE =  max(ts) - timedelta(700)
    # END_DATE = min(ts) + timedelta(1500)
    END_DATE = max(ts)

    # END_DATE = timedelta(100)
    slices_links = defaultdict(lambda : nx.DiGraph())
    slices_features = defaultdict(lambda : {})

    slice_id = -1
    snapshot_id = 0
    current_total_nodes = 0
    pbar = tqdm(links, leave=False)
    for (a, b, time) in pbar:
        prev_slice_id = slice_id

        datetime_object = time
        if datetime_object < START_DATE:
            continue
        if datetime_object > END_DATE:
            break
            days_diff = (END_DATE - START_DATE).days
        else:
            days_diff = (datetime_object - START_DATE).days


        slice_id = days_diff // slice_days

        if slice_id == 1+prev_slice_id and slice_id > 0:
            snapshot_id += 1
            slices_links[snapshot_id] = nx.DiGraph()
            # slices_links[snapshot_id].add_nodes_from(slices_links[snapshot_id-1].nodes(data=True))
            pbar.set_description("Creating snapshot {}: |V|={}, |E|={}".format(slice_id, 
                                                            len(slices_links[snapshot_id-1].nodes()), 
                                                            len(slices_links[snapshot_id-1].edges())))

            _save_graph(slices_links[snapshot_id-1], snapshot_id-1)
            slices_links[snapshot_id-1] = nx.DiGraph()

        if slice_id == 1+prev_slice_id and slice_id ==0:
            slices_links[snapshot_id] = nx.DiGraph()

        if a not in slices_links[snapshot_id]:
            slices_links[snapshot_id].add_node(a)
        if b not in slices_links[snapshot_id]:
            slices_links[snapshot_id].add_node(b)
        slices_links[snapshot_id].add_edge(a,b, date=datetime_object)

    if snapshot_id > 0:
        _save_graph(slices_links[snapshot_id], snapshot_id)

    # clear last snapshot to free memory
    slices_links[snapshot_id] = nx.DiGraph()

    return snapshot_id+1

def _graph_profile(N_snapshots):
    nodes = []
    edges = []
    snapshots = []
    neighbors = []
    pbar = tqdm(range(N_snapshots), desc='Profiling', leave=False)
    for i in pbar:
        graph_path = save_graph_path + f'snapshot_{i}.gpickle'
        snapshot = nx.read_gpickle(graph_path)
        snapshots.append(snapshot)
        for node in snapshot.nodes():
            neighbors.append(len(list(snapshot.neighbors(node))))
        if i > 0:
            snapshots[i].add_nodes_from(snapshots[i-1].nodes(data=True))
            # nodes.append(snapshots[i].number_of_nodes() - snapshots[i - 1].number_of_nodes())
            nodes.append(snapshots[i].number_of_nodes())
            edges.append(snapshots[i].number_of_edges())
            snapshots[i-1] = nx.DiGraph()
        else:
            nodes.append(snapshots[i].number_of_nodes())
            edges.append(snapshots[i].number_of_edges())

    # cdf of structural neighbors
    fig = plt.figure(figsize=(6, 4))
    sorted_neighbors = np.sort(neighbors)[:-10]
    condition_1 = np.array(sorted_neighbors) <= 100
    # 使用where()函数获取符合条件的索引，然后使用这些索引对原始列表进行筛选
    filtered_neighbors = np.array(sorted_neighbors)[condition_1].tolist()
    condition_2 = np.array(filtered_neighbors) > 0
    neighbors = np.array(filtered_neighbors)[condition_2].tolist()
    yvals = np.arange(len(neighbors))/float(len(neighbors))
    plt.plot(neighbors, yvals)
    plt.title('CDF of Neighbor Counts')
    plt.xlabel('Number of Structural Neighbors')
    plt.ylabel('CDF')
    plt.savefig(current_path + '/neighbor_cdf.png', dpi=900,bbox_inches='tight')


    # cdf of time lengths
    fig = plt.figure(figsize=(6, 4))
    Y = np.array(nodes, dtype=float)
    Y = Y / np.sum(Y)
    Y = Y[::-1]
    X = [i+1 for i in range(N_snapshots)]
    CDF = np.cumsum(Y)
    plt.plot(X, CDF)
    plt.title('CDF')
    plt.xlabel('Time Length')
    plt.ylabel('CDF')
    plt.savefig(current_path + '/time_cdf.png', dpi=900,bbox_inches='tight')

    # # cdf of nodes
    # sorted_nodes = np.sort(nodes)
    # yvals = np.arange(len(sorted_nodes))/float(len(sorted_nodes))
    # fig = plt.figure(figsize=(6, 4))
    # plt.plot(sorted_nodes, yvals)
    # plt.title('CDF')
    # plt.xlabel('Number of Nodes')
    # plt.ylabel('CDF')
    # plt.savefig(current_path + '/node_cdf.png', dpi=900,bbox_inches='tight')

    # cdf of edges
    sorted_edges = np.sort(edges)
    fig = plt.figure(figsize=(6, 4))
    yvals = np.arange(len(sorted_edges))/float(len(sorted_edges))
    plt.plot(sorted_edges, yvals)
    plt.title('CDF')
    plt.xlabel('Number of Edges')
    plt.ylabel('CDF')
    plt.savefig(current_path + '/edge_cdf.png', dpi=900,bbox_inches='tight')

if __name__ == '__main__':
    ts, links = _load_data()
    N_snapshots = _generate_snapshots(slice_days=30, ts = ts, links = links)
    # _graph_profile(274)



# # import dill
# from collections import defaultdict
# from datetime import datetime, timedelta
# import os

# current_path = os.getcwd()
# folder_in = os.path.exists(current_path + '/data/')
# if not folder_in:
#     os.makedirs(current_path + '/data/')

# links = []
# ts = []
# ctr = 0
# node_cnt = 0
# node_idx = {}
# idx_node = []

# file_path = current_path + '/rec-movielens-ratings.edges'
# save_graph_path = current_path + '/data/graphs.npz'
# save_features_path = current_path + '/data/features.npz'

# with open(file_path) as f:
#     lines = f.read().splitlines()
#     for l in lines:
#         if l[0] == '%':
#             continue

#         x, y, e, t = map(float, l.split(','))
#         # print (x,y,e,t)
#         timestamp = datetime.fromtimestamp(t)
#         ts.append(timestamp)

#         ctr += 1
#         if ctr % 100000 == 0:
#             print (ctr)

#         if x not in node_idx:
#             node_idx[x] = node_cnt
#             node_cnt += 1

#         if y not in node_idx:
#             node_idx[y] = node_cnt
#             node_cnt += 1

#         links.append((node_idx[x],node_idx[y], timestamp))

# print ("Min ts", min(ts), "max ts", max(ts))
# print ("Total time span: {} days".format((max(ts) - min(ts)).days))
# links.sort(key =lambda x: x[2])
# print ("# temporal links", len(links))

# # import networkx as nx
# # agg_G = nx.Graph()
# # for a,b,t in links:
# #     agg_G.add_edge(a,b)

# # print ("Agg graph", len(agg_G.nodes()), len(agg_G.edges()))

# import networkx as nx
# import numpy as np
# from datetime import datetime, timedelta
# '''
# collect data from 'START_DATE' and ends to 'END_DATE'.
# generate a graph per 'SLICE_DAYS'.
# '''
# # slice defaule = 30
# SLICE_DAYS = 30
# START_DATE = min(ts) + timedelta(100)
# END_DATE = min(ts) + timedelta(2000)

# print ("Start date", START_DATE)
# print ("End date", END_DATE)

# slices_links = defaultdict(lambda : nx.DiGraph())
# slices_features = defaultdict(lambda : {})

# slice_id = -1
# snapshot_id = 0
# # Split the set of links in order by slices to create the graphs.
# for (a, b, time) in links:
#     prev_slice_id = slice_id

#     datetime_object = time
#     if datetime_object < START_DATE:
#         continue
#     if datetime_object > END_DATE:
#         break
#         days_diff = (END_DATE - START_DATE).days
#     else:
#         days_diff = (datetime_object - START_DATE).days


#     slice_id = days_diff // SLICE_DAYS

#     if slice_id == 1+prev_slice_id and slice_id > 0:
#         snapshot_id += 1
#         slices_links[snapshot_id] = nx.DiGraph()
#         slices_links[snapshot_id].add_nodes_from(slices_links[snapshot_id-1].nodes(data=True))
#         # assert (len(slices_links[snapshot_id].edges()) ==0)
#         #assert len(slices_links[slice_id].nodes()) >0

#     if slice_id == 1+prev_slice_id and slice_id ==0:
#         slices_links[snapshot_id] = nx.DiGraph()

#     # if days_diff % SLICE_DAYS == 7 or days_diff % SLICE_DAYS == 6 or days_diff % SLICE_DAYS == 5:
#     #     if a not in slices_links[slice_id]:
#     #         slices_links[slice_id].add_node(a)
#     #     if b not in slices_links[slice_id]:
#     #         slices_links[slice_id].add_node(b)
#     #     slices_links[slice_id].add_edge(a,b, date=datetime_object)

#     if a not in slices_links[snapshot_id]:
#         slices_links[snapshot_id].add_node(a)
#     if b not in slices_links[snapshot_id]:
#         slices_links[snapshot_id].add_node(b)
#     slices_links[snapshot_id].add_edge(a,b, date=datetime_object)

# for slice_id in slices_links:
#     print ("# nodes in slice", slice_id, len(slices_links[slice_id].nodes()))
#     print ("# edges in slice", slice_id, len(slices_links[slice_id].edges()))

#     # temp = np.identity(len(slices_links[max(slices_links.keys())].nodes()))
#     # print ("Shape of temp matrix", temp.shape)
#     slices_features[slice_id] = {}
#     # for idx, node in enumerate(slices_links[slice_id].nodes()):
#     #     slices_features[slice_id][node] = temp[idx]


# # TODO : remap and output.
# from scipy.sparse import csr_matrix

# def remap(slices_graphs, slices_features):
#     snapshots = []
#     slices_features_remap = []
#     for slices_id in slices_graphs:
#         slices_graph = slices_graphs[slices_id]
#         node_mapping = {n: i for i, n in enumerate(slices_graph.nodes())}
#         slices_graph_remap = nx.relabel_nodes(slices_graph, node_mapping, copy=True)
#         snapshots.append(slices_graph_remap)
#     return snapshots, slices_features_remap

#     all_nodes = []
#     for slice_id in slices_graph:
#         # assert len(slices_graph[slice_id].nodes()) == len(slices_features[slice_id])
#         all_nodes.extend(slices_graph[slice_id].nodes())
#     all_nodes = list(set(all_nodes))
#     print ("Total # nodes", len(all_nodes), "max idx", max(all_nodes))
#     ctr = 0
#     node_idx = {}
#     idx_node = []
#     for slice_id in slices_graph:
#         for node in slices_graph[slice_id].nodes():
#             if node not in node_idx:
#                 node_idx[node] = ctr
#                 idx_node.append(node)
#                 ctr += 1
#     print('Get all nodes list complete!')

#     # generate snapshots list
#     slices_graph_remap = []
#     slices_features_remap = []
#     for (id, slice_id) in enumerate(slices_graph):
#         G = nx.DiGraph()
#         if id > 0:
#             # print('G nodes:', slices_graph_remap[id - 1].nodes())
#             # print('slice nodes:', slices_graph[id - 1].nodes())
#             # print(len(slices_graph_remap[id - 1].nodes()))
#             for x in slices_graph_remap[-1].nodes():
#                 G.add_node(x)
#         for x in slices_graph[slice_id].nodes():
#             if node_idx[x] not in G.nodes():
#                 G.add_node(node_idx[x])
#         for x in slices_graph[slice_id].edges(data=True):
#             G.add_edge(node_idx[x[0]], node_idx[x[1]], date=x[2]['date'])
#         # assert (len(G.nodes()) == len(slices_graph[slice_id].nodes()))
#         assert (len(G.edges()) == len(slices_graph[slice_id].edges()))
#         slices_graph_remap.append(G)
#     print('generate snapshots list complete!')

#     # # generate feature list
#     # for slice_id in slices_features:
#     #     features_remap = []
#     #     for x in slices_graph_remap[slice_id].nodes():
#     #         features_remap.append(slices_features[slice_id][idx_node[x]])
#     #         #features_remap.append(np.array(slices_features[slice_id][idx_node[x]]).flatten())
#     #     features_remap = csr_matrix(np.squeeze(np.array(features_remap)))
#     #     slices_features_remap.append(features_remap)
#     # print('generate feature list complete!')
#     return (slices_graph_remap, slices_features_remap)


# # compute the differences between two snapshots
# def comparison(snapshot_A, snapshot_B, all_nodes):
#     A = nx.DiGraph()
#     B = nx.DiGraph()

#     A.add_nodes_from(all_nodes)
#     B.add_nodes_from(all_nodes)

#     A.add_edges_from(snapshot_A.edges())
#     B.add_edges_from(snapshot_B.edges())

#     Dif = nx.symmetric_difference(A,B)
#     return len(Dif.edges())

# def remap_new(slices_graphs, slices_features):
#     slices_links_remap = []
#     slices_features_remap = []
#     for slice_id in slices_graphs:
#         slices_links_remap.append(slices_graphs[slice_id])

#     # TODO: remap for features

#     return slices_links_remap, slices_features_remap

# slices_links_remap, slices_features_remap = remap(slices_links, slices_features) # graphs dict to graphs list
# # slices_links_remap, slices_features_remap = remap(slices_links, slices_features)
# # slices_links_remap = slices_links

# # Links=[]
# # Nodes = []
# # Differences = []
# # for i in range (len(slices_links_remap)):
# #     Nodes.append(len(slices_links_remap[i].nodes()))
# #     Links.append(len(slices_links_remap[i].edges()))
# #     # temp = []
# #     # for j in range (len(slices_links_remap)):
# #     #     temp.append(comparison(slices_links[i], slices_links[j], all_nodes))
# #     # Differences.append(temp)
# # print(Links,Nodes)

# np.savez(save_graph_path, graph=slices_links_remap)  # graph为字典的key
# # np.savez(save_features_path, feats=slices_features_remap)
