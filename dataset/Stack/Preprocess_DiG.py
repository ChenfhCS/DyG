# import dill
from collections import defaultdict
from datetime import datetime, timedelta
import os

current_path = os.getcwd()
folder_in = os.path.exists(current_path + '/data/')
if not folder_in:
    os.makedirs(current_path + '/data/')

links = []
ts = []
ctr = 0
node_cnt = 0
node_idx = {}
idx_node = []

file_path = current_path + '/sx-stackoverflow.edges'
save_graph_path = current_path + '/data/graphs.npz'
save_features_path = current_path + '/data/features.npz'

with open(file_path) as f:
    lines = f.read().splitlines()
    for l in lines:
        if l[0] == '%':
            continue

        x, y, t = map(int, l.split(' '))
        # print (x,y,e,t)
        timestamp = datetime.fromtimestamp(t)
        ts.append(timestamp)

        ctr += 1
        if ctr % 100000 == 0:
            print (ctr)

        if x not in node_idx:
            node_idx[x] = node_cnt
            node_cnt += 1

        if y not in node_idx:
            node_idx[y] = node_cnt
            node_cnt += 1

        links.append((node_idx[x],node_idx[y], timestamp))

print ("Min ts", min(ts), "max ts", max(ts))
print ("Total time span: {} days".format((max(ts) - min(ts)).days))
links.sort(key =lambda x: x[2])
print ("# temporal links", len(links))

# import networkx as nx
# agg_G = nx.Graph()
# for a,b,t in links:
#     agg_G.add_edge(a,b)

# print ("Agg graph", len(agg_G.nodes()), len(agg_G.edges()))

import networkx as nx
import numpy as np
from datetime import datetime, timedelta
'''
collect data from 'START_DATE' and ends to 'END_DATE'.
generate a graph per 'SLICE_DAYS'.
'''
# slice defaule = 30
SLICE_DAYS = 30
START_DATE = min(ts)
END_DATE = min(ts) + timedelta(1000)

print ("Start date", START_DATE)
print ("End date", END_DATE)

slices_links = defaultdict(lambda : nx.DiGraph())
slices_features = defaultdict(lambda : {})

slice_id = -1
snapshot_id = 0
# Split the set of links in order by slices to create the graphs.
for (a, b, time) in links:
    prev_slice_id = slice_id

    datetime_object = time
    if datetime_object < START_DATE:
        continue
    if datetime_object > END_DATE:
        break
        days_diff = (END_DATE - START_DATE).days
    else:
        days_diff = (datetime_object - START_DATE).days


    slice_id = days_diff // SLICE_DAYS

    if slice_id == 1+prev_slice_id and slice_id > 0:
        snapshot_id += 1
        slices_links[snapshot_id] = nx.DiGraph()
        slices_links[snapshot_id].add_nodes_from(slices_links[snapshot_id-1].nodes(data=True))
        # assert (len(slices_links[snapshot_id].edges()) ==0)
        #assert len(slices_links[slice_id].nodes()) >0

    if slice_id == 1+prev_slice_id and slice_id ==0:
        slices_links[snapshot_id] = nx.DiGraph()

    # if days_diff % SLICE_DAYS == 7 or days_diff % SLICE_DAYS == 6 or days_diff % SLICE_DAYS == 5:
    #     if a not in slices_links[slice_id]:
    #         slices_links[slice_id].add_node(a)
    #     if b not in slices_links[slice_id]:
    #         slices_links[slice_id].add_node(b)
    #     slices_links[slice_id].add_edge(a,b, date=datetime_object)

    if a not in slices_links[snapshot_id]:
        slices_links[snapshot_id].add_node(a)
    if b not in slices_links[snapshot_id]:
        slices_links[snapshot_id].add_node(b)
    slices_links[snapshot_id].add_edge(a,b, date=datetime_object)

for slice_id in slices_links:
    print ("# nodes in slice", slice_id, len(slices_links[slice_id].nodes()))
    print ("# edges in slice", slice_id, len(slices_links[slice_id].edges()))

    # temp = np.identity(len(slices_links[max(slices_links.keys())].nodes()))
    # print ("Shape of temp matrix", temp.shape)
    slices_features[slice_id] = {}
    # for idx, node in enumerate(slices_links[slice_id].nodes()):
    #     slices_features[slice_id][node] = temp[idx]


# TODO : remap and output.
from scipy.sparse import csr_matrix

def remap(slices_graphs, slices_features):
    snapshots = []
    slices_features_remap = []
    for slices_id in slices_graphs:
        slices_graph = slices_graphs[slices_id]
        node_mapping = {n: i for i, n in enumerate(slices_graph.nodes())}
        slices_graph_remap = nx.relabel_nodes(slices_graph, node_mapping, copy=True)
        snapshots.append(slices_graph_remap)
    return snapshots, slices_features_remap

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

slices_links_remap, slices_features_remap = remap(slices_links, slices_features) # graphs dict to graphs list
# # slices_links_remap, slices_features_remap = remap(slices_links, slices_features)
# # slices_links_remap = slices_links

# Links=[]
# Nodes = []
# Differences = []
# for i in range (len(slices_links_remap)):
#     Nodes.append(len(slices_links_remap[i].nodes()))
#     Links.append(len(slices_links_remap[i].edges()))
#     # temp = []
#     # for j in range (len(slices_links_remap)):
#     #     temp.append(comparison(slices_links[i], slices_links[j], all_nodes))
#     # Differences.append(temp)
# print(Links,Nodes)

np.savez(save_graph_path, graph=slices_links_remap)  # graph为字典的key
# np.savez(save_features_path, feats=slices_features_remap)
