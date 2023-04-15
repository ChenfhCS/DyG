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
file_path = current_path + '/rec-amazon-ratings.edges'
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
            nodes.append(snapshots[i].number_of_nodes() - snapshots[i - 1].number_of_nodes())
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
    # _graph_profile(117)
