import networkx as nx
import random

def graph_concat(graphs):
    """
    Function to concatenate snapshots along the temporal dimension
    :param graphs: list of snapshots
    """
    G = nx.Graph()

    # add nodes and edges
    for i in range(len(graphs)):
        snap_id = i
        node_idx = {node: {'orig_id': node} for node in list(graphs[i].nodes())}
        nx.set_node_attributes(graphs[i], snap_id, "snap_id")
        nx.set_node_attributes(graphs[i], node_idx)
        # print('Snapshot {} has nodes {} and edges {}'.format(i, graphs[i].number_of_nodes(), graphs[i].number_of_edges()))
        # for k in range(10):
        #     u = random.randint(0, len(list(graphs[i].nodes())) -1)
        #     v = random.randint(0, len(list(graphs[i].nodes())) -1)
        #     graphs[i].add_edge(u, v)
        attr = {edge: {"type": 'str'} for edge in graphs[i].edges()}
        nx.set_edge_attributes(graphs[i], attr)
        G = nx.disjoint_union(G, graphs[i])
    for node in list(G):
        snap_id = G.nodes[node]['snap_id']
        tem_idx = 0
        # for i in range(snap_id):
        if snap_id > 0:
            tem_neighbor = node - graphs[snap_id - 1].number_of_nodes() - tem_idx
            tem_idx += graphs[snap_id - 1].number_of_nodes()
            if G.nodes[tem_neighbor]['snap_id'] == snap_id - 1:
                G.add_edge(tem_neighbor, node)
                attr = {(tem_neighbor, node): {"type": 'tem'}}
                nx.set_edge_attributes(G, attr)
    
    # print(G.nodes[60]['orig_id'])
    # print(G.number_of_nodes(), G.number_of_edges())
    return G