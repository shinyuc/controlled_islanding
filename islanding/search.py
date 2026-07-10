"""Local search machinery for Algorithm 1: initial spanning tree seed and
decoding a spanning tree of Gbar into an islanding.
"""

import networkx as nx

from islanding.augmented import SOURCE, SINK


def initial_spanning_tree(gbar, seed=0):
    return nx.random_spanning_tree(gbar, seed=seed)


def decode(tree, topology):
    """Decode a spanning tree of Gbar into island labels and failed nodes.

    Removing the (SOURCE, SINK) edge splits the tree into a SOURCE-side
    component (stable nodes) and a SINK-side component (failed nodes, Z).
    Stable islands are the connected components formed by original edges
    (both endpoints in V) among the stable nodes; each gets a distinct
    positive label. All failed nodes share the reserved label 0.
    """
    split = tree.copy()
    split.remove_edge(SOURCE, SINK)
    components = list(nx.connected_components(split))
    sink_component = next(c for c in components if SINK in c)
    source_component = next(c for c in components if SOURCE in c)

    original_nodes = set(topology.nodes())
    Z = sink_component & original_nodes
    stable_nodes = source_component & original_nodes

    island_graph = nx.Graph()
    island_graph.add_nodes_from(stable_nodes)
    for u, v in tree.edges():
        if u in stable_nodes and v in stable_nodes:
            island_graph.add_edge(u, v)

    labels = {}
    for label, component in enumerate(nx.connected_components(island_graph), start=1):
        for node in component:
            labels[node] = label

    for node in Z:
        labels[node] = 0

    return labels, Z
