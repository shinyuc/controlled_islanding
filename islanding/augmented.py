"""Augmented graph construction (Section IV-B).

Adds a source s0 and a failed-node sink d0 to the original topology;
these are algorithmic artifacts used by the islanding algorithm, not
physical nodes.
"""

import networkx as nx

SOURCE = "s0"
SINK = "d0"


def augmented_graph(topology):
    graph = nx.Graph()
    nodes = list(topology.nodes())
    graph.add_nodes_from(nodes)
    graph.add_edges_from(topology.edges())

    graph.add_node(SOURCE)
    graph.add_node(SINK)
    graph.add_edge(SOURCE, SINK)
    for node in nodes:
        graph.add_edge(SOURCE, node)
        graph.add_edge(SINK, node)

    return graph
