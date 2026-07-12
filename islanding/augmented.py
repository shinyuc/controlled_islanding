"""Augmented graph construction (Section IV-B).

Adds a source s0 and a failed-node sink d0, which attach only to
original nodes. Each original edge (i, j) is not carried over directly:
it is replaced by a length-3 path through two relay nodes, one sitting
next to each endpoint. Supernodes and relays are algorithmic artifacts,
not physical nodes.
"""

import networkx as nx

SOURCE = "s0"
SINK = "d0"


def _relay(i, j, endpoint):
    return ("n", i, j, endpoint)


def augmented_graph(topology):
    graph = nx.Graph()
    nodes = list(topology.nodes())
    graph.add_nodes_from(nodes)

    graph.add_node(SOURCE)
    graph.add_node(SINK)
    graph.add_edge(SOURCE, SINK)
    for node in nodes:
        graph.add_edge(SOURCE, node)
        graph.add_edge(SINK, node)

    for u, v in topology.edges():
        i, j = (u, v) if u < v else (v, u)
        relay_i = _relay(i, j, i)
        relay_j = _relay(i, j, j)
        graph.add_edge(i, relay_i)
        graph.add_edge(relay_i, relay_j)
        graph.add_edge(relay_j, j)

    return graph
