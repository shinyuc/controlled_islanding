"""Undirected network topology: nodes and edges only, no dynamics."""

import networkx as nx


class Topology:
    def __init__(self, nodes, edges):
        self._graph = nx.Graph()
        self._graph.add_nodes_from(nodes)
        self._graph.add_edges_from(edges)

    def nodes(self):
        return self._graph.nodes()

    def edges(self):
        return self._graph.edges()

    def neighbors(self, node):
        return self._graph.neighbors(node)
