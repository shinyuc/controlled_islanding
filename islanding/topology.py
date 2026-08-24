"""Undirected network topology: nodes and edges only, no dynamics."""

import networkx as nx


class Topology:
    def __init__(self, nodes, edges):
        self._graph = nx.Graph()
        self._graph.add_nodes_from(nodes)
        self._graph.add_edges_from(edges)
        # Memo for islanding.cost._induced_edges, keyed by node frozenset.
        # Underscore-private: it is an implementation detail of that
        # function, not part of Topology's interface.
        # Held here so its lifetime matches this topology's exactly: a
        # module-level cache keyed by id() would collide once an id is
        # reused after garbage collection, serving one graph's edges for
        # another. Safe because a Topology is never mutated after
        # construction -- there are no methods that add or remove edges.
        self._induced_edge_cache = {}

    def nodes(self):
        return self._graph.nodes()

    def edges(self):
        return self._graph.edges()

    def neighbors(self, node):
        return self._graph.neighbors(node)
