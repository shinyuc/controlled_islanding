"""Linear consensus dynamics: signed edge weights on a topology."""

import numpy as np


class LinearConsensus:
    def __init__(self, topology, weights):
        self.topology = topology
        self.weights = weights

    def _weight(self, u, v):
        if (u, v) in self.weights:
            return self.weights[(u, v)]
        return self.weights[(v, u)]

    def _build_laplacian(self, nodes, edges):
        index = {node: i for i, node in enumerate(nodes)}
        n = len(nodes)
        L = np.zeros((n, n))
        for u, v in edges:
            w = self._weight(u, v)
            i, j = index[u], index[v]
            L[i, j] -= w
            L[j, i] -= w
            L[i, i] += w
            L[j, j] += w
        return L

    def laplacian(self):
        nodes = list(self.topology.nodes())
        return self._build_laplacian(nodes, self.topology.edges())

    def is_island_stable(self, nodes):
        """Stability certificate for island V_r (eq. 2, Section V-A).

        The island Laplacian is built from intra-island edges only --
        cut edges leaving the island are dropped entirely rather than
        just excluded from the row/column selection, so degrees are
        recomputed from the surviving edges alone. The island is stable
        iff this Laplacian is PSD.
        """
        node_list = list(nodes)
        node_set = set(node_list)
        intra_edges = [
            (u, v) for u, v in self.topology.edges()
            if u in node_set and v in node_set
        ]
        L = self._build_laplacian(node_list, intra_edges)
        return np.linalg.eigvalsh(L).min() >= -1e-9
