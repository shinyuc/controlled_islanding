"""Local search machinery for Algorithm 1: initial spanning tree seed."""

import networkx as nx


def initial_spanning_tree(gbar, seed=0):
    return nx.random_spanning_tree(gbar, seed=seed)
