"""Tests for initial_spanning_tree."""

import networkx as nx

from islanding.augmented import augmented_graph
from islanding.search import initial_spanning_tree


def test_valid_spanning_tree(toy10_topology):
    gbar = augmented_graph(toy10_topology)
    tree = initial_spanning_tree(gbar, seed=0)

    assert tree.number_of_edges() == gbar.number_of_nodes() - 1
    assert nx.is_connected(tree)
    assert nx.is_tree(tree)


def test_reproducible_with_same_seed(toy10_topology):
    gbar = augmented_graph(toy10_topology)
    tree_a = initial_spanning_tree(gbar, seed=0)
    tree_b = initial_spanning_tree(gbar, seed=0)

    edges_a = {frozenset(e) for e in tree_a.edges()}
    edges_b = {frozenset(e) for e in tree_b.edges()}
    assert edges_a == edges_b


def test_different_seeds_both_valid_trees(toy10_topology):
    gbar = augmented_graph(toy10_topology)
    tree_0 = initial_spanning_tree(gbar, seed=0)
    tree_1 = initial_spanning_tree(gbar, seed=1)

    for tree in (tree_0, tree_1):
        assert tree.number_of_edges() == gbar.number_of_nodes() - 1
        assert nx.is_connected(tree)
        assert nx.is_tree(tree)
