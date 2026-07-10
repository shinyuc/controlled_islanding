"""Tests for search.decode, using small explicit trees (not random)."""

import networkx as nx

from islanding.topology import Topology
from islanding.augmented import SOURCE, SINK
from islanding.search import decode


def test_two_islands_two_failed_components():
    # Original nodes 1..7. Tree edges:
    #   (1,2), (3,4), (6,7), (s0,d0), (s0,1), (s0,3), (d0,5), (d0,6)
    #
    # Remove (s0,d0): the tree splits into
    #   source-side component: {s0, 1, 2, 3, 4}   (s0-1, 1-2, s0-3, 3-4)
    #   sink-side component:   {d0, 5, 6, 7}       (d0-5, d0-6, 6-7)
    # so Z = {5, 6, 7}, stable nodes = {1, 2, 3, 4}.
    #
    # Original-only edges among stable nodes: (1,2) and (3,4) -- two
    # separate stable islands. (6,7) is excluded because 6,7 in Z.
    topology = Topology(nodes=[1, 2, 3, 4, 5, 6, 7], edges=[])

    tree = nx.Graph()
    tree.add_edges_from([
        (1, 2), (3, 4), (6, 7),
        (SOURCE, SINK), (SOURCE, 1), (SOURCE, 3),
        (SINK, 5), (SINK, 6),
    ])

    labels, Z = decode(tree, topology)

    assert Z == {5, 6, 7}
    assert labels[1] == labels[2]
    assert labels[3] == labels[4]
    assert labels[1] != labels[3]
    assert labels[5] == labels[6] == labels[7] == 0


def test_all_stable_single_island():
    # Original nodes 1..3, all on the source side, no failed nodes.
    # Tree edges: (s0,d0), (s0,1), (1,2), (2,3).
    topology = Topology(nodes=[1, 2, 3], edges=[])

    tree = nx.Graph()
    tree.add_edges_from([
        (SOURCE, SINK), (SOURCE, 1), (1, 2), (2, 3),
    ])

    labels, Z = decode(tree, topology)

    assert Z == set()
    assert labels[1] == labels[2] == labels[3]
    assert labels[1] != 0


def test_all_failed_empty_stable_side():
    # Original nodes 1..3. Tree edges: (s0,d0), (d0,1), (1,2), (2,3).
    #
    # Remove (s0,d0): the tree splits into
    #   source-side component: {s0}               (s0 has no other edges)
    #   sink-side component:   {d0, 1, 2, 3}       (d0-1, 1-2, 2-3)
    # so Z = {1, 2, 3}, stable nodes = {} (empty).
    #
    # With no stable nodes, there are no stable islands to label; every
    # original node is failed and gets the reserved label 0.
    topology = Topology(nodes=[1, 2, 3], edges=[])

    tree = nx.Graph()
    tree.add_edges_from([
        (SOURCE, SINK), (SINK, 1), (1, 2), (2, 3),
    ])

    labels, Z = decode(tree, topology)

    assert Z == {1, 2, 3}
    assert labels[1] == labels[2] == labels[3] == 0
