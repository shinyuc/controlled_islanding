"""Tests for decode_reconnection, using hand-built trees plus toy10 invariants."""

import networkx as nx

from islanding.topology import Topology
from islanding.augmented import augmented_graph, SOURCE, SINK
from islanding.search import (
    initial_spanning_tree, decode_islanding, decode_reconnection,
)


def _common_tree_edges():
    return [
        (SOURCE, SINK), (SOURCE, 1), (SOURCE, 4),
        (1, ("n", 1, 2, 1)), (("n", 1, 2, 1), ("n", 1, 2, 2)), (("n", 1, 2, 2), 2),
        (1, ("n", 1, 3, 1)), (("n", 1, 3, 1), ("n", 1, 3, 3)), (("n", 1, 3, 3), 3),
        (2, ("n", 2, 3, 2)), (("n", 2, 3, 2), ("n", 2, 3, 3)),
        (4, ("n", 4, 5, 4)), (("n", 4, 5, 4), ("n", 4, 5, 5)), (("n", 4, 5, 5), 5),
        (2, ("n", 2, 5, 2)), (("n", 2, 5, 2), ("n", 2, 5, 5)),
    ]


def test_reconnect_and_2_5_exclusion_crux():
    # (1,4)'s relay path is middle-cut (u-a and b-v present, a-b missing)
    # -> islands {1,2,3} and {4,5} reconnect via edge (1,4). (2,5)'s relay
    # path is cut on the v-side (b-v missing), so it is NOT reconnected,
    # and since 2 and 5 end up in different islands it must not be
    # readmitted just because the two islands merge into one component.
    topology = Topology(nodes=[1, 2, 3, 4, 5],
                         edges=[(1, 2), (1, 3), (2, 3), (4, 5), (1, 4), (2, 5)])

    tree = nx.Graph()
    tree.add_edges_from(_common_tree_edges() + [
        (1, ("n", 1, 4, 1)), (("n", 1, 4, 4), 4),
    ])

    components, Z = decode_reconnection(tree, topology)

    assert len(components) == 1
    c = components[0]
    assert set(c.nodes()) == {1, 2, 3, 4, 5}
    assert {frozenset(e) for e in c.edges()} == {
        frozenset({1, 2}), frozenset({1, 3}), frozenset({2, 3}),
        frozenset({4, 5}), frozenset({1, 4}),
    }
    assert not c.has_edge(2, 5)
    assert set(c.neighbors(2)) == {1, 3}
    assert set(c.neighbors(5)) == {4}
    assert c.graph["islands"] == (frozenset({1, 2, 3}), frozenset({4, 5}))
    assert Z == frozenset()


def test_no_reconnection_two_components():
    # (1,4)'s relay path connects a-b but not b-v (r14_4 never reaches 4)
    # -> not middle-cut -> no reconnection -> islands stay separate.
    topology = Topology(nodes=[1, 2, 3, 4, 5],
                         edges=[(1, 2), (1, 3), (2, 3), (4, 5), (1, 4), (2, 5)])

    tree = nx.Graph()
    tree.add_edges_from(_common_tree_edges() + [
        (1, ("n", 1, 4, 1)), (("n", 1, 4, 1), ("n", 1, 4, 4)),
    ])

    components, Z = decode_reconnection(tree, topology)

    assert len(components) == 2
    comp0, comp1 = components

    assert set(comp0.nodes()) == {1, 2, 3}
    assert {frozenset(e) for e in comp0.edges()} == {
        frozenset({1, 2}), frozenset({1, 3}), frozenset({2, 3}),
    }
    assert comp0.graph["islands"] == (frozenset({1, 2, 3}),)

    assert set(comp1.nodes()) == {4, 5}
    assert {frozenset(e) for e in comp1.edges()} == {frozenset({4, 5})}
    assert comp1.graph["islands"] == (frozenset({4, 5}),)

    assert Z == frozenset()


def test_invariants_on_toy10(toy10_topology):
    gbar = augmented_graph(toy10_topology)
    tree = initial_spanning_tree(gbar, seed=0)
    tree_nodes_before = set(tree.nodes())
    tree_edges_before = {frozenset(e) for e in tree.edges()}
    topology_nodes_before = list(toy10_topology.nodes())
    topology_edges_before = list(toy10_topology.edges())

    components, Z = decode_reconnection(tree, toy10_topology)

    original = set(toy10_topology.nodes())
    topology_edge_set = {frozenset(e) for e in toy10_topology.edges()}

    # Component node-sets are pairwise disjoint; union with Z covers all
    # original nodes.
    covered = set(Z)
    for component in components:
        nodes = set(component.nodes())
        assert covered.isdisjoint(nodes)
        covered |= nodes
    assert covered == original

    # Every component edge is an original edge of topology; no leakage.
    for component in components:
        for u, v in component.edges():
            assert isinstance(u, int) and isinstance(v, int)
            assert frozenset({u, v}) in topology_edge_set

    # Z matches decode_islanding's Z.
    assert Z == decode_islanding(tree, toy10_topology)[1]

    # Components sorted by min node.
    min_nodes = [min(component.nodes()) for component in components]
    assert min_nodes == sorted(min_nodes)

    # No mutation of tree or topology.
    assert set(tree.nodes()) == tree_nodes_before
    assert {frozenset(e) for e in tree.edges()} == tree_edges_before
    assert list(toy10_topology.nodes()) == topology_nodes_before
    assert list(toy10_topology.edges()) == topology_edges_before
