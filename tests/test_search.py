"""Tests for initial_spanning_tree and decode_islanding."""

import networkx as nx

from islanding.topology import Topology
from islanding.augmented import augmented_graph, SOURCE, SINK
from islanding.search import initial_spanning_tree, decode_islanding


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


def test_tree_always_contains_source_sink_edge(toy10_topology):
    gbar = augmented_graph(toy10_topology)

    tree_0 = initial_spanning_tree(gbar, seed=0)
    tree_1 = initial_spanning_tree(gbar, seed=1)

    assert tree_0.has_edge(SOURCE, SINK)
    assert tree_1.has_edge(SOURCE, SINK)


def test_gbar_unmutated(toy10_topology):
    gbar = augmented_graph(toy10_topology)
    node_count_before = gbar.number_of_nodes()
    edge_count_before = gbar.number_of_edges()

    initial_spanning_tree(gbar, seed=0)

    assert gbar.number_of_nodes() == node_count_before
    assert gbar.number_of_edges() == edge_count_before


def test_decode_islanding_hand_built_tree():
    # Path graph 0-1-2. In the augmented graph, edge (0,1) becomes the
    # relay path 0-("n",0,1,0)-("n",0,1,1)-1, and (1,2) becomes
    # 1-("n",1,2,1)-("n",1,2,2)-2.
    #
    # The hand-built tree below keeps the (0,1) relay path fully intact
    # (0 and 1 end up in one island), attaches node 2 directly to SINK
    # (node 2 is failed), and only half-connects the (1,2) relay path
    # (("n",1,2,1) is a dead-end leaf, never reaching ("n",1,2,2)), i.e.
    # edge (1,2) is effectively cut.
    topology = Topology(nodes=[0, 1, 2], edges=[(0, 1), (1, 2)])
    augmented_graph(topology)  # sanity: construction succeeds for this graph

    r01_0 = ("n", 0, 1, 0)
    r01_1 = ("n", 0, 1, 1)
    r12_1 = ("n", 1, 2, 1)
    r12_2 = ("n", 1, 2, 2)

    tree = nx.Graph()
    tree.add_edges_from([
        (SOURCE, SINK),
        (SOURCE, 0),
        (0, r01_0),
        (r01_0, r01_1),
        (r01_1, 1),
        (1, r12_1),
        (SINK, 2),
        (r12_2, 2),
    ])

    islands, Z = decode_islanding(tree, topology)

    assert islands == [frozenset({0, 1})]
    assert Z == frozenset({2})


def test_decode_islanding_invariants_on_toy10(toy10_topology):
    gbar = augmented_graph(toy10_topology)
    tree = initial_spanning_tree(gbar, seed=0)
    nodes_before = set(tree.nodes())
    edges_before = {frozenset(e) for e in tree.edges()}

    islands, Z = decode_islanding(tree, toy10_topology)

    original_nodes = set(toy10_topology.nodes())

    # Partition is complete and pairwise disjoint.
    covered = set(Z)
    for island in islands:
        assert covered.isdisjoint(island)
        covered |= island
    assert covered == original_nodes

    # No relay/SOURCE/SINK leakage.
    for island in islands:
        assert all(isinstance(node, int) for node in island)
    assert all(isinstance(node, int) for node in Z)

    # islands is sorted by each island's min node.
    assert islands == sorted(islands, key=min)

    # decode_islanding does not mutate tree.
    assert set(tree.nodes()) == nodes_before
    assert {frozenset(e) for e in tree.edges()} == edges_before


def test_decode_islanding_two_islands_two_failed_groups():
    # Original nodes 1..7. Two intact relay paths off SOURCE build islands
    # {1,2} and {3,4}. Node 5 hangs directly off SINK; node 6 hangs
    # directly off SINK too, with an intact relay path 6-7 chained off
    # it -- two distinct "sub-groups" on the SINK side that both collapse
    # into the single failed set Z once SOURCE-SINK is removed.
    topology = Topology(nodes=[1, 2, 3, 4, 5, 6, 7],
                         edges=[(1, 2), (3, 4), (6, 7)])

    r12_1, r12_2 = ("n", 1, 2, 1), ("n", 1, 2, 2)
    r34_3, r34_4 = ("n", 3, 4, 3), ("n", 3, 4, 4)
    r67_6, r67_7 = ("n", 6, 7, 6), ("n", 6, 7, 7)

    tree = nx.Graph()
    tree.add_edges_from([
        (SOURCE, SINK),
        (SOURCE, 1), (1, r12_1), (r12_1, r12_2), (r12_2, 2),
        (SOURCE, 3), (3, r34_3), (r34_3, r34_4), (r34_4, 4),
        (SINK, 5),
        (SINK, 6), (6, r67_6), (r67_6, r67_7), (r67_7, 7),
    ])

    islands, Z = decode_islanding(tree, topology)

    assert islands == [frozenset({1, 2}), frozenset({3, 4})]
    assert Z == frozenset({5, 6, 7})


def test_decode_islanding_all_stable_single_island():
    # Original nodes 1..3, chained by two intact relay paths off SOURCE
    # (1-2 and 2-3); nothing attaches to SINK, so Z is empty.
    topology = Topology(nodes=[1, 2, 3], edges=[(1, 2), (2, 3)])

    r12_1, r12_2 = ("n", 1, 2, 1), ("n", 1, 2, 2)
    r23_2, r23_3 = ("n", 2, 3, 2), ("n", 2, 3, 3)

    tree = nx.Graph()
    tree.add_edges_from([
        (SOURCE, SINK),
        (SOURCE, 1), (1, r12_1), (r12_1, r12_2), (r12_2, 2),
        (2, r23_2), (r23_2, r23_3), (r23_3, 3),
    ])

    islands, Z = decode_islanding(tree, topology)

    assert islands == [frozenset({1, 2, 3})]
    assert Z == frozenset()


def test_decode_islanding_all_failed_empty_stable_side():
    # Original nodes 1..3, chained by two intact relay paths off SINK
    # (1-2 and 2-3); SOURCE has no other attachments, so the stable side
    # is empty and islands == [].
    topology = Topology(nodes=[1, 2, 3], edges=[(1, 2), (2, 3)])

    r12_1, r12_2 = ("n", 1, 2, 1), ("n", 1, 2, 2)
    r23_2, r23_3 = ("n", 2, 3, 2), ("n", 2, 3, 3)

    tree = nx.Graph()
    tree.add_edges_from([
        (SOURCE, SINK),
        (SINK, 1), (1, r12_1), (r12_1, r12_2), (r12_2, 2),
        (2, r23_2), (r23_2, r23_3), (r23_3, 3),
    ])

    islands, Z = decode_islanding(tree, topology)

    assert islands == []
    assert Z == frozenset({1, 2, 3})

def test_decode_islanding_connected_graph_with_cut_edge():
    # Original graph 1-2-3 is CONNECTED. The tree keeps relay path (1,2)
    # intact but CUTS (2,3): its middle relay edge (r23_2, r23_3) is
    # omitted, so node 3's relay attaches to 3 while r23_2 stays on 2's
    # side. Result: island {1,2} stable, node 3 pushed to the SINK side.
    topology = Topology(nodes=[1, 2, 3], edges=[(1, 2), (2, 3)])

    r12_1, r12_2 = ("n", 1, 2, 1), ("n", 1, 2, 2)
    r23_2, r23_3 = ("n", 2, 3, 2), ("n", 2, 3, 3)

    tree = nx.Graph()
    tree.add_edges_from([
        (SOURCE, SINK),
        (SOURCE, 1), (1, r12_1), (r12_1, r12_2), (r12_2, 2),
        (2, r23_2),              # r23_2 stays on the stable side via node 2
        (SINK, 3), (3, r23_3),   # (2,3) is cut: node 3 and its relay hang off SINK
    ])

    islands, Z = decode_islanding(tree, topology)

    assert islands == [frozenset({1, 2})]
    assert Z == frozenset({3})