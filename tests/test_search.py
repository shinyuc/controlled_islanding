"""Tests for initial_spanning_tree, decode_islanding, and decode_reconnection."""

import networkx as nx

from islanding.topology import Topology
from islanding.augmented import augmented_graph, SOURCE, SINK
from islanding.search import (
    initial_spanning_tree, decode_islanding, decode_reconnection,
)


class TestInitialSpanningTree:
    def test_valid_spanning_tree(self, toy10_topology):
        gbar = augmented_graph(toy10_topology)
        tree = initial_spanning_tree(gbar, seed=0)

        assert tree.number_of_edges() == gbar.number_of_nodes() - 1
        assert nx.is_connected(tree)
        assert nx.is_tree(tree)

    def test_reproducible_with_same_seed(self, toy10_topology):
        gbar = augmented_graph(toy10_topology)
        tree_a = initial_spanning_tree(gbar, seed=0)
        tree_b = initial_spanning_tree(gbar, seed=0)

        edges_a = {frozenset(e) for e in tree_a.edges()}
        edges_b = {frozenset(e) for e in tree_b.edges()}
        assert edges_a == edges_b

    def test_different_seeds_both_valid_trees(self, toy10_topology):
        gbar = augmented_graph(toy10_topology)
        tree_0 = initial_spanning_tree(gbar, seed=0)
        tree_1 = initial_spanning_tree(gbar, seed=1)

        for tree in (tree_0, tree_1):
            assert tree.number_of_edges() == gbar.number_of_nodes() - 1
            assert nx.is_connected(tree)
            assert nx.is_tree(tree)

    def test_tree_always_contains_source_sink_edge(self, toy10_topology):
        gbar = augmented_graph(toy10_topology)

        tree_0 = initial_spanning_tree(gbar, seed=0)
        tree_1 = initial_spanning_tree(gbar, seed=1)

        assert tree_0.has_edge(SOURCE, SINK)
        assert tree_1.has_edge(SOURCE, SINK)

    def test_gbar_unmutated(self, toy10_topology):
        gbar = augmented_graph(toy10_topology)
        node_count_before = gbar.number_of_nodes()
        edge_count_before = gbar.number_of_edges()

        initial_spanning_tree(gbar, seed=0)

        assert gbar.number_of_nodes() == node_count_before
        assert gbar.number_of_edges() == edge_count_before


class TestDecodeIslanding:
    def test_decode_islanding_hand_built_tree(self):
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

        stable_islands, unstable_islands = decode_islanding(tree, topology)

        assert stable_islands == [frozenset({0, 1})]
        assert unstable_islands == [frozenset({2})]

    def test_decode_islanding_invariants_on_toy10(self, toy10_topology):
        gbar = augmented_graph(toy10_topology)
        tree = initial_spanning_tree(gbar, seed=0)
        nodes_before = set(tree.nodes())
        edges_before = {frozenset(e) for e in tree.edges()}

        stable_islands, unstable_islands = decode_islanding(tree, toy10_topology)

        original_nodes = set(toy10_topology.nodes())

        # Partition is complete and pairwise disjoint.
        covered = set()
        for island in stable_islands + unstable_islands:
            assert covered.isdisjoint(island)
            covered |= island
        assert covered == original_nodes

        # No relay/SOURCE/SINK leakage.
        for island in stable_islands + unstable_islands:
            assert all(isinstance(node, int) for node in island)

        # Each side is sorted by each island's min node.
        assert stable_islands == sorted(stable_islands, key=min)
        assert unstable_islands == sorted(unstable_islands, key=min)

        # decode_islanding does not mutate tree.
        assert set(tree.nodes()) == nodes_before
        assert {frozenset(e) for e in tree.edges()} == edges_before

    def test_decode_islanding_two_islands_two_failed_groups(self):
        # Original nodes 1..7. Two intact relay paths off SOURCE build islands
        # {1,2} and {3,4}. Node 5 hangs directly off SINK; node 6 hangs
        # directly off SINK too, with an intact relay path 6-7 chained off
        # it -- two distinct, disconnected "sub-groups" on the SINK side:
        # {5} and {6,7}.
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

        stable_islands, unstable_islands = decode_islanding(tree, topology)

        assert stable_islands == [frozenset({1, 2}), frozenset({3, 4})]
        assert unstable_islands == [frozenset({5}), frozenset({6, 7})]

        # Union of unstable_islands equals the old flat Z.
        flat_unstable = frozenset().union(*unstable_islands)
        assert flat_unstable == frozenset({5, 6, 7})

    def test_decode_islanding_all_stable_single_island(self):
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

        stable_islands, unstable_islands = decode_islanding(tree, topology)

        assert stable_islands == [frozenset({1, 2, 3})]
        assert unstable_islands == []

    def test_decode_islanding_all_failed_empty_stable_side(self):
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

        stable_islands, unstable_islands = decode_islanding(tree, topology)

        assert stable_islands == []
        assert unstable_islands == [frozenset({1, 2, 3})]

    def test_decode_islanding_connected_graph_with_cut_edge(self):
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

        stable_islands, unstable_islands = decode_islanding(tree, topology)

        assert stable_islands == [frozenset({1, 2})]
        assert unstable_islands == [frozenset({3})]


def _common_tree_edges():
    return [
        (SOURCE, SINK), (SOURCE, 1), (SOURCE, 4),
        (1, ("n", 1, 2, 1)), (("n", 1, 2, 1), ("n", 1, 2, 2)), (("n", 1, 2, 2), 2),
        (1, ("n", 1, 3, 1)), (("n", 1, 3, 1), ("n", 1, 3, 3)), (("n", 1, 3, 3), 3),
        (2, ("n", 2, 3, 2)), (("n", 2, 3, 2), ("n", 2, 3, 3)),
        (4, ("n", 4, 5, 4)), (("n", 4, 5, 4), ("n", 4, 5, 5)), (("n", 4, 5, 5), 5),
        (2, ("n", 2, 5, 2)), (("n", 2, 5, 2), ("n", 2, 5, 5)),
    ]


class TestDecodeReconnection:
    def test_reconnect_and_2_5_exclusion_crux(self):
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

        components, unstable_islands = decode_reconnection(tree, topology)

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
        assert unstable_islands == []

    def test_no_reconnection_two_components(self):
        # (1,4)'s relay path connects a-b but not b-v (r14_4 never reaches 4)
        # -> not middle-cut -> no reconnection -> islands stay separate.
        topology = Topology(nodes=[1, 2, 3, 4, 5],
                             edges=[(1, 2), (1, 3), (2, 3), (4, 5), (1, 4), (2, 5)])

        tree = nx.Graph()
        tree.add_edges_from(_common_tree_edges() + [
            (1, ("n", 1, 4, 1)), (("n", 1, 4, 1), ("n", 1, 4, 4)),
        ])

        components, unstable_islands = decode_reconnection(tree, topology)

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

        assert unstable_islands == []

    def test_invariants_on_toy10(self, toy10_topology):
        gbar = augmented_graph(toy10_topology)
        tree = initial_spanning_tree(gbar, seed=0)
        tree_nodes_before = set(tree.nodes())
        tree_edges_before = {frozenset(e) for e in tree.edges()}
        topology_nodes_before = list(toy10_topology.nodes())
        topology_edges_before = list(toy10_topology.edges())

        components, unstable_islands = decode_reconnection(tree, toy10_topology)

        original = set(toy10_topology.nodes())
        topology_edge_set = {frozenset(e) for e in toy10_topology.edges()}

        # Component node-sets are pairwise disjoint; union with the unstable
        # islands covers all original nodes.
        covered = frozenset().union(*unstable_islands) if unstable_islands else frozenset()
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

        # unstable_islands matches decode_islanding's unstable_islands.
        assert unstable_islands == decode_islanding(tree, toy10_topology)[1]

        # Components sorted by min node.
        min_nodes = [min(component.nodes()) for component in components]
        assert min_nodes == sorted(min_nodes)

        # No mutation of tree or topology.
        assert set(tree.nodes()) == tree_nodes_before
        assert {frozenset(e) for e in tree.edges()} == tree_edges_before
        assert list(toy10_topology.nodes()) == topology_nodes_before
        assert list(toy10_topology.edges()) == topology_edges_before
