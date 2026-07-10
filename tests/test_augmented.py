"""Tests for the augmented graph Gbar (Section IV-B)."""

from islanding.augmented import augmented_graph, SOURCE, SINK


def test_node_count(toy10_topology):
    n = len(list(toy10_topology.nodes()))
    gbar = augmented_graph(toy10_topology)
    assert gbar.number_of_nodes() == n + 2


def test_edge_count(toy10_topology):
    n = len(list(toy10_topology.nodes()))
    num_edges = len(list(toy10_topology.edges()))
    gbar = augmented_graph(toy10_topology)
    assert gbar.number_of_edges() == num_edges + 1 + 2 * n


def test_source_sink_edge_present(toy10_topology):
    gbar = augmented_graph(toy10_topology)
    assert gbar.has_edge(SOURCE, SINK)


def test_source_and_sink_adjacent_to_all_original_nodes(toy10_topology):
    nodes = list(toy10_topology.nodes())
    gbar = augmented_graph(toy10_topology)
    for node in nodes:
        assert gbar.has_edge(SOURCE, node)
        assert gbar.has_edge(SINK, node)


def test_original_topology_unchanged(toy10_topology):
    n = len(list(toy10_topology.nodes()))
    augmented_graph(toy10_topology)

    assert len(list(toy10_topology.nodes())) == n
    assert SOURCE not in toy10_topology.nodes()
    assert SINK not in toy10_topology.nodes()
