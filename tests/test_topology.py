"""Basic structural checks for Topology."""


def test_node_count(toy10_topology):
    assert len(list(toy10_topology.nodes())) == 10


def test_edge_count(toy10_topology):
    assert len(list(toy10_topology.edges())) == 15


def test_known_neighbors_node_0(toy10_topology):
    assert sorted(toy10_topology.neighbors(0)) == [1, 4, 5]


def test_known_neighbors_node_5(toy10_topology):
    assert sorted(toy10_topology.neighbors(5)) == [0, 7, 8]
