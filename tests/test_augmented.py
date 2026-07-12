"""Tests for the augmented graph Gbar (Section IV-B).

Original edges are replaced by length-3 paths through relay nodes
("n", i, j, i) / ("n", i, j, j); s0 and d0 attach only to original nodes.
"""

import pytest

from islanding.topology import Topology
from islanding.augmented import augmented_graph, SOURCE, SINK


@pytest.fixture
def path3_topology():
    # Tiny hand-checkable graph: path 0-1-2 (n=3, m=2).
    return Topology(nodes=[0, 1, 2], edges=[(0, 1), (1, 2)])


def _relay(i, j, endpoint):
    return ("n", i, j, endpoint)


def _check_counts(topology, gbar):
    n = len(list(topology.nodes()))
    m = len(list(topology.edges()))
    assert gbar.number_of_nodes() == n + 2 + 2 * m
    assert gbar.number_of_edges() == 1 + 2 * n + 3 * m


def _check_no_original_edges(topology, gbar):
    for i, j in topology.edges():
        assert not gbar.has_edge(i, j)


def _check_relay_paths(topology, gbar):
    for u, v in topology.edges():
        i, j = (u, v) if u < v else (v, u)
        relay_i = _relay(i, j, i)
        relay_j = _relay(i, j, j)

        assert gbar.has_edge(i, relay_i)
        assert gbar.has_edge(relay_i, relay_j)
        assert gbar.has_edge(relay_j, j)

        assert gbar.degree(relay_i) == 2
        assert gbar.degree(relay_j) == 2


def _check_original_node_degrees(topology, gbar):
    for node in topology.nodes():
        deg_g = len(list(topology.neighbors(node)))
        assert gbar.degree(node) == 2 + deg_g


def _check_supernodes(topology, gbar):
    nodes = list(topology.nodes())
    n = len(nodes)

    assert gbar.degree(SOURCE) == n + 1
    assert gbar.degree(SINK) == n + 1
    assert gbar.has_edge(SOURCE, SINK)

    for node in nodes:
        assert gbar.has_edge(SOURCE, node)
        assert gbar.has_edge(SINK, node)

    for neighbor in gbar.neighbors(SOURCE):
        assert not (isinstance(neighbor, tuple) and neighbor[0] == "n")
    for neighbor in gbar.neighbors(SINK):
        assert not (isinstance(neighbor, tuple) and neighbor[0] == "n")


def _check_topology_unchanged(topology, before_nodes, before_edges):
    assert list(topology.nodes()) == before_nodes
    assert list(topology.edges()) == before_edges


@pytest.mark.parametrize("topology_fixture", ["toy10_topology", "path3_topology"])
def test_augmented_graph_construction(topology_fixture, request):
    topology = request.getfixturevalue(topology_fixture)
    before_nodes = list(topology.nodes())
    before_edges = list(topology.edges())

    gbar = augmented_graph(topology)

    _check_counts(topology, gbar)
    _check_no_original_edges(topology, gbar)
    _check_relay_paths(topology, gbar)
    _check_original_node_degrees(topology, gbar)
    _check_supernodes(topology, gbar)
    _check_topology_unchanged(topology, before_nodes, before_edges)
