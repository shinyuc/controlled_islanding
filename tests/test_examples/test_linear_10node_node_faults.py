"""Tests for examples/linear_10node/linear_10node.py's NODE_AND_EDGE_FAULTS
setup. None of these call solve() -- they check the setup, not the optimizer.
"""

import numpy as np

from examples.linear_10node.linear_10node import (
    NODES, EDGES, NODE_AND_EDGE_FAULTS,
    build_topology, build_M, faulty_nodes, faulty_edges,
)


def _canon(edge):
    i, j = edge
    return (i, j) if i < j else (j, i)


def test_faulty_nodes_and_edges():
    assert faulty_nodes(NODE_AND_EDGE_FAULTS) == [10]
    assert faulty_edges(NODE_AND_EDGE_FAULTS) == [(1, 2), (7, 10)]


def test_full_system_is_genuinely_unstable():
    topology = build_topology()
    M = build_M(
        NODES, list(topology.edges()),
        NODE_AND_EDGE_FAULTS.node_weights, NODE_AND_EDGE_FAULTS.edge_weights,
    )
    lambda_min = np.linalg.eigvalsh(M).min()
    assert lambda_min <= 0


def test_node_10_cannot_be_rescued():
    # Largest island containing node 10 and all its neighbours (6, 7, 9):
    # even at full coupling, node 10's diagonal is
    # w_10 + w_6,10 + w_7,10 + w_9,10 = -5 + 1 - 3 + 1 = -6 < 0.
    unit_nodes = [6, 7, 9, 10]
    induced_edges = [
        (u, v) for u, v in EDGES if u in unit_nodes and v in unit_nodes
    ]
    M = build_M(
        unit_nodes, induced_edges,
        NODE_AND_EDGE_FAULTS.node_weights, NODE_AND_EDGE_FAULTS.edge_weights,
    )
    lambda_min = np.linalg.eigvalsh(M).min()
    assert lambda_min <= 0

    # Isolated too: M = [-5].
    M_isolated = build_M(
        [10], [],
        NODE_AND_EDGE_FAULTS.node_weights, NODE_AND_EDGE_FAULTS.edge_weights,
    )
    lambda_min_isolated = np.linalg.eigvalsh(M_isolated).min()
    assert lambda_min_isolated <= 0


def test_rest_of_network_is_fine_without_node_10():
    # Nodes 1-9, with both faulty edges (1,2) and (7,10) removed -- (7,10)
    # is already excluded since node 10 isn't in this unit, but drop it
    # explicitly too for clarity/robustness.
    unit_nodes = [n for n in NODES if n != 10]
    excluded = {(1, 2), (7, 10)}
    induced_edges = [
        (u, v) for u, v in EDGES
        if u in unit_nodes and v in unit_nodes and _canon((u, v)) not in excluded
    ]
    M = build_M(
        unit_nodes, induced_edges,
        NODE_AND_EDGE_FAULTS.node_weights, NODE_AND_EDGE_FAULTS.edge_weights,
    )
    lambda_min = np.linalg.eigvalsh(M).min()
    assert lambda_min > 0
