"""Tests for examples/linear_10node/linear_10node.py (spec section 7). None
of these call solve() -- they check the setup, not the optimizer.
"""

import numpy as np

from examples.linear_10node.linear_10node import (
    NODES, EDGES, EDGE_FAULTS, build_M,
    build_topology, build_model, build_problem,
)
from examples.linear_10node.config import build_config


def _canon(edge):
    i, j = edge
    return (i, j) if i < j else (j, i)


def test_topology_has_10_nodes_17_edges_and_matching_degree_sequence():
    topology = build_topology()

    assert sorted(topology.nodes()) == list(range(1, 11))
    assert len(list(topology.edges())) == 17

    expected_degrees = {1: 2, 2: 4, 3: 2, 4: 3, 5: 5, 6: 5, 7: 3, 8: 3, 9: 4, 10: 3}
    for node, degree in expected_degrees.items():
        assert len(list(topology.neighbors(node))) == degree


def test_full_system_is_genuinely_unstable():
    # M over all 10 nodes and all 17 edges. If this were positive
    # definite, the whole problem would be vacuous -- nothing to island.
    M = build_M(NODES, EDGES, EDGE_FAULTS.node_weights, EDGE_FAULTS.edge_weights)
    lambda_min = np.linalg.eigvalsh(M).min()
    assert lambda_min <= 0


def test_faulty_edge_makes_its_island_unstable():
    # Unit ({1,2,4,5}, induced edges). Node 1's diagonal here is
    # w_1 + w_12 + w_14 = 1 - 3 + 1 = -1 < 0, so lambda_min <= -1 <= 0
    # (Rayleigh quotient with the standard basis vector for node 1).
    unit_nodes = [1, 2, 4, 5]
    induced_edges = [
        (u, v) for u, v in EDGES if u in unit_nodes and v in unit_nodes
    ]

    M = build_M(unit_nodes, induced_edges, EDGE_FAULTS.node_weights, EDGE_FAULTS.edge_weights)
    lambda_min = np.linalg.eigvalsh(M).min()
    assert lambda_min <= 0


def test_cutting_the_faulty_edge_fixes_the_island():
    # Same node set, but with (1,2) removed from the edge list -- node 1's
    # diagonal becomes 1 + 1 = 2 > 0, and (per the spec) the whole unit is
    # now genuinely stable.
    unit_nodes = [1, 2, 4, 5]
    induced_edges = [
        (u, v) for u, v in EDGES
        if u in unit_nodes and v in unit_nodes and _canon((u, v)) != (1, 2)
    ]

    M = build_M(unit_nodes, induced_edges, EDGE_FAULTS.node_weights, EDGE_FAULTS.edge_weights)
    lambda_min = np.linalg.eigvalsh(M).min()
    assert lambda_min > 0


def test_build_problem_draws_matching_sample_shapes():
    problem = build_problem(EDGE_FAULTS, build_topology(), build_config(), seed=0)

    assert problem.samples_1.shape == (20, 10, 2)
    assert problem.samples_2.shape == (20, 10, 2)
    assert problem.config.P_1 == 20
    assert problem.config.P_2 == 20
