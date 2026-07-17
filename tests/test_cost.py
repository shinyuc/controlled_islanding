"""Tests for islanding/cost.py, reproducing the worked example in
docs/cost_function_spec.md section 6 (11-node, 5-island reconnection case).
"""

import pytest

from islanding.topology import Topology
from islanding.cost import (
    CostData, AlgorithmConfig, canon,
    f1_unstable_nodes, f2_unstable_edges, f3_operation_edges,
    f4_reconnection, F,
)


def _worked_example():
    edges = [
        (1, 2), (1, 3), (2, 3), (1, 4), (4, 5),
        (3, 5), (5, 7), (4, 6), (6, 7), (6, 8),
        (7, 9), (8, 9), (8, 10), (9, 11), (10, 11),
    ]
    topology = Topology(nodes=list(range(1, 12)), edges=edges)

    stable_islands = [frozenset({1, 2, 3}), frozenset({4, 5}), frozenset({6, 7})]
    unstable_islands = [frozenset({8, 9}), frozenset({10, 11})]
    reconnected_edges = {(1, 4)}

    node_costs = {1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 1, 7: 2, 8: 3, 9: 4, 10: 5, 11: 1}
    edge_costs = {
        (1, 2): 1, (1, 3): 2, (2, 3): 3, (1, 4): 4, (4, 5): 5,
        (3, 5): 1, (5, 7): 2, (4, 6): 3, (6, 7): 4, (6, 8): 5,
        (7, 9): 1, (8, 9): 2, (8, 10): 3, (9, 11): 4, (10, 11): 5,
    }
    cost_data = CostData(node_costs=node_costs, edge_costs=edge_costs)
    config = AlgorithmConfig(kappa_op=1.0, kappa_rec=0.7)

    return topology, stable_islands, unstable_islands, reconnected_edges, cost_data, config


def test_canon_sorts_endpoints():
    assert canon((3, 1)) == (1, 3)
    assert canon((1, 3)) == (1, 3)


def test_algorithm_config_requires_kappa_op_greater_than_kappa_rec():
    with pytest.raises(AssertionError):
        AlgorithmConfig(kappa_op=0.5, kappa_rec=0.7)


def test_worked_example_terms_and_combined_cost():
    topology, stable_islands, unstable_islands, reconnected_edges, cost_data, config = _worked_example()

    f1 = f1_unstable_nodes(unstable_islands, cost_data)
    f2 = f2_unstable_edges(unstable_islands, topology, cost_data)
    f3 = f3_operation_edges(stable_islands, unstable_islands, topology, cost_data, config)
    f4 = f4_reconnection(reconnected_edges, cost_data, config)

    assert f1 == 13
    assert f2 == 7
    assert f3 == 23
    assert f4 == pytest.approx(2.8)

    assert F(13, 7, 23, 2.8) == pytest.approx(40.2)
    assert F(f1, f2, f3, f4) == pytest.approx(40.2)
