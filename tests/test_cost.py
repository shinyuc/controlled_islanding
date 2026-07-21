"""Tests for islanding/cost.py, reproducing the worked example in
docs/cost_function_spec.md section 6 (11-node, 5-island reconnection case)
and the F5 fixture in section 13.
"""

import numpy as np
import pytest

from islanding.topology import Topology
from islanding.cost import (
    CostData, AlgorithmConfig, canon,
    f1_unstable_nodes, f2_unstable_edges, f3_operation_edges,
    f4_reconnection, F,
    linear_consensus_model, f5_stability_penalty,
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


def test_F_adds_nonzero_f5():
    assert F(13, 7, 23, 2.8, 5.0) == pytest.approx(45.2)


def _f5_fixture():
    # Nodes 1-7, stable-island structure from the 11-node fixture:
    # islands {1,2,3} (triangle), {4,5}, {6,7}.
    topology = Topology(
        nodes=[1, 2, 3, 4, 5, 6, 7],
        edges=[(1, 2), (1, 3), (2, 3), (4, 5), (6, 7)],
    )

    # w_1 = -1 is the faulty (grounding-sign-flipped) node; all others = 1.
    node_weights = {1: -1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 1}
    edge_weights = {(1, 2): 1, (1, 3): 1, (2, 3): 1, (4, 5): 1, (6, 7): 1}
    model = linear_consensus_model(node_weights, edge_weights)

    config = AlgorithmConfig(
        kappa_op=1.0, kappa_rec=0.7, alpha_1=1.0, beta=1.0, k_1=1.0, P_1=1,
    )

    return topology, model, config


def _eigenvector_aligned_samples():
    # L_{1,2,3} = diag(-1,1,1) + Laplacian(triangle) = [[1,-1,-1],[-1,3,-1],[-1,-1,3]].
    # Hand-verified via numpy.linalg.eigh: eigenvalues [0, 3, 4], with (2,1,1)
    # (normalized) in the kernel (L @ (2,1,1) == 0 exactly). That kernel
    # direction maximizes bracket = x^T((k_1/2)I - L_r)x for unit x; with
    # k_1=1 this gives bracket = 0.5 - 0 = 0.5 > 0, so the hinge for island
    # {1,2,3} fires deterministically -- no reliance on random sampling.
    # Nodes 4-7 are left at 0 (irrelevant: islands {4,5} and {6,7} have L_r
    # eigenvalues {1,3}, both >= k_1/2 = 0.5, so their bracket is <= 0 for
    # ANY state, not just zero).
    samples = np.zeros((1, 7, 1))
    samples[0, 0, 0] = 0.81649658  # node 1
    samples[0, 1, 0] = 0.40824829  # node 2
    samples[0, 2, 0] = 0.40824829  # node 3
    return samples


def test_f5_hinge_fires_on_faulty_island():
    topology, model, config = _f5_fixture()
    samples = _eigenvector_aligned_samples()
    stable_islands = [frozenset({1, 2, 3}), frozenset({4, 5}), frozenset({6, 7})]

    f5 = f5_stability_penalty(stable_islands, topology, model, samples, config)

    assert f5 > 0


def test_f5_no_false_positive_when_faulty_island_excluded():
    topology, model, config = _f5_fixture()
    samples = _eigenvector_aligned_samples()
    stable_islands = [frozenset({4, 5}), frozenset({6, 7})]

    f5 = f5_stability_penalty(stable_islands, topology, model, samples, config)

    assert f5 == 0


def test_f5_zero_state_gives_zero_even_with_faulty_island():
    topology, model, config = _f5_fixture()
    samples = np.zeros((1, 7, 1))
    stable_islands = [frozenset({1, 2, 3}), frozenset({4, 5}), frozenset({6, 7})]

    f5 = f5_stability_penalty(stable_islands, topology, model, samples, config)

    assert f5 == 0
