"""Tests for islanding/evaluate.py.

Reuses the 11-node, 5-island spanning tree from the decode_reconnection
multi-island merge test (tests/test_search.py) and the cost data/kappas
from the cost spec's worked example (F1=13, F2=7, F3=23, F4=2.8).
"""

import networkx as nx
import numpy as np
import pytest

from islanding.topology import Topology
from islanding.augmented import SOURCE, SINK
from islanding.cost import (
    CostData, AlgorithmConfig, linear_consensus_model,
    f1_unstable_nodes, f2_unstable_edges, f3_operation_edges,
    f4_reconnection, f5_stability_penalty, f6_reconnection_stability, F,
)
from islanding.evaluate import Problem, evaluate
from islanding.search import decode_islanding, decode_reconnection


def _worked_example_topology_and_tree():
    # Same construction as
    # TestDecodeReconnection.test_reconnect_merges_two_of_three_stable_islands:
    # stable islands (1,2,3), (4,5), (6,7); unstable (8,9), (10,11);
    # reconnected edge (1,4).
    full_path_edges = [(1, 2), (2, 3), (4, 5), (6, 7), (8, 9), (10, 11)]
    middle_cut_edges = [(1, 3), (1, 4), (6, 8), (8, 10)]
    not_middle_edges = [(3, 5), (5, 7), (4, 6), (7, 9), (9, 11)]

    topology = Topology(
        nodes=list(range(1, 12)),
        edges=full_path_edges + middle_cut_edges + not_middle_edges,
    )

    tree_edges = [
        (SOURCE, SINK),
        (SOURCE, 1), (SOURCE, 4), (SOURCE, 6),
        (SINK, 8), (SINK, 10),
    ]
    for i, j in full_path_edges:
        relay_i, relay_j = ("n", i, j, i), ("n", i, j, j)
        tree_edges += [(i, relay_i), (relay_i, relay_j), (relay_j, j)]
    for i, j in middle_cut_edges:
        relay_i, relay_j = ("n", i, j, i), ("n", i, j, j)
        tree_edges += [(i, relay_i), (relay_j, j)]
    for i, j in not_middle_edges:
        relay_i, relay_j = ("n", i, j, i), ("n", i, j, j)
        tree_edges += [(i, relay_i), (relay_i, relay_j)]

    tree = nx.Graph()
    tree.add_edges_from(tree_edges)

    return topology, tree


def _worked_example_cost_data():
    node_costs = {1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 1, 7: 2, 8: 3, 9: 4, 10: 5, 11: 1}
    edge_costs = {
        (1, 2): 1, (1, 3): 2, (2, 3): 3, (1, 4): 4, (4, 5): 5,
        (3, 5): 1, (5, 7): 2, (4, 6): 3, (6, 7): 4, (6, 8): 5,
        (7, 9): 1, (8, 9): 2, (8, 10): 3, (9, 11): 4, (10, 11): 5,
    }
    return CostData(node_costs=node_costs, edge_costs=edge_costs)


def test_evaluate_matches_worked_example_with_zero_samples():
    topology, tree = _worked_example_topology_and_tree()
    cost_data = _worked_example_cost_data()
    config = AlgorithmConfig(
        kappa_op=1.0, kappa_rec=0.7,
        alpha_1=1.0, beta=1.0, k_1=1.0, P_1=1,
        alpha_2=1.0, gamma=1.0, k_2=0.2, P_2=1,
    )

    # All-healthy weights: doesn't matter with all-zero samples, F5 and F6
    # are exactly 0 regardless.
    node_weights = {i: 1.0 for i in range(1, 12)}
    edge_weights = {edge: 1.0 for edge in cost_data.edge_costs}
    model = linear_consensus_model(node_weights, edge_weights)

    samples_1 = np.zeros((1, 11, 1))  # F5: indexed over all 11 topology nodes
    samples_2 = np.zeros((1, 11, 1))  # F6: indexed over all 11 topology nodes too

    problem = Problem(
        topology=topology, model=model, cost_data=cost_data, config=config,
        samples_1=samples_1, samples_2=samples_2,
    )

    result = evaluate(tree, problem)

    assert result == pytest.approx(40.2)


def test_evaluate_penalty_terms_raise_total_above_worked_example():
    topology, tree = _worked_example_topology_and_tree()
    cost_data = _worked_example_cost_data()
    config = AlgorithmConfig(
        kappa_op=1.0, kappa_rec=0.7,
        alpha_1=1.0, beta=1.0, k_1=1.0, P_1=1,
        alpha_2=1.0, gamma=1.0, k_2=0.2, P_2=1,
    )

    # w_1 = -1 is the faulty node (as in the F5 worked example); all
    # others = 1.
    node_weights = {i: 1.0 for i in range(1, 12)}
    node_weights[1] = -1.0
    edge_weights = {edge: 1.0 for edge in cost_data.edge_costs}
    model = linear_consensus_model(node_weights, edge_weights)

    # Eigenvector-aligned state for island {1,2,3} (kernel of
    # diag(-1,1,1) + Laplacian(triangle); hand-verified in test_cost.py),
    # which fires F5's hinge; F6's samples stay all-zero.
    samples_1 = np.zeros((1, 11, 1))
    samples_1[0, 0, 0] = 0.81649658  # node 1
    samples_1[0, 1, 0] = 0.40824829  # node 2
    samples_1[0, 2, 0] = 0.40824829  # node 3
    samples_2 = np.zeros((1, 11, 1))

    problem = Problem(
        topology=topology, model=model, cost_data=cost_data, config=config,
        samples_1=samples_1, samples_2=samples_2,
    )

    result = evaluate(tree, problem)

    assert result > 40.2


def test_evaluate_healthy_partition_with_real_samples_gives_only_f1_to_f4():
    # Minimal topology with stable islands {4,5}, {6,7} (both healthy: w=1
    # weights give L_r eigenvalues {1,3}, lambda_min=1 >= k_1/2 = k_2/2 =
    # 0.5) and one unstable island {8,9}; no cross edges at all, so no cut
    # edges and no reconnection. This isolates F5/F6 from F1..F4: with real
    # seeded samples (not hand-picked or all-zero), F5 and F6 must both be
    # exactly 0 for ANY draw, since bracket <= 0 on every stable island
    # regardless of state.
    topology = Topology(
        nodes=[4, 5, 6, 7, 8, 9],
        edges=[(4, 5), (6, 7), (8, 9)],
    )

    r45_4, r45_5 = ("n", 4, 5, 4), ("n", 4, 5, 5)
    r67_6, r67_7 = ("n", 6, 7, 6), ("n", 6, 7, 7)
    r89_8, r89_9 = ("n", 8, 9, 8), ("n", 8, 9, 9)
    tree = nx.Graph()
    tree.add_edges_from([
        (SOURCE, SINK), (SOURCE, 4), (SOURCE, 6), (SINK, 8),
        (4, r45_4), (r45_4, r45_5), (r45_5, 5),
        (6, r67_6), (r67_6, r67_7), (r67_7, 7),
        (8, r89_8), (r89_8, r89_9), (r89_9, 9),
    ])

    node_costs = {4: 4, 5: 5, 6: 1, 7: 2, 8: 3, 9: 4}
    edge_costs = {(4, 5): 5, (6, 7): 4, (8, 9): 2}
    cost_data = CostData(node_costs=node_costs, edge_costs=edge_costs)

    config = AlgorithmConfig(
        kappa_op=1.0, kappa_rec=0.7,
        alpha_1=1.0, beta=1.0, k_1=1.0, P_1=20,
        alpha_2=1.0, gamma=1.0, k_2=1.0, P_2=20,
    )

    node_weights = {4: 1.0, 5: 1.0, 6: 1.0, 7: 1.0, 8: 1.0, 9: 1.0}
    edge_weights = {(4, 5): 1.0, (6, 7): 1.0, (8, 9): 1.0}
    model = linear_consensus_model(node_weights, edge_weights)

    np.random.seed(0)
    samples_1 = model.sampler(config.beta, config.P_1, 6, 1)
    samples_2 = model.sampler(config.gamma, config.P_2, 6, 1)

    problem = Problem(
        topology=topology, model=model, cost_data=cost_data, config=config,
        samples_1=samples_1, samples_2=samples_2,
    )

    result = evaluate(tree, problem)

    # Compute the F1..F4 baseline independently, via the same decode calls
    # evaluate() itself makes, to avoid a hand-derived "magic number".
    stable, unstable = decode_islanding(tree, topology)
    components, _, reconnected = decode_reconnection(tree, topology)
    f1 = f1_unstable_nodes(unstable, cost_data)
    f2 = f2_unstable_edges(unstable, topology, cost_data)
    f3 = f3_operation_edges(stable, unstable, topology, cost_data, config)
    f4 = f4_reconnection(reconnected, cost_data, config)

    # Assert F5 and F6 are exactly 0 directly -- comparing only the total
    # against F(f1, f2, f3, f4) would still pass if f5/f6 were broken and
    # always returned 0, so check the terms themselves too.
    f5 = f5_stability_penalty(stable, topology, model, samples_1, config)
    f6 = f6_reconnection_stability(components, topology, model, samples_2, config)
    assert f5 == 0
    assert f6 == 0

    assert stable == [frozenset({4, 5}), frozenset({6, 7})]
    assert unstable == [frozenset({8, 9})]
    assert reconnected == set()

    assert result == pytest.approx(F(f1, f2, f3, f4))
