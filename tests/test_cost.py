"""Tests for islanding/cost.py, reproducing the worked example in
docs/cost_function_spec.md section 6 (11-node, 5-island reconnection case)
and the F5 fixture in section 13.
"""

from dataclasses import replace

import networkx as nx
import numpy as np
import pytest

from islanding.topology import Topology
from islanding.cost import (
    CostData, AlgorithmConfig, canon,
    f1_unstable_nodes, f2_unstable_edges, f3_operation_edges,
    f4_reconnection, F,
    linear_consensus_model, f5_stability_penalty, f6_reconnection_stability,
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
    # islands {4,5} and {6,7} have L_r eigenvalues {1,3}, both >= k_1/2 =
    # 0.5, so bracket = x^T((k_1/2)I - L_r)x <= 0 for ANY state -- this
    # holds for any draw, so use real samples from the model's sampler
    # (fixed seed for reproducibility) rather than a hand-picked state.
    topology, model, config = _f5_fixture()
    config = replace(config, P_1=10)

    np.random.seed(0)
    samples = model.sampler(config.beta, config.P_1, 7, 1)

    stable_islands = [frozenset({4, 5}), frozenset({6, 7})]

    f5 = f5_stability_penalty(stable_islands, topology, model, samples, config)

    assert f5 == 0


def test_f5_zero_state_gives_zero_even_with_faulty_island():
    topology, model, config = _f5_fixture()
    samples = np.zeros((1, 7, 1))
    stable_islands = [frozenset({1, 2, 3}), frozenset({4, 5}), frozenset({6, 7})]

    f5 = f5_stability_penalty(stable_islands, topology, model, samples, config)

    assert f5 == 0


def _f6_topology_model_and_config():
    # Same nodes/weights as the F5 fixture, plus reconnected edge (1,4),
    # w_14 = 5. topology carries the full 7-node system so
    # f6_reconnection_stability's node ordering matches how samples were
    # generated (sorted(topology.nodes())), same convention as F5.
    topology = Topology(
        nodes=[1, 2, 3, 4, 5, 6, 7],
        edges=[(1, 2), (1, 3), (2, 3), (4, 5), (6, 7)],
    )

    node_weights = {1: -1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 1}
    edge_weights = {
        (1, 2): 1, (1, 3): 1, (2, 3): 1, (4, 5): 1, (6, 7): 1, (1, 4): 5,
    }
    model = linear_consensus_model(node_weights, edge_weights)

    config = AlgorithmConfig(
        kappa_op=1.0, kappa_rec=0.7, alpha_2=1.0, gamma=1.0, k_2=0.2, P_2=1,
    )
    return topology, model, config


def _f6_eigenvector_aligned_samples():
    # Same eigenvector-aligned state as the F5 fixture: (2,1,1) normalized,
    # in the kernel of L_{1,2,3} = [[1,-1,-1],[-1,3,-1],[-1,-1,3]]. With
    # k_2=0.2, bracket at this direction for the UNMERGED island {1,2,3} is
    # k_2/2 - 0 = 0.1 > 0 (hand-verified via numpy.linalg.eigvalsh).
    # Nodes 4-7 are left at 0 -- irrelevant, since both the merged component
    # {1,2,3,4,5} (eigenvalues [0.459, 1.866, 3, 4, 11.675], all >= k_2/2 =
    # 0.1) and the standalone islands {4,5}, {6,7} (eigenvalues {1,3} each)
    # have bracket <= 0 for ANY state, hand-verified the same way.
    samples = np.zeros((1, 7, 1))
    samples[0, 0, 0] = 0.81649658  # node 1
    samples[0, 1, 0] = 0.40824829  # node 2
    samples[0, 2, 0] = 0.40824829  # node 3
    return samples


def _merged_components():
    # {1,2,3,4,5}: islands {1,2,3} and {4,5} merged via reconnected edge
    # (1,4). Edge set is (1,2),(2,3),(1,3),(4,5),(1,4) -- NOT (3,5), which
    # is a cut edge whose endpoints land in the same component but was
    # never reconnected.
    merged = nx.Graph()
    merged.add_edges_from([(1, 2), (2, 3), (1, 3), (4, 5), (1, 4)])

    # {6,7}: stands alone.
    standalone = nx.Graph()
    standalone.add_edges_from([(6, 7)])

    return [merged, standalone]


def _unmerged_components():
    # No reconnection: each stable island is its own component.
    island_123 = nx.Graph()
    island_123.add_edges_from([(1, 2), (2, 3), (1, 3)])

    island_45 = nx.Graph()
    island_45.add_edges_from([(4, 5)])

    island_67 = nx.Graph()
    island_67.add_edges_from([(6, 7)])

    return [island_123, island_45, island_67]


def test_f6_reconnection_rescues_failing_island():
    # Component {1,2,3,4,5} has lambda_min = 0.459 >= k_2/2 = 0.1, and
    # {6,7} has lambda_min = 1 >= 0.1, so bracket <= 0 for ANY state -- this
    # holds for any draw, so use real samples from the model's sampler
    # (fixed seed for reproducibility) rather than a hand-picked state.
    topology, model, config = _f6_topology_model_and_config()
    config = replace(config, P_2=10)

    np.random.seed(0)
    samples = model.sampler(config.gamma, config.P_2, 7, 1)

    components = _merged_components()

    f6 = f6_reconnection_stability(components, topology, model, samples, config)

    assert f6 == 0


def test_f6_hinge_fires_when_reconnection_absent():
    topology, model, config = _f6_topology_model_and_config()
    samples = _f6_eigenvector_aligned_samples()
    components = _unmerged_components()

    f6 = f6_reconnection_stability(components, topology, model, samples, config)

    assert f6 > 0


def test_f6_zero_state_gives_zero():
    topology, model, config = _f6_topology_model_and_config()
    samples = np.zeros((1, 7, 1))
    components = _merged_components()

    f6 = f6_reconnection_stability(components, topology, model, samples, config)

    assert f6 == 0


def test_d2_states_match_d1_case_for_f5_and_f6():
    # 2-node fixture (w_1=-1, w_2=3, w_12=3, k_1=k_2=0.2), same as
    # test_local_search.py / test_counterexample.py, but with d=2 states:
    # node 1 = [1, 0], node 2 = [0, 0]. Since the dynamics decouple per
    # coordinate and only coordinate 0 is nonzero, every term reduces to
    # exactly the d=1 case:
    #   island {1} alone: dyn=[1,0], grad_W_r=[1,0], dot=1, W_r=0.5,
    #     bracket = 1 + 0.2*0.5 = 1.1 -> f5 = 1.1
    #   merged {1,2}: dyn_1=[-2,0], dyn_2=[3,0], grad_W_r_1=[1,0],
    #     grad_W_r_2=[0,0], dot sum = -2, W_r=0.5,
    #     bracket = -2 + 0.2*0.5 = -1.9 < 0 -> f5 = 0
    # k_1 == k_2 == 0.2 here, so the F6 side (components mirroring the
    # same node/edge sets) reproduces the identical numbers.
    topology = Topology(nodes=[1, 2], edges=[(1, 2)])

    node_weights = {1: -1.0, 2: 3.0}
    edge_weights = {(1, 2): 3.0}
    model = linear_consensus_model(node_weights, edge_weights)

    config = AlgorithmConfig(
        kappa_op=1.0, kappa_rec=0.7,
        alpha_1=1.0, beta=1.0, k_1=0.2, P_1=1,
        alpha_2=1.0, gamma=1.0, k_2=0.2, P_2=1,
    )

    samples = np.zeros((1, 2, 2))
    samples[0, 0] = [1.0, 0.0]  # node 1
    samples[0, 1] = [0.0, 0.0]  # node 2

    f5_alone = f5_stability_penalty([frozenset({1})], topology, model, samples, config)
    assert f5_alone == pytest.approx(1.1)

    f5_merged = f5_stability_penalty([frozenset({1, 2})], topology, model, samples, config)
    assert f5_merged == pytest.approx(0.0)

    component_1 = nx.Graph()
    component_1.add_node(1)
    f6_alone = f6_reconnection_stability([component_1], topology, model, samples, config)
    assert f6_alone == pytest.approx(1.1)

    component_12 = nx.Graph()
    component_12.add_edges_from([(1, 2)])
    f6_merged = f6_reconnection_stability([component_12], topology, model, samples, config)
    assert f6_merged == pytest.approx(0.0)


def test_f6_merged_component_excludes_cut_edge_3_5():
    merged_component = _merged_components()[0]
    edge_set = {frozenset(e) for e in merged_component.edges()}
    assert frozenset({3, 5}) not in edge_set


def test_f6_node_indexing_uses_global_not_component_local_order():
    # Regression test for interleaved node labels: stable nodes {1,2,5,6},
    # unstable {3,4}. Global order sorted(topology.nodes()) is
    # [1,2,3,4,5,6] -> node 5 at index 4, node 6 at index 5. The union of
    # component nodes is only {1,2,5,6} -> a (now-removed) local-index
    # scheme would put node 5 at index 2, node 6 at index 3 -- silently
    # reading the unstable nodes' rows instead.
    #
    # samples is built using the CORRECT global convention: nodes 5 and 6
    # (indices 4, 5) are exactly zero, so both components should see a
    # zero state and contribute exactly 0. Indices 2 and 3 (nodes 3, 4 --
    # never touched by any component) are deliberately set to the
    # eigenvector direction that fires component {5,6}'s hinge (hand-
    # verified via numpy.linalg.eigh on L_{5,6} = diag(-1,1) +
    # Laplacian(edge) = [[0,-1],[-1,2]], eigenvalues [-0.414, 2.414];
    # bracket at that eigenvector with k_2=0.2 is 0.514 > 0). If
    # f6_reconnection_stability ever again indexed by component-local
    # order, it would misread node 5 and 6's states from these rows and
    # report F6 > 0 instead of the correct 0.
    topology = Topology(
        nodes=[1, 2, 3, 4, 5, 6],
        edges=[(1, 2), (3, 4), (5, 6)],
    )

    node_weights = {1: 1, 2: 1, 3: 1, 4: 1, 5: -1, 6: 1}
    edge_weights = {(1, 2): 1, (3, 4): 1, (5, 6): 1}
    model = linear_consensus_model(node_weights, edge_weights)

    config = AlgorithmConfig(
        kappa_op=1.0, kappa_rec=0.7, alpha_2=1.0, gamma=1.0, k_2=0.2, P_2=1,
    )

    component_12 = nx.Graph()
    component_12.add_edges_from([(1, 2)])
    component_56 = nx.Graph()
    component_56.add_edges_from([(5, 6)])
    components = [component_12, component_56]

    samples = np.zeros((1, 6, 1))
    samples[0, 2, 0] = -0.92387953  # node 3's row -- would-be node 5 under the old bug
    samples[0, 3, 0] = -0.38268343  # node 4's row -- would-be node 6 under the old bug

    f6 = f6_reconnection_stability(components, topology, model, samples, config)

    assert f6 == 0
