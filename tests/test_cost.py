"""Tests for islanding/cost.py, reproducing the worked example in
docs/cost_function_spec.md section 6 (11-node, 5-island reconnection case)
and the F5 fixture in section 13.
"""

from dataclasses import replace

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


def test_algorithm_config_requires_k1_at_least_k2():
    with pytest.raises(AssertionError):
        AlgorithmConfig(kappa_op=1.0, kappa_rec=0.7, k_1=0.2, k_2=1.0)

    # k_1 == k_2 is allowed -- the constraint is >=, not strict.
    config = AlgorithmConfig(kappa_op=1.0, kappa_rec=0.7, k_1=0.5, k_2=0.5)
    assert config.k_1 == config.k_2


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


def _f6_fixture():
    # Spec section 16's traced fixture. Same nodes/weights as the F5
    # fixture, plus the reconnected edge (1,4) with w_14 = 5. (1,4) is an
    # original edge that was cut by the islanding and then restored, so it
    # belongs in the topology; being cross-island, _induced_edges excludes
    # it from every E_r, and it reaches F6 only via E^rec.
    #
    # topology carries the full 7-node system so f6_reconnection_stability's
    # node ordering matches how samples were generated
    # (sorted(topology.nodes())), same convention as F5.
    topology = Topology(
        nodes=[1, 2, 3, 4, 5, 6, 7],
        edges=[(1, 2), (1, 3), (2, 3), (4, 5), (6, 7), (1, 4)],
    )

    node_weights = {1: -1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 1}
    edge_weights = {
        (1, 2): 1, (1, 3): 1, (2, 3): 1, (4, 5): 1, (6, 7): 1, (1, 4): 5,
    }
    model = linear_consensus_model(node_weights, edge_weights)

    # k_1 = 1 > k_2 = 0.2, so Part 4 is nonzero (factor -(k_1 - k_2) = -0.8).
    config = AlgorithmConfig(
        kappa_op=1.0, kappa_rec=0.7,
        alpha_1=1.0, beta=1.0, k_1=1.0, P_1=1,
        alpha_2=1.0, gamma=1.0, k_2=0.2, P_2=1,
    )

    stable_islands = [frozenset({1, 2, 3}), frozenset({4, 5}), frozenset({6, 7})]
    reconnected_edges = {(1, 4)}

    return topology, model, config, stable_islands, reconnected_edges


def _f6_traced_samples():
    # Spec section 16's sample state: x_1 = 0.5, all others 0.25.
    # Deliberately non-uniform -- with every node equal, all
    # f_ij = -w_ij(x_i - x_j) vanish and Part 1 would be zero, leaving only
    # the trivial Part 4. W(x) = 0.5*(0.25 + 6*0.0625) = 0.3125 <= gamma = 1.
    samples = np.full((1, 7, 1), 0.25)
    samples[0, 0, 0] = 0.5  # node 1
    return samples


def test_f6_traced_example():
    # Hand-traced ground truth from spec section 16. On linear consensus
    # W_ij = 0, so Parts 2 and 3 vanish identically and this covers Parts 1
    # and 4 only (a model with nonzero W_ij is deferred).
    #
    #   island {1,2,3}: Psi = (0.5 * -1.25) - 0.8*0.1875 = -0.775 -> 0
    #   island {4,5}:   Psi = (0.25 * 1.25) - 0.8*0.0625 = +0.2625 -> 0.2625
    #   island {6,7}:   Psi = Part4 only = -0.8*0.0625 = -0.05 -> 0
    #
    # The two endpoints of (1,4) give opposite-signed Part 1 contributions
    # (-0.625 at node 1, +0.3125 at node 4), so a sign error in the gradient
    # or the dynamics surfaces instead of cancelling.
    #
    # This also pins the hinge ORDER: hinging per island gives 0.2625, while
    # summing Psi_r first would give max(-0.5625, 0) = 0.
    topology, model, config, stable_islands, reconnected_edges = _f6_fixture()
    samples = _f6_traced_samples()

    f6 = f6_reconnection_stability(
        stable_islands, reconnected_edges, topology, model, samples, config
    )

    assert f6 == pytest.approx(0.2625)


def test_f6_zero_state_gives_zero():
    topology, model, config, stable_islands, reconnected_edges = _f6_fixture()
    samples = np.zeros((1, 7, 1))

    f6 = f6_reconnection_stability(
        stable_islands, reconnected_edges, topology, model, samples, config
    )

    assert f6 == 0


def test_f6_no_reconnection_gives_zero_for_any_state():
    # E^rec empty -> Parts 1-3 vanish for every island, leaving
    # Psi_r = -(k_1 - k_2) W_r <= 0 whenever k_1 >= k_2 (guaranteed by
    # AlgorithmConfig). Draw-independent, so this uses real seeded samples
    # rather than a hand-picked state.
    topology, model, config, stable_islands, _ = _f6_fixture()
    config = replace(config, P_2=10)

    np.random.seed(0)
    samples = model.sampler(config.gamma, config.P_2, 7, 1)

    f6 = f6_reconnection_stability(
        stable_islands, set(), topology, model, samples, config
    )

    assert f6 == 0


def test_f6_part_4_vanishes_when_k1_equals_k2():
    # Spec section 16, third additional assertion. With k_1 == k_2 the
    # island-energy correction drops out and only Part 1 survives:
    # island {4,5} contributes [0.3125]_+, islands {1,2,3} and {6,7} give
    # -0.625 and 0, both clipping to zero.
    topology, model, config, stable_islands, reconnected_edges = _f6_fixture()
    config = replace(config, k_2=config.k_1)
    samples = _f6_traced_samples()

    f6 = f6_reconnection_stability(
        stable_islands, reconnected_edges, topology, model, samples, config
    )

    assert f6 == pytest.approx(0.3125)


def test_f6_ignores_cut_edge_that_was_not_reconnected():
    # (3,5) joins islands {1,2,3} and {4,5} in the original graph but is not
    # in E^rec, so it must contribute nothing: _induced_edges leaves it out
    # of both E_r (cross-island), and N_i^rec is read off E^rec alone. Adding
    # it to the topology must therefore leave the traced value untouched.
    # Guards against F6 ever reaching for topology neighbours instead of
    # E^rec -- the successor to the old component-edge-set check, now
    # testing the code rather than a hand-built fixture.
    topology, model, config, stable_islands, reconnected_edges = _f6_fixture()
    topology_with_cut_edge = Topology(
        nodes=[1, 2, 3, 4, 5, 6, 7],
        edges=list(topology.edges()) + [(3, 5)],
    )
    samples = _f6_traced_samples()

    f6 = f6_reconnection_stability(
        stable_islands, reconnected_edges, topology_with_cut_edge, model,
        samples, config,
    )

    assert f6 == pytest.approx(0.2625)


def test_f6_d2_states_match_d1_traced_example():
    # Same fixture and states as test_f6_traced_example, but each x_i is a
    # 2-vector with the traced value in coordinate 0 and zero in coordinate
    # 1. Linear consensus decouples per coordinate and coordinate 1 is
    # identically zero, so every part reduces to the d=1 case and F6 must
    # come out at the same 0.2625. Checks that the dot products reduce to
    # scalars rather than broadcasting.
    topology, model, config, stable_islands, reconnected_edges = _f6_fixture()

    samples = np.zeros((1, 7, 2))
    samples[0, :, 0] = 0.25
    samples[0, 0, 0] = 0.5  # node 1

    f6 = f6_reconnection_stability(
        stable_islands, reconnected_edges, topology, model, samples, config
    )

    assert f6 == pytest.approx(0.2625)


def test_d2_states_match_d1_case_for_f5():
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
    # F6 no longer shares F5's bracket (Psi_r is a different expression), so
    # it no longer reproduces these numbers; its d=2 reduction is covered
    # separately by test_f6_d2_states_match_d1_traced_example.
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


def test_f6_node_indexing_uses_global_not_island_local_order():
    # Regression test for interleaved node labels: stable islands {1,2} and
    # {5,6}, unstable {3,4}, reconnected edge (2,5) with w = 5. Global order
    # sorted(topology.nodes()) is [1,2,3,4,5,6] -> node 5 at index 4, node 6
    # at index 5. A local-index scheme over the stable nodes {1,2,5,6} would
    # put node 5 at index 2 and node 6 at index 3 -- silently reading the
    # UNSTABLE nodes' rows instead.
    #
    # samples is built with the CORRECT global convention: x_1 = 0,
    # x_2 = 0.5, x_5 = x_6 = 0.25. Rows 2 and 3 (nodes 3, 4 -- in no stable
    # island, so read by nothing) hold large poison values.
    #
    # Hand-traced expected value, structurally the section 16 trace with the
    # islands relabelled (w_i = 1 throughout, w_12 = w_56 = 1, w_25 = 5,
    # k_1 = 1, k_2 = 0.2):
    #   island {1,2}: g_2 = 0.5, f_{2,5} = -5(0.5 - 0.25) = -1.25,
    #     Part1 = -0.625; W_r = 0.5*(0 + 0.25) = 0.125,
    #     Part4 = -0.8*0.125 = -0.1  ->  Psi = -0.725 -> 0
    #   island {5,6}: g_5 = 0.25, f_{5,2} = -5(0.25 - 0.5) = +1.25,
    #     Part1 = +0.3125; W_r = 0.0625, Part4 = -0.05  ->  Psi = 0.2625
    #   F6 = 0.2625
    # Under local indexing node 5 would read row 2 (10.0), making island
    # {1,2}'s Part 1 = 0.5 * -5(0.5 - 10) = +23.75 and the total wildly
    # different -- the assertion fails in both directions, not just when a
    # positive penalty appears.
    topology = Topology(
        nodes=[1, 2, 3, 4, 5, 6],
        edges=[(1, 2), (3, 4), (5, 6), (2, 5)],
    )

    node_weights = {1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1}
    edge_weights = {(1, 2): 1, (3, 4): 1, (5, 6): 1, (2, 5): 5}
    model = linear_consensus_model(node_weights, edge_weights)

    config = AlgorithmConfig(
        kappa_op=1.0, kappa_rec=0.7,
        alpha_1=1.0, beta=1.0, k_1=1.0, P_1=1,
        alpha_2=1.0, gamma=1.0, k_2=0.2, P_2=1,
    )

    stable_islands = [frozenset({1, 2}), frozenset({5, 6})]
    reconnected_edges = {(2, 5)}

    samples = np.zeros((1, 6, 1))
    samples[0, 1, 0] = 0.5    # node 2
    samples[0, 2, 0] = 10.0   # node 3's row -- would-be node 5 under the old bug
    samples[0, 3, 0] = -10.0  # node 4's row -- would-be node 6 under the old bug
    samples[0, 4, 0] = 0.25   # node 5
    samples[0, 5, 0] = 0.25   # node 6

    f6 = f6_reconnection_stability(
        stable_islands, reconnected_edges, topology, model, samples, config
    )

    assert f6 == pytest.approx(0.2625)


def test_f6_all_four_parts_with_nonzero_edge_energy(synthetic_edge_energy_model):
    # Hand-traced ground truth from spec section 16b: the section 16 fixture
    # with ONLY the model swapped, so every difference from 0.2625 comes
    # from the W_ij terms.
    #
    #   island {1,2,3}: g_1 = 0.5 + 0.25 + 0.25 = 1.0, and node 1's full
    #     dynamics sum to -1.25, giving parts
    #     -1.25, -0.3125, +0.003125, -0.2  ->  Psi = -1.759375 -> 0
    #   island {4,5}:  x_4 == x_5, so the within-island edge drops out of
    #     the gradients (g_4 = 0.25) and W_45 = 0; parts
    #     +0.3125, -0.25, +0.003125, -0.05  ->  Psi = +0.015625
    #   island {6,7}:  no reconnected neighbours, x_6 == x_7, so Part 4
    #     only  ->  Psi = -0.05 -> 0
    #
    # The surviving island's Psi_r is a small positive assembled from four
    # mixed-sign terms, so dropping or mis-signing any one part shifts the
    # total visibly -- and most single-part errors flip it negative, which
    # collapses F6 to 0. Section 16's linear fixture cannot detect a wrong
    # Part 2 or 3 at all, since both are identically zero there.
    topology, linear_model, config, stable_islands, reconnected_edges = _f6_fixture()
    model = synthetic_edge_energy_model(linear_model)
    samples = _f6_traced_samples()

    f6 = f6_reconnection_stability(
        stable_islands, reconnected_edges, topology, model, samples, config
    )

    assert f6 == pytest.approx(0.015625)
