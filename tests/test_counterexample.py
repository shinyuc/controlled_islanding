"""Tests for islanding/counterexample.py, reusing the 2-node local-search
fixture from docs/counterexample_spec.md section 7 (w_1=-1, w_2=3, w_12=3).
"""

from dataclasses import replace

import networkx as nx
import numpy as np
import pytest

from islanding.topology import Topology
from islanding.augmented import SOURCE, SINK
from islanding.cost import CostData, AlgorithmConfig, linear_consensus_model
from islanding.evaluate import Problem
from islanding.counterexample import (
    find_counterexamples, find_counterexamples_for_partition,
    linear_counterexample,
)


def _fixture():
    topology = Topology(nodes=[1, 2], edges=[(1, 2)])

    node_weights = {1: -1.0, 2: 3.0}
    edge_weights = {(1, 2): 3.0}
    model = linear_consensus_model(node_weights, edge_weights)

    cost_data = CostData(node_costs={1: 1, 2: 1}, edge_costs={(1, 2): 1})

    config = AlgorithmConfig(
        kappa_op=1.0, kappa_rec=0.7,
        alpha_1=1.0, beta=1.0, k_1=0.2, P_1=1,
        alpha_2=1.0, gamma=1.0, k_2=0.2, P_2=1,
    )

    samples = np.zeros((1, 2, 1))
    problem = Problem(
        topology=topology, model=model, cost_data=cost_data, config=config,
        samples_1=samples, samples_2=samples,
    )
    return problem


def _n(i, j, endpoint):
    return ("n", i, j, endpoint)


def _tree_a0():
    # Islands {1}, {2} separate: edge (1,2) is "cut near i".
    n1, n2 = _n(1, 2, 1), _n(1, 2, 2)
    tree = nx.Graph()
    tree.add_edges_from([
        (SOURCE, SINK), (SOURCE, 1), (SOURCE, 2), (n1, n2), (n2, 2),
    ])
    return tree


def _tree_a1():
    # Merged island {1,2}: edge (1,2) is "through".
    n1, n2 = _n(1, 2, 1), _n(1, 2, 2)
    tree = nx.Graph()
    tree.add_edges_from([
        (SOURCE, SINK), (SOURCE, 1), (1, n1), (n1, n2), (n2, 2),
    ])
    return tree


def test_a0_yields_one_island_witness_for_node_1():
    # Island {1} alone: M = [-1], eigenvalue -1 <= 0 -> one witness. Island
    # {2}: M = [3], eigenvalue 3 > 0 -> no witness. Witness is scaled to
    # ||x|| = sqrt(2 * factor * beta) = sqrt(2) (factor = 1, no inflation),
    # zero-padded at node 2's position. Assert on magnitude, not sign.
    problem = _fixture()
    island_witnesses, _ = find_counterexamples(_tree_a0(), problem)

    assert len(island_witnesses) == 1
    witness = island_witnesses[0]
    assert witness.shape == (2, 1)

    expected_radius = (2.0 * 1.0 * problem.config.beta) ** 0.5
    assert np.linalg.norm(witness) == pytest.approx(expected_radius)
    assert abs(witness[0, 0]) == pytest.approx(expected_radius)
    assert witness[1, 0] == pytest.approx(0.0)


def test_a0_component_witnesses_mirror_island_witnesses():
    # A0 has no reconnection (edge (1,2) is "cut near i", not middle-cut),
    # so components == stable islands unmerged: component {1} alone
    # reproduces the same M = [-1] as island {1} (beta == gamma == 1 in
    # this fixture, so the radius matches too).
    problem = _fixture()
    island_witnesses, component_witnesses = find_counterexamples(_tree_a0(), problem)

    assert len(component_witnesses) == 1
    assert component_witnesses[0].shape == (2, 1)
    assert np.linalg.norm(component_witnesses[0]) == pytest.approx(
        np.linalg.norm(island_witnesses[0])
    )
    assert abs(component_witnesses[0][0, 0]) == pytest.approx(
        abs(island_witnesses[0][0, 0])
    )
    assert component_witnesses[0][1, 0] == pytest.approx(0.0)


def test_a1_yields_no_witnesses():
    # Merged island {1,2}: M = [[2,-3],[-3,6]], eigenvalues ~0.394 and
    # ~7.6, both > 0 -- genuinely stable, no witness for either the island
    # (F5 side) or its (identical, unreconnected) component (F6 side).
    problem = _fixture()
    island_witnesses, component_witnesses = find_counterexamples(_tree_a1(), problem)

    assert island_witnesses == []
    assert component_witnesses == []


def test_linear_counterexample_d1_unit_1_alone():
    # Spec section 7, assertion 3. Unit ({1}, {}): M=[-1], eigenvalue
    # -1<=0 -> one witness. node_order=[1,2], d=1, threshold_radius=sqrt(2).
    find = linear_counterexample({1: -1.0, 2: 3.0}, {(1, 2): 3.0})

    witnesses = find(unit_nodes=[1], unit_edges=[], node_order=[1, 2], d=1,
                     threshold_radius=2.0 ** 0.5, stability_tolerance=0.0)

    assert len(witnesses) == 1
    witness = witnesses[0]
    assert witness.shape == (2, 1)
    assert abs(witness[0, 0]) == pytest.approx(2.0 ** 0.5)
    assert witness[1, 0] == pytest.approx(0.0)


def test_linear_counterexample_d2_unit_1_alone():
    # Spec section 7, assertion 4. Same unit and node_order, d=2 -> still
    # exactly ONE witness (one per non-positive eigenvalue, independent of
    # d), with the node-1 entry in coordinate 0 only and Frobenius norm
    # sqrt(2).
    find = linear_counterexample({1: -1.0, 2: 3.0}, {(1, 2): 3.0})

    witnesses = find(unit_nodes=[1], unit_edges=[], node_order=[1, 2], d=2,
                     threshold_radius=2.0 ** 0.5, stability_tolerance=0.0)

    assert len(witnesses) == 1
    witness = witnesses[0]
    assert witness.shape == (2, 2)

    assert abs(witness[0, 0]) == pytest.approx(2.0 ** 0.5)
    assert witness[0, 1] == pytest.approx(0.0)
    assert witness[1, 0] == pytest.approx(0.0)
    assert witness[1, 1] == pytest.approx(0.0)
    assert np.linalg.norm(witness) == pytest.approx(2.0 ** 0.5)


# ---------------------------------------------------------------------------
# The stability threshold carries a tolerance (spec section 2.1)
# ---------------------------------------------------------------------------

def _marginal_unit_weights(epsilon):
    """A 2-node unit whose M_u has a structural zero, nudged by `epsilon`.

    With no grounding (w_i = 0) and one edge of weight 1,
    M_u = [[1, -1], [-1, 1]] -- a pure graph Laplacian, whose all-ones
    kernel vector gives an EXACT zero eigenvalue by construction. That is
    the shape the tolerance exists for: Laplacian-like M_u matrices have
    structural zeros, so eigh returning one as a small value of either
    sign is the normal case.

    Adding `epsilon` to both diagonal entries shifts the whole spectrum by
    epsilon, so the zero eigenvalue becomes exactly epsilon -- positive but
    numerically indistinguishable from zero.
    """
    return {1: epsilon, 2: epsilon}, {(1, 2): 1.0}


def test_marginal_eigenvalue_nudged_positive_still_yields_a_witness():
    # THE REGRESSION. The old test was `lam > 0: continue`, so an
    # eigenvalue of +1e-15 -- a structural zero that eigh (or the tiniest
    # perturbation in the weights) reported on the wrong side of zero --
    # was classified stable. No witness meant no augmentation, and the
    # outer loop then reported "converged" on a unit that is marginally
    # unstable: a false stability claim, silently.
    node_weights, edge_weights = _marginal_unit_weights(1e-15)
    find = linear_counterexample(node_weights, edge_weights)

    witnesses = find(
        unit_nodes=[1, 2], unit_edges=[(1, 2)], node_order=[1, 2], d=1,
        threshold_radius=2.0 ** 0.5, stability_tolerance=1e-9,
    )

    assert len(witnesses) == 1
    assert witnesses[0].shape == (2, 1)
    # Scaled to the boundary radius like any other witness.
    assert np.linalg.norm(witnesses[0]) == pytest.approx(2.0 ** 0.5)


def test_the_same_unit_is_missed_by_an_exact_zero_comparison():
    # Pins WHY the tolerance is needed rather than trusting the claim: with
    # stability_tolerance = 0 the test reduces to the old `lam > 0` and the
    # marginal unit goes unreported. If eigh ever returned this eigenvalue
    # as exactly 0.0 or negative, this test would fail and the regression
    # test above would be proving nothing.
    node_weights, edge_weights = _marginal_unit_weights(1e-15)
    find = linear_counterexample(node_weights, edge_weights)

    witnesses = find(
        unit_nodes=[1, 2], unit_edges=[(1, 2)], node_order=[1, 2], d=1,
        threshold_radius=2.0 ** 0.5, stability_tolerance=0.0,
    )

    assert witnesses == []


def test_a_comfortably_stable_unit_is_not_dragged_in_by_the_tolerance():
    # The tolerance must not turn every stable unit into a witness. This
    # unit's eigenvalues are 1 and 3, both far above 1e-9.
    find = linear_counterexample({1: 1.0, 2: 1.0}, {(1, 2): 1.0})

    witnesses = find(
        unit_nodes=[1, 2], unit_edges=[(1, 2)], node_order=[1, 2], d=1,
        threshold_radius=2.0 ** 0.5, stability_tolerance=1e-9,
    )

    assert witnesses == []


def test_tolerance_reaches_the_search_from_the_config():
    # End to end: find_counterexamples_for_partition must forward
    # config.stability_tolerance, or the fix never takes effect in the
    # outer loop where the false-convergence risk actually lives.
    node_weights, edge_weights = _marginal_unit_weights(1e-15)
    topology = Topology(nodes=[1, 2], edges=[(1, 2)])
    model = linear_consensus_model(node_weights, edge_weights)
    cost_data = CostData(node_costs={1: 1, 2: 1}, edge_costs={(1, 2): 1})
    base = AlgorithmConfig(
        kappa_op=1.0, kappa_rec=0.7,
        alpha_1=1.0, beta=1.0, k_1=0.2, P_1=1,
        alpha_2=1.0, gamma=1.0, k_2=0.2, P_2=1,
    )
    samples = np.zeros((1, 2, 1))

    def witnesses_with(tolerance):
        problem = Problem(
            topology=topology, model=model, cost_data=cost_data,
            config=replace(base, stability_tolerance=tolerance),
            samples_1=samples, samples_2=samples,
        )
        island, _ = find_counterexamples_for_partition(
            [frozenset({1, 2})], set(), problem
        )
        return island

    # Default-sized tolerance catches it; an exact-zero tolerance does not.
    assert len(witnesses_with(1e-9)) == 1
    assert witnesses_with(0.0) == []
