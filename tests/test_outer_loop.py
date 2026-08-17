"""Tests for islanding/outer_loop.py, reusing the 2-node fixture from
docs/outer_loop_spec.md section 6 (w_1=-1, w_2=3, w_12=3, beta=gamma=1).
"""

import networkx as nx
import numpy as np

from islanding.topology import Topology
from islanding.augmented import SOURCE, SINK
from islanding.cost import CostData, AlgorithmConfig, linear_consensus_model
from islanding.evaluate import Problem
from islanding.counterexample import find_counterexamples_for_partition
from islanding.outer_loop import solve, _augment


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

    # Single sample point x = (1, 0) -- NOT the node-1 witness (sqrt(2), 0):
    # same direction, different magnitude, so it is not literally present
    # in the initial sample sets.
    samples = np.array([[[1.0], [0.0]]])

    problem = Problem(
        topology=topology, model=model, cost_data=cost_data, config=config,
        samples_1=samples, samples_2=samples,
    )
    return problem


def _n(i, j, endpoint):
    return ("n", i, j, endpoint)


def _tree_a_init():
    # Stable island {1}, unstable island {2}: edge (1,2) cut near node 1.
    n1, n2 = _n(1, 2, 1), _n(1, 2, 2)
    tree = nx.Graph()
    tree.add_edges_from([
        (SOURCE, 1), (SOURCE, SINK), (SINK, 2), (n1, n2), (n2, 2),
    ])
    return tree


def test_solve_converges_to_a_genuinely_stable_tree():
    # NOTE: with this exact fixture, local_search's very first swap from
    # A_init reaches the merged {1,2} island directly -- that state is
    # both cost-optimal (F_hat=0, the only edge is now within-island) and
    # sample-independent genuinely stable (M's eigenvalues don't depend on
    # sampling), so the augment path never actually fires here. This test
    # therefore only asserts what's structurally guaranteed: the loop
    # converges, and the returned partition really is genuinely stable (no
    # remaining counterexample) -- not witnesses_added or outer_iterations,
    # which this particular fixture cannot exercise. See
    # test_augment_grows_island_samples_and_leaves_components_unchanged
    # below for a direct, trajectory-independent test of the augment path.
    problem = _fixture()
    max_outer_iterations = 10

    result = solve(problem, max_outer_iterations, initial_tree=_tree_a_init())

    assert result.reason == "converged"

    island_witnesses, component_witnesses = find_counterexamples_for_partition(
        result.stable_islands, result.reconnected_edges, problem
    )
    assert island_witnesses == []
    assert component_witnesses == []


def test_augment_grows_island_samples_and_leaves_components_unchanged():
    problem = _fixture()

    # Hand-built node-1 witness: the (sqrt(2), 0) direction from
    # counterexample.py's worked example, as an (n, d) = (2, 1) array.
    witness = np.array([[2.0 ** 0.5], [0.0]])

    augmented = _augment(problem, [witness], [])

    assert len(augmented.samples_1) == len(problem.samples_1) + 1
    assert augmented.config.P_1 == len(augmented.samples_1)
    np.testing.assert_allclose(augmented.samples_1[-1], witness)
    np.testing.assert_allclose(augmented.samples_1[:-1], problem.samples_1)

    assert len(augmented.samples_2) == len(problem.samples_2)
    assert augmented.config.P_2 == len(problem.samples_2)
    np.testing.assert_allclose(augmented.samples_2, problem.samples_2)


def test_augment_with_no_witnesses_leaves_both_sample_sets_unchanged():
    problem = _fixture()

    augmented = _augment(problem, [], [])

    assert augmented.samples_1 is problem.samples_1
    assert augmented.samples_2 is problem.samples_2
    assert augmented.config.P_1 == problem.config.P_1
    assert augmented.config.P_2 == problem.config.P_2
