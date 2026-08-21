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
from islanding.inner import InnerResult
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
    # remaining counterexample) -- not the witness totals or the solve/
    # augmentation counts, which this particular fixture cannot exercise
    # (see the stub-driven tests at the end of this module for those). See
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


# ---------------------------------------------------------------------------
# Work counts and the objective's sample-set basis
#
# Driven by a stub inner solver rather than local_search: the counts depend
# on how many passes the loop takes, and a real search's trajectory on this
# fixture converges immediately (see the note above), so it cannot produce a
# multi-pass run at all -- let alone a deterministic one.
# ---------------------------------------------------------------------------

# Split partition: island {1} alone has M = [-1], eigenvalue -1 <= 0, so
# the counterexample search yields the node-1 witness on BOTH sides (one
# island witness, and one component witness since {1} is also its own
# reconnected component). Every pass returning this partition therefore
# adds 1 to samples_1 and 1 to samples_2.
_SPLIT = ([frozenset({1})], [frozenset({2})], set())
# Merged partition: M = [[2,-3],[-3,6]], both eigenvalues > 0, so no
# witness on either side -> the loop converges on the pass that returns it.
_MERGED = ([frozenset({1, 2})], [], set())


def _scripted_solver(script):
    """Inner solver returning `script`'s partitions in order, last one held.

    Its objective is the size of samples_1 it was handed, which makes the
    returned objective's sample-set basis directly assertable: on the cap
    path it must equal final_K_phi, NOT the grown sample set the loop
    finished holding.
    """
    calls = []

    def solver(problem, **kwargs):
        stable, unstable, reconnected = script[min(len(calls), len(script) - 1)]
        calls.append(len(problem.samples_1))
        return InnerResult(
            stable_islands=stable,
            unstable_islands=unstable,
            reconnected_edges=reconnected,
            objective=float(len(problem.samples_1)),
        )

    solver.calls = calls
    return solver


def test_converging_on_the_first_solve_counts_that_solve():
    # The old outer_iterations returned the zero-based loop index here, so
    # this run reported 0 despite having performed one inner solve.
    problem = _fixture()
    solver = _scripted_solver([_MERGED])

    result = solve(problem, max_outer_iterations=10, inner_solver=solver)

    assert result.reason == "converged"
    assert result.inner_solve_count == 1
    assert result.augmentation_rounds == 0
    assert len(solver.calls) == 1          # the count is the real call count
    assert result.total_witnesses_added == 0
    # Nothing was added, so the objective is on the initial sample sets.
    assert result.final_K_phi == len(problem.samples_1)
    assert result.final_K_psi == len(problem.samples_2)


def test_convergence_path_reports_one_more_solve_than_augmentation():
    # Three non-converged passes, then a converged fourth. The old field
    # reported outer_iterations = 3 for exactly this run; it is now
    # inner_solve_count = 4 with augmentation_rounds = 3.
    problem = _fixture()
    solver = _scripted_solver([_SPLIT, _SPLIT, _SPLIT, _MERGED])

    result = solve(problem, max_outer_iterations=10, inner_solver=solver)

    assert result.reason == "converged"
    assert result.inner_solve_count == 4
    assert result.augmentation_rounds == 3
    assert result.inner_solve_count == result.augmentation_rounds + 1
    assert len(solver.calls) == 4

    # One island and one component witness per non-converged pass.
    assert result.total_island_witnesses_added == 3
    assert result.total_component_witnesses_added == 3
    assert result.total_witnesses_added == 6

    # No augmentation follows the converged solve, so here -- and only here
    # -- the reconstruction from the witness totals does agree.
    assert result.final_K_phi == len(problem.samples_1) + 3
    assert result.final_K_psi == len(problem.samples_2) + 3
    assert result.objective == result.final_K_phi


def test_cap_path_counts_are_equal_and_objective_predates_last_augmentation():
    # Never converges, so the loop exhausts the cap. The final pass finds
    # witnesses, appends them, and exits WITHOUT re-solving -- the subtlety
    # final_K_phi/final_K_psi exist to record.
    problem = _fixture()
    solver = _scripted_solver([_SPLIT])

    result = solve(problem, max_outer_iterations=3, inner_solver=solver)

    assert result.reason == "max_outer_iterations"
    assert result.inner_solve_count == 3
    assert result.augmentation_rounds == 3
    assert result.inner_solve_count == result.augmentation_rounds  # equal here
    assert len(solver.calls) == 3

    # All three passes' witnesses are counted, including the last pass's.
    assert result.total_witnesses_added == 6

    # But the final solve ran on the sample sets as they stood BEFORE that
    # last augmentation: initial 1, plus the first two passes' witnesses.
    assert result.final_K_phi == 3
    assert result.final_K_psi == 3
    assert solver.calls == [1, 2, 3]

    # Reconstructing the sizes from the totals overshoots by the last
    # pass's witnesses -- the reason the fields are recorded, not derived.
    reconstructed = len(problem.samples_1) + result.total_island_witnesses_added
    assert reconstructed == 4
    assert result.final_K_phi != reconstructed

    # The objective is the sample count of the problem it was solved on.
    assert result.objective == result.final_K_phi
