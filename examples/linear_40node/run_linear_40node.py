"""Diagnostic run script for the 40-node linear consensus example.

Not a test -- makes no assertions, because the answer isn't known in
advance, and this case has no spec document to predict it. Mirrors
examples/linear_10node/run_linear_10node.py's report format; see
docs/run_linear_10_node_spec.md for that format's rationale.

How to run
----------
From the repo root, with the venv active:

    python -m examples.linear_40node.run_linear_40node

Use the `-m` form, not `python examples/linear_40node/run_linear_40node.py`
-- that puts the repo root on sys.path so `examples.linear_40node.builder`
resolves.
"""

import time

import numpy as np

from examples.linear_40node.builder import (
    LINEAR_40, build_topology, build_problem, build_M, faulty_nodes, faulty_edges,
)
from examples.linear_40node.config import build_config
from utils.save import save_result
from islanding.outer_loop import solve
from islanding.search import build_reconnected_components, DEFAULT_TREE_SEED
from islanding.counterexample import find_counterexamples_for_partition
from islanding.cost import (
    canon, f1_unstable_nodes, f3_operation_edges,
    f4_reconnection, f5_stability_penalty, f6_reconnection_stability, F,
)

SETUP = LINEAR_40

# Labels result.objective in the report. The two methods' objectives are
# the same function of the same partition, but each is evaluated on the
# sample sets its own run finished with -- so the report names which
# solver produced the number rather than presenting it as a bare "F".
INNER_METHOD = "local search"

# Run parameters, named so the saved result records exactly what ran.
SAMPLE_SEED = 0
# SEPARATE from SAMPLE_SEED, which only drives the sample draw. local_search
# calls initial_spanning_tree() with no seed argument, so the tree start is
# whatever islanding.search.DEFAULT_TREE_SEED is -- imported rather than
# duplicated here, so this cannot silently disagree with what actually runs.
# Every figure this script reports is therefore a SINGLE-START result.
TREE_INIT_SEED = DEFAULT_TREE_SEED
MAX_OUTER_ITERATIONS = 10


def _lambda_min(nodes, edges):
    M = build_M(nodes, edges, SETUP.node_weights, SETUP.edge_weights)
    return float(np.linalg.eigvalsh(M).min())


def _induced_edges(topology, nodes):
    return [(u, v) for u, v in topology.edges() if u in nodes and v in nodes]


def build_run_problem():
    """The Problem this script solves, built from the module constants.

    Extracted from main() so the parity tests in
    tests/test_examples/test_script_parity.py can exercise the REAL
    construction path. A test that rebuilt the Problem from SETUP,
    SAMPLE_SEED and build_config() itself would be comparing two
    reimplementations and would pass even if main() had drifted to a
    hardcoded seed or a different config builder.
    """
    return build_problem(
        SETUP, build_topology(), seed=SAMPLE_SEED, config=build_config()
    )


def main():
    problem = build_run_problem()
    topology = problem.topology

    print(f"Setup: {SETUP.name}")
    print(f"Faulty nodes: {faulty_nodes(SETUP)}")
    print(f"Faulty edges: {faulty_edges(SETUP)}")
    print()

    config = problem.config

    start = time.perf_counter()
    # 10 rather than the 10-/50-node scripts' 20: the 20-node case
    # converged after SEVEN inner solves and six augmentations, so this
    # leaves headroom without letting a non-converging run go on
    # indefinitely. (Seven and six, not "6 passes" -- the two counts differ
    # by one on the convergence path; see islanding/outer_loop.py.)
    result = solve(problem, max_outer_iterations=MAX_OUTER_ITERATIONS)
    elapsed = time.perf_counter() - start

    # The partition comes straight off the result -- the outer loop no
    # longer hands back a tree to decode (an inner MILP would have none).
    stable = result.stable_islands
    unstable = result.unstable_islands
    reconnected = result.reconnected_edges
    components = build_reconnected_components(stable, reconnected, topology)

    print("=== Solve ===")
    # `reason` is the STABILITY ORACLE's verdict on the returned partition,
    # never an optimisation status. "converged" says a partition survived
    # the counterexample search; it says nothing about how well, or how
    # provably, the inner problem was solved on the pass that produced it.
    print(f"reason:                     {result.reason}"
          f"   -- stability oracle's verdict, not an optimisation status")
    # Two counts, not one: they differ by one when the loop converges (the
    # final solve found nothing to add) and are equal when it hits the cap.
    print(f"inner solves:               {result.inner_solve_count}")
    print(f"augmentation rounds:        {result.augmentation_rounds}")
    print(f"total witnesses added:      {result.total_witnesses_added}")
    print(f"  island witnesses (F5):    {result.total_island_witnesses_added}")
    print(f"  component witnesses (F6): {result.total_component_witnesses_added}")
    # The sample sets the objective below was actually computed on. On the
    # cap path these are SMALLER than initial P + total witnesses added:
    # the last pass's witnesses were appended after the final solve.
    print(f"final sample counts:        K_phi = {result.final_K_phi}, "
          f"K_psi = {result.final_K_psi}")
    if result.reason == "max_outer_iterations":
        print(
            "  NOTE: hit the outer cap. The reported objective belongs to "
            "the PRE-augmentation sample sets shown above -- the final "
            "pass's witnesses were added but never solved against."
        )
    print(f"sample seed:                {SAMPLE_SEED}")
    print(f"tree init seed:             {TREE_INIT_SEED}"
          f"   -- SINGLE-START result: local_search was initialised from one\n"
          f"                                tree at this seed, so the figures below are "
          f"not a\n                                best-of-several and a different start "
          f"may do better")
    print(f"runtime (s):                {elapsed:.3f}")
    print()

    print("=== Post-islanding ===")
    print(f"Stable islands ({len(stable)}):")
    island_lambda_min = []
    for island in stable:
        nodes = sorted(island)
        lam = _lambda_min(nodes, _induced_edges(topology, island))
        island_lambda_min.append(lam)
        print(f"  {nodes}  lambda_min = {lam:.4f}")

    print(f"Unstable islands ({len(unstable)}):")
    for island in unstable:
        print(f"  {sorted(island)}")

    within_island_edges = set()
    for island in list(stable) + list(unstable):
        within_island_edges |= {canon(e) for e in _induced_edges(topology, island)}
    all_edges = {canon((u, v)) for u, v in topology.edges()}
    cut_edges = sorted(all_edges - within_island_edges)
    print(f"Cut edges: {cut_edges}")
    print()

    print("=== Reconnection ===")
    print(f"Reconnected edges: {sorted(reconnected)}")
    print(f"Reconnected components ({len(components)}):")
    component_lambda_min = []
    for component in components:
        nodes = sorted(component.nodes())
        lam = _lambda_min(nodes, list(component.edges()))
        component_lambda_min.append(lam)
        print(f"  {nodes}  lambda_min = {lam:.4f}")
    print(f"Unstable islands ({len(unstable)}):")
    for island in unstable:
        print(f"  {sorted(island)}")
    print()

    print("=== Cost ===")
    f1 = f1_unstable_nodes(unstable, problem.cost_data)
    f3 = f3_operation_edges(stable, unstable, topology, problem.cost_data, config)
    f4 = f4_reconnection(reconnected, problem.cost_data, config)
    f5 = f5_stability_penalty(stable, topology, problem.model, problem.samples_1, config)
    f6 = f6_reconnection_stability(
        stable, reconnected, topology, problem.model, problem.samples_2, config
    )
    # `problem` is the un-augmented problem built at the top of main(), so
    # these are the sample sets F5/F6 were just evaluated on. solve() grew
    # its own internal copy; those final sizes are result.final_K_phi/psi.
    initial_K_phi = len(problem.samples_1)
    initial_K_psi = len(problem.samples_2)

    print(f"F1 (unstable nodes):      {f1}")
    print(f"F3 (operation/cut edges): {f3}")
    print(f"F4 (reconnection reward): {f4}")
    print(f"F5 (island stability):    {f5}   [samples_1, K_phi={initial_K_phi}]")
    print(f"F6 (reconnection stab.):  {f6}   [samples_2, K_psi={initial_K_psi}]")
    print()

    # Two totals, because they support different comparisons.
    #
    # The economic cost depends on the partition alone -- no sample set
    # enters it -- so the local search's and the MILP's are like-for-like.
    #
    # The penalised total is a DIAGNOSTIC. Both methods' F5/F6 are
    # recomputed here on the same initial samples, so the two totals ARE
    # comparable as scores on a common evaluation set. What they do not
    # establish is an inner-solver optimality gap, or a like-for-like
    # comparison of what each method actually optimised.
    #
    # Two methods CAN follow different witness trajectories: each pass
    # draws its counterexamples from its own partition, so once the
    # witness ARRAYS they append differ on any pass, their sampled
    # problems diverge from there on. Nothing forces that. Differing
    # partitions are not sufficient on their own -- two different
    # partitions can yield the same witnesses, and then the augmented
    # sample sets stay identical -- and two methods that returned the same
    # partitions throughout would be answers to the same sampled problem.
    #
    # It is not settled by the round counts either. On the recorded
    # 20-node run both methods took 7 solves and 6 augmentations, and
    # their sample counts AGREED for five passes before splitting:
    # 20, 23, 26, 27, 28, 31, 32 (local search) versus
    # 20, 23, 26, 27, 28, 30, 31 (MILP). So equal round counts do not
    # imply equal sample sets, and an early agreement does not imply a
    # lasting one.
    #
    # Nor can the saved pass_history settle it the other way: it records
    # K values and witness COUNTS, not witness identities, so two runs
    # with identical histories may still have appended different arrays.
    # Equal histories are evidence of nothing stronger than equal counts.
    economic = F(f1, 0.0, f3, f4)
    total = F(f1, 0.0, f3, f4, f5, f6)
    print(f"Economic cost (F1 + F3 - F4):    {economic}"
          f"   -- sample-independent, comparable across methods")
    print(f"Total with penalties (+F5 +F6):  {total}"
          f"   -- on the INITIAL sample sets, K_phi={initial_K_phi}, "
          f"K_psi={initial_K_psi}")
    print()

    # The inner solver's own value at the returned partition, on the sample
    # sets as they stood for the FINAL inner solve.
    print(f"{INNER_METHOD} objective: {result.objective}"
          f"   (on the FINAL sample sets: K_phi={result.final_K_phi}, "
          f"K_psi={result.final_K_psi})")
    print(f"Total with penalties above: {total}"
          f"   (on the INITIAL sample sets: K_phi={initial_K_phi}, "
          f"K_psi={initial_K_psi})")
    # Sample sets only ever grow by appending, so equal K's mean the final
    # inner solve ran on the same arrays used above. That is all this
    # condition establishes -- NOT that no augmentation happened: a run
    # capped at one outer iteration can solve, find witnesses, append them,
    # and return an objective computed before that augmentation, leaving
    # the K's equal. result.augmentation_rounds, printed in the Solve
    # section, is what answers the augmentation question.
    if (result.final_K_phi, result.final_K_psi) == (initial_K_phi, initial_K_psi):
        print(
            "  Both values were computed on the same sample sets, so they "
            "are the same quantity and must agree."
        )
    else:
        print(
            "  These two are NOT comparable, and their difference is not the\n"
            "  witnesses' contribution. The objective is a sample average, so\n"
            "  augmentation changes numerator and denominator together --\n"
            "  A/K becomes (A+B)/(K+m), reweighting every original sample. The\n"
            "  augmented objective can even fall despite adding positive\n"
            "  violations."
        )
    print()

    print("=== Stability check ===")
    island_witnesses, component_witnesses = find_counterexamples_for_partition(
        stable, reconnected, problem
    )
    print(f"Final island witnesses:    {len(island_witnesses)}")
    print(f"Final component witnesses: {len(component_witnesses)}")

    k1_half = config.k_1 / 2
    for island in stable:
        nodes = sorted(island)
        lam = _lambda_min(nodes, _induced_edges(topology, island))
        if lam < k1_half:
            print(
                f"  FLAG: island {nodes} lambda_min={lam:.4f} < k_1/2={k1_half:.4f} "
                "-- passes the counterexample check but still contributes to F5"
            )

    k2_half = config.k_2 / 2
    for component in components:
        nodes = sorted(component.nodes())
        lam = _lambda_min(nodes, list(component.edges()))
        if lam < k2_half:
            print(
                f"  FLAG: component {nodes} lambda_min={lam:.4f} < k_2/2={k2_half:.4f} "
                "-- passes the counterexample check (which is still "
                "component-based; revised F6 is per-island)"
            )
    print()

    path = save_result(
        result, problem, example_name="linear_40node",
        method=INNER_METHOD,
        sample_seed=SAMPLE_SEED,
        tree_init_seed=TREE_INIT_SEED,
        max_outer_iterations=MAX_OUTER_ITERATIONS,
        runtime_seconds=elapsed,
        setup_name=SETUP.name,
        faulty_nodes=faulty_nodes(SETUP),
        faulty_edges=faulty_edges(SETUP),
        island_lambda_min=island_lambda_min,
        component_lambda_min=component_lambda_min,
        cost_terms={
            "f1": f1, "f3": f3, "f4": f4, "f5": f5, "f6": f6,
            # Recorded separately, each with its basis. The economic
            # cost is unconditionally sample-independent. The penalised
            # total is comparable only as a score on an identical common
            # sample set -- never as an optimality-gap measurement -- and
            # is uninterpretable without the K it was computed on.
            "economic_cost": economic,
            "total_with_penalties": total,
            "K_phi": initial_K_phi, "K_psi": initial_K_psi,
        },
        final_island_witnesses=len(island_witnesses),
        final_component_witnesses=len(component_witnesses),
    )
    print(f"Saved result to {path}")


if __name__ == "__main__":
    main()
