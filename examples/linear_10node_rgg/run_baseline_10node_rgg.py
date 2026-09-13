"""Diagnostic run script for the MILP baseline on the 10-node RGG case.

The counterpart to run_linear_10node_rgg.py: same problem, same
AlgorithmConfig, same outer loop, same report -- only the inner solver
differs. run_linear_10node_rgg.py runs the local search; this runs the MILP of
docs/milp_baseline_spec.md. Run both and the two reports are directly
comparable line for line, which is the whole point of the baseline.

Mirrors examples/linear_10node/run_baseline_10node.py section for section,
so the two cases' MILP reports are themselves comparable. Only the setup,
the config module, the outer-iteration budget and the example_name differ.

This case's topology is DRAWN by the same random geometric rule as the
20-, 30- and 40-node cases, unlike examples/linear_10node/, which is
hand-specified at the same size. Its INITIAL model -- first pass, at
K_phi = K_psi = 20 -- is 2,445 variables, 84 binaries and 6,206
constraints, and grows as

    n_vars    = 2,045 + 10 * (K_phi + K_psi)
    n_constrs = 5,806 + 10 * (K_phi + K_psi)

with the binary count fixed at 84. See build_milp_config in
examples/linear_10node_rgg/config.py for the full table and the time limit.

EXERCISES BOTH ISLANDING AND RECONNECTION. Shedding node 5 and separating
the faulty edges' endpoints gives two stable islands, {1, 2, 4, 6, 8, 10}
(lambda_min 3.192) and {3, 7, 9} (lambda_min 2.831). Three edges cross
between them: (1,7) and (3,8) are faulty and so cut, and (7,8) is HEALTHY
-- an eligible reconnection edge. Expect a non-empty E^rec and nonzero F4
and F6.

A faulty edge does not need to form a graph CUT to force this: its -20
weight makes any island containing both endpoints unstable, so the
partition separates them however well connected the graph is, and healthy
edges crossing that induced separation become candidates. An earlier
version of this docstring predicted no reconnection by asking only whether
the faulty edges disconnect the graph, which is the wrong question.

Not a test -- makes no assertions.

What the MILP buys, stated as two separate facts so they do not conflict.

On a SINGLE fixed Problem, the EXACT global optimum of that sampled MILP
is <= any value the local search can reach on the same Problem. That is
the mathematical statement, and it requires both solvers to see identical
sample sets.

What Gurobi returns is not that exact optimum. A run it declares OPTIMAL
has an incumbent within the requested MIP gap (and its feasibility and
integrality tolerances) of the proven dual bound. What that gives is the
INTERVAL [best_bound, incumbent], which brackets the exact optimum. With a
nonzero requested gap the incumbent can therefore sit slightly ABOVE a
feasible solution found independently, without contradicting anything: the
bound, not the incumbent, is the side that carries the guarantee. Both
gaps and both interval endpoints are printed in the MILP section below and
recorded in the saved result, so the width of that slack is always
visible.

It does NOT follow that:

- the final economic cost is globally optimal. Each pass's optimum is
  optimal for ITS sampled objective; the outer loop then changes that
  objective, and nothing makes the last pass's answer optimal for the
  underlying problem.
- comparing the two runs' FINAL partitions measures an inner optimality
  gap. By then the two have followed different witness trajectories and
  are answers to different sampled problems.
- the MILP must win on the common-initial-sample diagnostic printed in
  the Cost section below. Neither final partition was selected by
  optimising that initial sample set, so that comparison carries no
  guarantee in either direction.

The economic cost (F1 + F3 - F4) is the figure that compares cleanly
across methods: no sample set enters it.

Two things differ from the local-search script, both consequences of the
inner solver:

- The MILP takes solver mechanics (time limit, gap) that the local search
  has no analogue for, so it is bound with functools.partial into the
  one-argument inner_solver that solve() expects.
- SolveResult.tree is None here. The MILP has no matroid encoding and so
  no spanning tree to decode; every part of this report therefore reads
  result.stable_islands / .unstable_islands / .reconnected_edges directly.
  (run_linear_10node_rgg.py already does the same, so the two stay in step.)

Results are saved under example_name="linear_10node_rgg_milp" so they sit
beside, and never overwrite, the local search's "linear_10node_rgg" files.

How to run
----------
From the repo root, with the venv active:

    python -m examples.linear_10node_rgg.run_baseline_10node_rgg

Use the `-m` form, not
`python examples/linear_10node_rgg/run_baseline_10node_rgg.py` -- that puts the
repo root on sys.path so `examples.linear_10node_rgg.builder` resolves.

This script runs the same SETUP as run_linear_10node_rgg.py (LINEAR_10_RGG).
Comparing the two methods on different setups is meaningless, so change
SETUP in BOTH scripts or neither.
"""

import time
from functools import partial

import numpy as np

from examples.linear_10node_rgg.builder import (
    LINEAR_10_RGG, build_topology, build_problem, build_M, faulty_nodes, faulty_edges,
)
from examples.linear_10node_rgg.config import build_config, build_milp_config
from utils.save import save_result, save_failure_result
from islanding.outer_loop import solve
from islanding.search import build_reconnected_components
from islanding.counterexample import find_counterexamples_for_partition
from islanding.cost import (
    canon, f1_unstable_nodes, f3_operation_edges,
    f4_reconnection, f5_stability_penalty, f6_reconnection_stability, F,
)
from baseline.milp import milp_solver, NoIncumbentError

# Must match run_linear_10node_rgg.py -- see the module docstring.
SETUP = LINEAR_10_RGG

# Labels result.objective in the report. The two methods' objectives are
# the same function of the same partition, but each is evaluated on the
# sample sets its own run finished with -- so the report names which
# solver produced the number rather than presenting it as a bare "F".
INNER_METHOD = "MILP"

# Also must match run_linear_10node_rgg.py: the outer loop's witness-growing
# schedule is part of what is being compared, so both methods get the
# same budget.
MAX_OUTER_ITERATIONS = 20

# Named so the saved result records exactly what ran.
SAMPLE_SEED = 0
# The MILP initialises no spanning tree, so there is no second seed to
# record. None, not 0 -- writing 0 would imply a start that never happened.
TREE_INIT_SEED = None


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

    # Printed before the solve, matching run_linear_10node_rgg.py, so the two
    # reports diff cleanly from their first line.
    print(f"Setup: {SETUP.name}")
    print(f"Faulty nodes: {faulty_nodes(SETUP)}")
    print(f"Faulty edges: {faulty_edges(SETUP)}")
    print()

    config = problem.config

    # solve() calls inner_solver(problem) (plus initial_tree on the first
    # pass, for solvers that take one), so the solver mechanics are bound
    # here rather than threaded through the outer loop -- the outer loop
    # stays agnostic about which inner solver it is driving.
    inner = partial(milp_solver, milp_config=build_milp_config())

    start = time.perf_counter()
    try:
        result = solve(
            problem, max_outer_iterations=MAX_OUTER_ITERATIONS, inner_solver=inner
        )
    except NoIncumbentError as exc:
        # The MILP stopped with no feasible solution -- no partition, so no
        # report below can be produced. Record the termination anyway:
        # "timed out before the first incumbent" and "infeasible" are
        # results, and writing nothing makes them indistinguishable from a
        # run that never happened.
        elapsed = time.perf_counter() - start
        print("=== Solve ===")
        print(f"NO INCUMBENT: {exc}")
        print(f"runtime (s):                {elapsed:.3f}")
        path = save_failure_result(
            "linear_10node_rgg_milp",
            method=INNER_METHOD,
            sample_seed=SAMPLE_SEED,
        tree_init_seed=TREE_INIT_SEED,
            max_outer_iterations=MAX_OUTER_ITERATIONS,
            runtime_seconds=elapsed,
            setup_name=SETUP.name,
            faulty_nodes=faulty_nodes(SETUP),
            faulty_edges=faulty_edges(SETUP),
            problem=problem,
            info=exc.info,
            error=exc,
            # solve() attached which pass failed and the sample sets it was
            # solving against. `problem` above is the ORIGINAL one, so after
            # any augmentation its P_1/P_2 are not what the failed pass saw
            # -- this is what keeps the record self-consistent.
            outer_state=exc.outer_state,
        )
        print(f"Saved no-incumbent result to {path}")
        return
    elapsed = time.perf_counter() - start

    # The partition comes straight off the result. result.tree is None for
    # the MILP -- there is no tree to decode -- so nothing below may reach
    # for one.
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
          f"   -- the MILP initialises no tree")
    print(f"runtime (s):                {elapsed:.3f}")
    print()

    # Diagnostics from the LAST outer pass only: result.info is the final
    # inner solve's, since that is the pass whose partition was returned.
    # solve_time is that one pass, not the whole run -- compare it against
    # the total runtime above to see how much of the wall clock the solver
    # itself accounts for.
    print("=== MILP ===")
    info = result.info
    print(f"status (final pass):        {info.get('status')}")
    print(f"mip_gap attained:           {info.get('attained_mip_gap')}")
    print(f"mip_gap requested:          {info.get('requested_mip_gap')}")
    # Incumbent and dual bound together: the inner optimum lies in
    # [best_bound, objective]. On a TIME_LIMIT run this interval is what
    # says how much was left on the table -- a relative gap near zero is
    # hard to read on its own.
    print(f"incumbent (objective):      {result.objective}")
    print(f"best bound:                 {info.get('best_bound')}")
    print(f"time limit (s):             {info.get('time_limit')}")
    print(f"solve time, final pass (s): {info.get('solve_time')}")
    print(f"n_vars:                     {info.get('n_vars')}")
    print(f"n_constrs:                  {info.get('n_constrs')}")
    if info.get("status") == "TIME_LIMIT":
        print(
            "  FLAG: the final pass hit the time limit, so its partition is "
            "the best incumbent found, NOT the inner optimum -- raise "
            "time_limit in build_milp_config() before comparing costs"
        )
    # The two words mean different things and can disagree. Printed
    # unconditionally, because the reader has just seen both.
    print(f"outer reason:               {result.reason}   -- the STABILITY "
          f"ORACLE's verdict")
    print(f"inner status:               {info.get('status')}   -- the "
          f"OPTIMISER's verdict")
    print(
        "  'converged' means the returned partition has no stability\n"
        "  counterexample. It does NOT mean the inner problem was solved to\n"
        "  optimality: reason='converged' with status='TIME_LIMIT' is a\n"
        "  valid combination, and means an unproven incumbent passed the\n"
        "  oracle."
    )
    # result.info is the FINAL pass only. An earlier pass that hit the time
    # limit shaped its own partition, the witnesses drawn from it, and every
    # pass after it -- so scan the per-pass statuses rather than trusting
    # the last one to represent the run.
    # EARLIER passes only -- the final one is already covered by the FLAG
    # above, and including it here would make "earlier passes also" false
    # on a run where only the last pass timed out.
    stalled = [
        entry["pass"] for entry in result.pass_history[:-1]
        if entry.get("status") == "TIME_LIMIT"
    ]
    if stalled:
        print(
            f"  FLAG: earlier passes also hit the time limit: {stalled}. "
            "Those passes' partitions were unproven incumbents, and every "
            "later pass inherited the witnesses drawn from them."
        )
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
    # Equal round counts do not imply equal sample sets, and an early
    # agreement does not imply a lasting one. (An earlier version of this
    # comment quoted a seven-solve, six-augmentation history with specific
    # K sequences; that was copied from another case and is not this one's
    # -- the local path here converges in two inner solves.)
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
        result, problem, example_name="linear_10node_rgg_milp",
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
