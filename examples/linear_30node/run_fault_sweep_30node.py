"""Fault-position sweep for the 30-node case.

WHAT THIS SHOWS, and what it does not. The method's behaviour should not
depend on WHERE the faults happen to sit. This sweep holds the topology
and the fault COUNTS fixed -- 2 faulty nodes and 4 faulty edges, matching
the committed case -- and varies only the positions, over 10 randomly
drawn configurations. It is a sensitivity check on the instance, not a
comparison against the MILP: no MILP is run here and no optimality claim
is made about any configuration.

POSITIONS ARE DRAWN, NOT CHOSEN. A hand-picked set of alternative fault
placements would be open to exactly the objection this sweep exists to
answer -- that the result depends on a favourable arrangement. They come
from a seeded RNG (SWEEP_SEED below), so the sample is reproducible
without being selected.

DEGENERATE DRAWS ARE REJECTED, AND COUNTED. A draw whose faulty nodes,
once shed, leave the remaining graph disconnected is rejected and
redrawn. The committed case was built around a connected remainder, and a
draw that strands a leaf or a small stub is a different KIND of instance:
its island count is forced by the topology rather than chosen by the
search, so its runtime and island structure are not comparable with the
rest. The rejection count is reported and saved, so the sample is
filtered visibly rather than silently.

Note the check is on NODE removal only, which is what makes a remainder
disconnected in the sense above. Faulty EDGES are not part of it: cutting
them is what the method is supposed to decide about, and rejecting draws
whose faulty edges happen to disconnect the graph would be filtering on
the answer.

Faulty edges incident to a faulty node are NOT excluded. The weight rule
is well defined either way -- node faults are applied first, off the
healthy couplings, exactly as the generator does -- and excluding them
would bias the draw. Such a configuration simply gives that node an even
more negative diagonal.

How to run
----------
From the repo root, with the venv active:

    python -m examples.linear_30node.run_fault_sweep_30node

Roughly 430 s per configuration on the reference machine, so about 1.2
hours for all ten. Progress is printed per configuration as it goes.

Writes one full result per configuration through the ordinary
save_result, under example_name "linear_30node_sweep_NN", plus one
summary JSON collecting the comparison fields across all ten so a table
can be generated from a single file (see
presentation/table_fault_sweep.py).
"""

import argparse
import json
import statistics
import sys
import time
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import networkx as nx
import numpy as np

from islanding.local_search import local_search
from examples.linear_30node.builder import (
    LINEAR_30, build_topology, build_problem, build_M,
    faulty_nodes as setup_faulty_nodes, faulty_edges as setup_faulty_edges,
)
from examples.linear_30node.config import build_config
from examples.linear_30node.linear_30node_topology import (
    NODES, EDGES, NODE_WEIGHTS, EDGE_WEIGHTS,
)
# THE FAULT RULE IS NOT EXPOSED AS A FUNCTION. In
# generate_linear_30node.py it lives inline inside generate(), lines
# 162-174, closing over that module's FAULTY_NODES/FAULTY_EDGES
# constants -- so it cannot be called with a different fault set. What IS
# importable is imported here, so at least the -20 magnitude and the edge
# canonicalisation cannot drift; the two-line node rule is reproduced
# below and pinned by _check_fault_rule_matches_the_artefact() against
# the committed ARTEFACT. Read that function's docstring for what the pin
# does and does not establish -- in particular it cannot see a change to
# the generator that is not followed by regenerating the artefact.
from examples.linear_30node.generate_linear_30node import (
    DEFAULT_SEED as TOPOLOGY_SEED,
    FAULTY_EDGE_WEIGHT,
    FAULTY_NODES as COMMITTED_FAULTY_NODES,
    FAULTY_EDGES as COMMITTED_FAULTY_EDGES,
    _canon,
    _generate_connected,
)
from utils.save import save_result, REPO_ROOT
from islanding.outer_loop import solve
from islanding.search import build_reconnected_components, DEFAULT_TREE_SEED
from islanding.counterexample import find_counterexamples_for_partition
from islanding.cost import (
    f1_unstable_nodes, f3_operation_edges,
    f4_reconnection, f5_stability_penalty, f6_reconnection_stability, F,
)

INNER_METHOD = "local search"

# Identical to run_linear_30node.py, so the only thing that differs
# between this sweep and the committed case is where the faults sit.
SAMPLE_SEED = 0
TREE_INIT_SEED = DEFAULT_TREE_SEED
# The DEFAULT cap. Overridable per invocation with
# --max-outer-iterations, because a configuration that exhausts it
# returns without converging and its cost and runtime are then not
# comparable with the rows that did converge -- rerunning that one row
# with more room is the fix, and it must be recorded per row rather than
# assumed uniform. See --only.
MAX_OUTER_ITERATIONS = 20

# Fault COUNTS, held at the committed case's, so the sweep varies
# position and nothing else.
N_FAULTY_NODES = 2
N_FAULTY_EDGES = 4
N_CONFIGURATIONS = 10

# Separate from SAMPLE_SEED, which drives the sample draw and is held
# fixed. This one drives the fault POSITIONS and is the only new source
# of randomness in the sweep.
SWEEP_SEED = 20260906

# A runaway guard, not a tuning knob. If the rejection rate were ever so
# high that ten acceptable draws could not be found, silently looping
# forever would be worse than stopping with the count in hand.
MAX_DRAW_ATTEMPTS = 1000


def healthy_weights():
    """The weights as drawn, BEFORE any fault is applied.

    The committed artefact stores FAULTED weights -- nodes 5 and 16 and
    four edges already carry their fault values -- so starting from it
    would carry the committed faults into every configuration on top of
    the new ones. This replays the generator's own stream instead:
    _generate_connected returns its RNG mid-stream, positioned exactly
    where generate() draws the weights, so the two dictionaries below are
    the same objects generate() starts from.
    """
    _, rng, nodes, _, edges = _generate_connected(TOPOLOGY_SEED)
    node_weights = {node: float(rng.uniform(1.0, 5.0)) for node in nodes}
    edge_weights = {edge: float(rng.uniform(1.0, 10.0)) for edge in edges}
    return nodes, edges, node_weights, edge_weights


def apply_faults(node_weights, edge_weights, neighbors, nodes, edges):
    """The generator's fault rule, applied to an arbitrary fault set.

    Mirrors generate_linear_30node.py lines 162-174 exactly:

        w_i  = -(sum of w_ij over the FULL neighbourhood) - 1
        w_ij = FAULTY_EDGE_WEIGHT

    Node faults first, off the drawn (healthy) couplings, which is what
    makes the rule unambiguous when a faulty edge is incident to a faulty
    node -- the generator fixes the same order for the same reason.

    Returns fresh dictionaries; the inputs are not mutated, so one healthy
    baseline can be reused across every configuration.
    """
    node_weights = dict(node_weights)
    edge_weights = dict(edge_weights)

    for node in nodes:
        full_coupling = sum(
            edge_weights[_canon((node, j))] for j in neighbors[node]
        )
        node_weights[node] = -full_coupling - 1.0

    for edge in edges:
        edge_weights[_canon(edge)] = FAULTY_EDGE_WEIGHT

    return node_weights, edge_weights


def _check_fault_rule_matches_the_artefact(neighbors, healthy_nodes, healthy_edges):
    """Reapplying the rule with the COMMITTED faults must reproduce the
    committed artefact, bit for bit.

    WHAT THIS GUARANTEES, precisely: that the copy of the rule in this
    file still reproduces the committed weights in
    linear_30node_topology.py. That is worth having -- it catches an edit
    to apply_faults here, a change to FAULTY_EDGE_WEIGHT, and a
    regenerated artefact that this copy no longer matches.

    WHAT IT DOES NOT GUARANTEE: that this copy still agrees with
    generate_linear_30node.py. The sweep never calls the generator's
    inline rule, so nothing here executes it; both sides are only ever
    compared against the artefact, and the artefact is a frozen file. Edit
    the generator's rule WITHOUT regenerating the artefact and this check
    passes unchanged while the two rules have silently diverged. Closing
    that gap needs the rule extracted into a function both callers use --
    see the import block above.

    Exact equality, not a tolerance: both sides perform the same
    additions in the same order on the same floats.
    """
    node_weights, edge_weights = apply_faults(
        healthy_nodes, healthy_edges, neighbors,
        COMMITTED_FAULTY_NODES, COMMITTED_FAULTY_EDGES,
    )
    if node_weights != NODE_WEIGHTS or edge_weights != EDGE_WEIGHTS:
        raise RuntimeError(
            "the fault rule reproduced here no longer matches "
            "generate_linear_30node.py: reapplying it with the committed "
            "fault positions did not reproduce linear_30node_topology.py. "
            "The rule lives inline in generate(), lines 162-174, so the two "
            "copies must be kept in step by hand -- reconcile them before "
            "running the sweep."
        )


def draw_configuration(rng, graph):
    """One fault configuration, or None if the draw is degenerate.

    Degenerate means: shedding the faulty nodes disconnects what is left.
    Returned separately from the accept/reject decision so the caller can
    count rejections rather than have them disappear inside a retry.
    """
    nodes = sorted(
        int(n) for n in rng.choice(sorted(graph.nodes()),
                                   size=N_FAULTY_NODES, replace=False)
    )
    edge_list = sorted(_canon(e) for e in graph.edges())
    picked = rng.choice(len(edge_list), size=N_FAULTY_EDGES, replace=False)
    edges = sorted(edge_list[int(i)] for i in picked)

    remainder = graph.copy()
    remainder.remove_nodes_from(nodes)
    if not nx.is_connected(remainder):
        return None, nodes, edges

    return {"faulty_nodes": nodes, "faulty_edges": edges}, nodes, edges


def draw_all_configurations():
    """The full draw, replayed from SWEEP_SEED.

    Always draws all N_CONFIGURATIONS, even when only one is going to be
    run. The RNG stream is sequential: configuration k depends on every
    draw before it, including the rejected ones, so there is no way to
    reach one configuration without replaying the sequence that produced
    it. Replaying is free -- it is a handful of numpy choices -- and it
    is what makes `--only k` provably the same fault positions the full
    sweep gave that index.
    """
    graph = nx.Graph()
    graph.add_nodes_from(NODES)
    graph.add_edges_from(EDGES)

    rng = np.random.default_rng(SWEEP_SEED)
    configurations, rejected, attempts = [], [], 0
    while len(configurations) < N_CONFIGURATIONS:
        attempts += 1
        if attempts > MAX_DRAW_ATTEMPTS:
            raise RuntimeError(
                f"only {len(configurations)} acceptable draws in "
                f"{MAX_DRAW_ATTEMPTS} attempts ({len(rejected)} rejected). "
                "Either the rejection criterion or the fault counts need "
                "revisiting -- not the attempt cap."
            )
        drawn, nodes, edges = draw_configuration(rng, graph)
        if drawn is None:
            rejected.append({"faulty_nodes": nodes,
                             "faulty_edges": [list(e) for e in edges]})
            continue
        configurations.append(drawn)
    return configurations, rejected, attempts


def timed_inner_solver(record):
    """`local_search`, wrapped so each call's wall clock lands in `record`.

    WHAT IS AND IS NOT INSIDE t_k. solve() calls inner_solver(current)
    and then, separately, runs the counterexample search, grows the
    sample sets, appends to pass_history and prints. Wrapping the solver
    itself brackets exactly the inner optimisation and nothing else --
    every one of those outer-loop activities happens outside this
    function and is therefore excluded from t_k by construction, not by
    subtraction.

    That is the reason for instrumenting here rather than inside
    outer_loop.solve: solve() already exposes inner_solver as a
    parameter, so the measurement needs no change to the solver or the
    loop, and cannot drift from what they actually do.

    perf_counter, not process_time: the quantity of interest is elapsed
    wall clock, which is what total_runtime is measured in and what the
    overhead figure is subtracted from. Mixing the two would make
    non_inner_overhead meaningless.
    """
    def wrapped(problem, **kwargs):
        start = time.perf_counter()
        try:
            return local_search(problem, **kwargs)
        finally:
            record.append(time.perf_counter() - start)
    return wrapped


def _lambda_min(setup, nodes, edges):
    M = build_M(nodes, edges, setup.node_weights, setup.edge_weights)
    return float(np.linalg.eigvalsh(M).min())


def _induced_edges(topology, nodes):
    return [(u, v) for u, v in topology.edges() if u in nodes and v in nodes]


def run_one(index, configuration, healthy, neighbors, max_outer_iterations):
    """Solve one configuration and return its summary row.

    BUILDS ITS OWN Topology, and that is not incidental. Topology carries
    a mutable induced-edge cache (islanding/topology.py:19, read at
    islanding/cost.py:243) which every island evaluated during a search
    populates. Sharing one object across configurations would let row 1
    start cold while every later row inherited the entries its
    predecessors left -- and the post-solve cost reporting warms it again
    before the next timed solve. The cached values are correct, since the
    topology is identical in every row, so no solution would change; but
    RUNTIME is what this sweep measures, and it would become a function
    of row order. A fresh object per configuration is what makes the ten
    timings comparable.

    Everything else that caches is already per-configuration, because
    build_problem below constructs a fresh model from the fresh Setup:
    linear_consensus_model's island_quadratic_form cache is a closure
    over that model, and _verified_quadratic_forms and
    _verified_zero_pairwise are WeakKeyDictionaries keyed by the model
    object. The one genuinely shared cache left is cost._zero_arrays,
    which is module level and keyed by (shape, dtype) rather than by
    model -- it holds two small read-only arrays, is filled by a single
    np.zeros_like on first use, and so warms during row 1 and costs
    nothing measurable thereafter. It is called out here rather than
    fixed: emptying it between rows would be measuring an allocation that
    no real run pays.
    """
    topology = build_topology()
    _, _, healthy_nodes, healthy_edges = healthy
    node_weights, edge_weights = apply_faults(
        healthy_nodes, healthy_edges, neighbors,
        configuration["faulty_nodes"], configuration["faulty_edges"],
    )
    setup = replace(
        LINEAR_30, name=f"linear_30_sweep_{index:02d}",
        node_weights=node_weights, edge_weights=edge_weights,
    )
    problem = build_problem(setup, topology, build_config(), seed=SAMPLE_SEED)
    config = problem.config

    inner_solve_times = []
    start = time.perf_counter()
    result = solve(
        problem, max_outer_iterations=max_outer_iterations,
        inner_solver=timed_inner_solver(inner_solve_times),
    )
    elapsed = time.perf_counter() - start

    # INTERNAL CONSISTENCY, checked rather than assumed. The wrapper
    # appends once per call including the call that raises, so the count
    # must match what the loop reports; and the inner solves are a strict
    # subset of the run, so their total cannot exceed it. A violation
    # means the instrumentation is measuring something other than what it
    # claims, which would be worse than not measuring it.
    if len(inner_solve_times) != result.inner_solve_count:
        raise RuntimeError(
            f"timed {len(inner_solve_times)} inner solves but the loop "
            f"reports {result.inner_solve_count}; the wrapper is not "
            "bracketing inner_solver one-to-one"
        )
    total_inner = sum(inner_solve_times)
    overhead = elapsed - total_inner
    if total_inner > elapsed or overhead < 0.0:
        raise RuntimeError(
            f"inner-solve total {total_inner:.6f} s exceeds the run's "
            f"{elapsed:.6f} s; the two are not measured on the same clock"
        )

    stable = result.stable_islands
    unstable = result.unstable_islands
    reconnected = result.reconnected_edges
    components = build_reconnected_components(stable, reconnected, topology)

    f1 = f1_unstable_nodes(unstable, problem.cost_data)
    f3 = f3_operation_edges(stable, unstable, topology, problem.cost_data, config)
    f4 = f4_reconnection(reconnected, problem.cost_data, config)
    f5 = f5_stability_penalty(
        stable, topology, problem.model, problem.samples_1, config
    )
    f6 = f6_reconnection_stability(
        stable, reconnected, topology, problem.model, problem.samples_2, config
    )
    economic = F(f1, 0.0, f3, f4)
    total = F(f1, 0.0, f3, f4, f5, f6)
    initial_K_phi = len(problem.samples_1)
    initial_K_psi = len(problem.samples_2)

    island_lambda_min = [
        _lambda_min(setup, sorted(island), _induced_edges(topology, island))
        for island in stable
    ]
    component_lambda_min = [
        _lambda_min(setup, sorted(c.nodes()), list(c.edges())) for c in components
    ]
    island_witnesses, component_witnesses = find_counterexamples_for_partition(
        stable, reconnected, problem
    )

    # THE INNER CAP, which the outer `reason` does not report. A local
    # search pass that reaches config.max_iterations returns WITHOUT
    # HAVING ESTABLISHED that its candidate is locally optimal -- it
    # stopped before examining another neighbourhood, so the candidate may
    # in fact be a local optimum and may not. What is missing is the
    # certification, not necessarily the property.
    #
    # Either way it spends the full inner budget, and the outer loop can
    # still finish with reason="converged": convergence is the stability
    # oracle's verdict on the returned partition, not a statement about
    # how that partition was found. A row with a truncated pass costs more
    # time and rests on a weaker guarantee, so it has to be visible.
    capped = [
        entry["pass"] for entry in result.pass_history
        if entry.get("reason") == "max_iterations"
    ]
    iterations = [
        entry["iterations"] for entry in result.pass_history
        if "iterations" in entry
    ]

    path = save_result(
        result, problem, example_name=f"linear_30node_sweep_{index:02d}",
        method=INNER_METHOD,
        sample_seed=SAMPLE_SEED,
        tree_init_seed=TREE_INIT_SEED,
        max_outer_iterations=max_outer_iterations,
        runtime_seconds=elapsed,
        setup_name=setup.name,
        faulty_nodes=setup_faulty_nodes(setup),
        faulty_edges=setup_faulty_edges(setup),
        island_lambda_min=island_lambda_min,
        component_lambda_min=component_lambda_min,
        cost_terms={
            "f1": f1, "f3": f3, "f4": f4, "f5": f5, "f6": f6,
            "economic_cost": economic,
            "total_with_penalties": total,
            "K_phi": initial_K_phi, "K_psi": initial_K_psi,
        },
        final_island_witnesses=len(island_witnesses),
        final_component_witnesses=len(component_witnesses),
    )

    return {
        "index": index,
        "faulty_nodes": configuration["faulty_nodes"],
        "faulty_edges": [list(e) for e in configuration["faulty_edges"]],
        "n_stable_islands": len(stable),
        "n_unstable_islands": len(unstable),
        "stable_island_sizes": sorted((len(i) for i in stable), reverse=True),
        "unstable_island_sizes": sorted((len(i) for i in unstable), reverse=True),
        # The partitions themselves, not only their sizes: a sensitivity
        # study is about WHICH islands each fault placement produces, and
        # a size list cannot answer that after the fact.
        "stable_islands": [sorted(island) for island in stable],
        "unstable_islands": [sorted(island) for island in unstable],
        # Eigenvalue-based stability of what the run returned. Recorded
        # per island AND reduced to a single pass/fail, so the aggregate
        # question ("did every stable component pass?") does not have to
        # be re-derived from the list by whoever reads this.
        "island_lambda_min": island_lambda_min,
        "component_lambda_min": component_lambda_min,
        "min_stable_lambda_min": (
            min(island_lambda_min) if island_lambda_min else None
        ),
        "all_stable_islands_positive_definite": all(
            lam > 0.0 for lam in island_lambda_min
        ),
        "all_components_positive_definite": all(
            lam > 0.0 for lam in component_lambda_min
        ),
        "f1": f1, "f3": f3, "f4": f4,
        "economic_cost": economic,
        "f5": f5, "f6": f6,
        # The sample basis F5 and F6 above were evaluated on. Recorded
        # with them, never separately: a penalty term without its K is
        # uninterpretable. See the note in main() on why this is the
        # INITIAL basis rather than the converged one.
        "f5_f6_K_phi": initial_K_phi,
        "f5_f6_K_psi": initial_K_psi,
        "total_with_penalties": total,
        # TIMING. total_runtime brackets solve() alone -- not the
        # problem construction before it, nor the cost reporting after --
        # so `non_inner_overhead` is the outer loop's own work:
        # counterexample searches, sample augmentation, pass bookkeeping.
        "runtime_seconds": elapsed,
        "total_runtime": elapsed,
        "inner_solve_times": inner_solve_times,
        "num_inner_solves": len(inner_solve_times),
        "total_inner_solve_time": total_inner,
        "mean_inner_solve_time": (
            total_inner / len(inner_solve_times) if inner_solve_times else None
        ),
        "median_inner_solve_time": (
            float(np.median(inner_solve_times)) if inner_solve_times else None
        ),
        "max_inner_solve_time": (
            max(inner_solve_times) if inner_solve_times else None
        ),
        "non_inner_overhead": overhead,
        "inner_solve_count": result.inner_solve_count,
        "augmentation_rounds": result.augmentation_rounds,
        "reason": result.reason,
        "converged": result.reason == "converged",
        # PER ROW, not per run. A configuration that exhausted the cap can
        # be rerun with more room without redoing the others, so the cap
        # is a property of the row -- reading it off the run would then be
        # wrong for exactly the row it matters for.
        "max_outer_iterations": max_outer_iterations,
        # Inner-solver diagnostics, per pass. `hit_inner_cap` is about the
        # LOCAL SEARCH reaching config.max_iterations on some pass;
        # `converged` above is about the OUTER loop's stability oracle.
        # The two are independent and can disagree in either direction.
        "hit_inner_cap": bool(capped),
        "capped_passes": capped,
        "max_inner_iterations_observed": max(iterations) if iterations else None,
        "inner_iteration_cap": config.max_iterations,
        # The inner solver's own value, on the sample sets the LAST inner
        # solve ran on. What that solve WAS depends on how the loop
        # exited, so the basis is recorded beside it rather than assumed:
        # on the converged path it is the solve whose partition survived
        # the oracle; on the cap path it is simply the last pass the
        # budget allowed, and nothing about it is converged.
        "objective": result.objective,
        "objective_basis": (
            "the inner solve whose partition passed the stability oracle"
            if result.reason == "converged" else
            "the LAST PERMITTED inner solve; the outer loop hit "
            "max_outer_iterations, so this is not a converged value"
        ),
        "final_K_phi": result.final_K_phi,
        "final_K_psi": result.final_K_psi,
        "final_island_witnesses": len(island_witnesses),
        "final_component_witnesses": len(component_witnesses),
        "n_reconnected_edges": len(reconnected),
        "reconnected_edges": sorted(list(e) for e in reconnected),
        "result_path": str(path),
    }


def _prepare():
    """Healthy baseline, neighbour map, and the fault-rule pin.

    Shared by the full sweep and by --only, so a single-configuration
    rerun goes through exactly the same checks and starts from exactly
    the same healthy weights.
    """
    healthy = healthy_weights()
    _, _, healthy_nodes, healthy_edges = healthy
    neighbors = {node: set() for node in NODES}
    for u, v in EDGES:
        neighbors[u].add(v)
        neighbors[v].add(u)

    _check_fault_rule_matches_the_artefact(neighbors, healthy_nodes, healthy_edges)
    print("Fault rule checked against the committed artefact: OK", flush=True)
    return healthy, neighbors


def run_only(index, max_outer_iterations, merge_into=None):
    """Rerun a SINGLE configuration, by its 0-based index.

    WHY THIS EXISTS. A configuration that exhausts max_outer_iterations
    returns without converging, so its cost and its runtime answer a
    different question from the rows that converged and cannot sit beside
    them unqualified. Raising the cap for that one row fixes it, and
    rerunning the whole sweep to do so would be both wasteful and
    misleading -- the rows that converged strictly below the old cap
    cannot be affected by raising it, so re-solving them would only
    introduce fresh timing noise into numbers that are already correct.

    THE INDEX IS 0-BASED, matching the `index` field in the summary JSON
    and the NN in the linear_30node_sweep_NN result filenames. It is NOT
    the table's row number, which is index + 1. The fault positions are
    printed before the solve starts so a mis-specified index is visible
    immediately rather than an hour later.

    Everything else is held identical to the full sweep: the same draw
    seed (the whole sequence is replayed, so this index is the same
    positions), the same sample seed, the same AlgorithmConfig, and a
    fresh Topology from run_one.
    """
    healthy, neighbors = _prepare()
    configurations, _, _ = draw_all_configurations()
    if not 0 <= index < len(configurations):
        raise SystemExit(
            f"--only {index} is out of range: this sweep has "
            f"{len(configurations)} configurations, indices 0 to "
            f"{len(configurations) - 1} (table rows 1 to "
            f"{len(configurations)})"
        )

    configuration = configurations[index]
    print(f"Rerunning configuration index {index} (table row {index + 1}) "
          f"at max_outer_iterations = {max_outer_iterations}", flush=True)
    print(f"  faulty nodes: {configuration['faulty_nodes']}", flush=True)
    print(f"  faulty edges: "
          f"{[list(e) for e in configuration['faulty_edges']]}", flush=True)
    print(flush=True)

    start = time.perf_counter()
    row = run_one(index, configuration, healthy, neighbors, max_outer_iterations)
    print(f"  reason:          {row['reason']}", flush=True)
    print(f"  converged:       {row['converged']}", flush=True)
    print(f"  inner solves:    {row['inner_solve_count']}", flush=True)
    print(f"  augmentations:   {row['augmentation_rounds']}", flush=True)
    print(f"  runtime (s):     {row['runtime_seconds']:.1f}", flush=True)
    print(f"  economic cost:   {row['economic_cost']}", flush=True)
    print(f"  islands:         {row['n_stable_islands']} stable, "
          f"{row['n_unstable_islands']} unstable", flush=True)
    print(f"  hit inner cap:   {row['hit_inner_cap']}", flush=True)
    print(f"  saved:           {row['result_path']}", flush=True)
    print(f"  wall clock:      {(time.perf_counter() - start) / 60:.1f} min",
          flush=True)

    if merge_into is not None:
        merged = merge_row(merge_into, row)
        print(flush=True)
        print(f"Merged into a new summary: {merged}", flush=True)
    return row


def merge_row(summary_path, row):
    """Write a NEW summary with `row` substituted for its index.

    A new file, not an edit in place: the original run is a record of
    what was actually executed at the time, and overwriting it would
    destroy the only evidence of the capped result. The table reads the
    most recent summary by name, so the updated one is what it picks up.

    Every other configuration's record is copied through UNCHANGED, with
    one addition: rows that predate per-row cap recording get
    max_outer_iterations filled in from the run-level value they actually
    ran under. That is the field the table needs to say which rows shared
    a cap, and it is recording what happened rather than changing it.
    """
    summary = json.loads(Path(summary_path).read_text())
    run_level_cap = summary.get("max_outer_iterations")

    rows = []
    replaced = False
    for existing in summary["configurations"]:
        if existing["index"] == row["index"]:
            rows.append(row)
            replaced = True
            continue
        if "max_outer_iterations" not in existing:
            existing = dict(existing, max_outer_iterations=run_level_cap)
        rows.append(existing)
    if not replaced:
        raise SystemExit(
            f"index {row['index']} is not in {summary_path}; nothing merged"
        )

    caps = sorted({r["max_outer_iterations"] for r in rows})
    summary = dict(summary)
    summary["configurations"] = rows
    summary["timestamp"] = datetime.now().strftime("%Y%m%d_%H%M%S")
    # The run-level field would now be a half-truth. Kept, because it says
    # what the original sweep ran at, but renamed in meaning by the two
    # fields beside it.
    summary["original_max_outer_iterations"] = run_level_cap
    summary["max_outer_iterations_by_row"] = {
        str(r["index"]): r["max_outer_iterations"] for r in rows
    }
    summary["outer_cap_uniform"] = len(caps) == 1
    summary["reruns"] = summary.get("reruns", []) + [{
        "index": row["index"],
        "max_outer_iterations": row["max_outer_iterations"],
        "replaced_reason": "the original run exhausted the outer cap",
        "timestamp": summary["timestamp"],
    }]

    out = (Path(summary_path).parent /
           f"linear_30node_sweep_summary_{summary['timestamp']}.json")
    out.write_text(json.dumps(summary, indent=2))
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description="30-node fault-position sweep")
    parser.add_argument(
        "--only", type=int, default=None, metavar="N",
        help="rerun a single configuration by its 0-BASED index "
             "(table row N+1) instead of the whole sweep",
    )
    parser.add_argument(
        "--max-outer-iterations", type=int, default=MAX_OUTER_ITERATIONS,
        metavar="M", help=f"outer cap (default {MAX_OUTER_ITERATIONS})",
    )
    parser.add_argument(
        "--merge-into", type=Path, default=None, metavar="SUMMARY",
        help="with --only: write a new summary with that row replaced",
    )
    args = parser.parse_args(argv)

    if args.only is not None:
        run_only(args.only, args.max_outer_iterations, args.merge_into)
        return None
    if args.merge_into is not None:
        raise SystemExit("--merge-into only makes sense with --only")

    return run_sweep(args.max_outer_iterations)


def run_sweep(max_outer_iterations):
    healthy, neighbors = _prepare()

    print(f"Topology: N = {len(NODES)}, M = {len(EDGES)}, "
          f"seed {TOPOLOGY_SEED} (fixed)", flush=True)
    print(f"Faults per configuration: {N_FAULTY_NODES} nodes, "
          f"{N_FAULTY_EDGES} edges (fixed counts, varying positions)",
          flush=True)
    print(f"Sweep seed {SWEEP_SEED}; sample seed {SAMPLE_SEED} and "
          f"AlgorithmConfig held identical to the committed case",
          flush=True)
    print(f"Committed case for reference: nodes {COMMITTED_FAULTY_NODES}, "
          f"edges {[list(e) for e in COMMITTED_FAULTY_EDGES]}", flush=True)
    print(flush=True)

    configurations, rejected, attempts = draw_all_configurations()
    print(f"Drew {N_CONFIGURATIONS} acceptable configurations in {attempts} "
          f"attempts; REJECTED {len(rejected)} as degenerate "
          f"(shedding the faulty nodes disconnected the remainder)", flush=True)
    for entry in rejected:
        print(f"  rejected: nodes {entry['faulty_nodes']}, "
              f"edges {entry['faulty_edges']}", flush=True)
    print(flush=True)

    rows = []
    sweep_start = time.perf_counter()
    for index, configuration in enumerate(configurations):
        print(f"[{index + 1}/{N_CONFIGURATIONS}] solving: "
              f"nodes {configuration['faulty_nodes']}, "
              f"edges {[list(e) for e in configuration['faulty_edges']]}",
              flush=True)
        row = run_one(index, configuration, healthy, neighbors,
                      max_outer_iterations)
        rows.append(row)

        done = index + 1
        spent = time.perf_counter() - sweep_start
        remaining = spent / done * (N_CONFIGURATIONS - done)
        print(f"    {row['reason']}  runtime {row['runtime_seconds']:.1f} s  "
              f"islands {row['n_stable_islands']}S/{row['n_unstable_islands']}U  "
              f"economic {row['economic_cost']:.4f}  "
              f"solves {row['inner_solve_count']}  "
              f"augmentations {row['augmentation_rounds']}", flush=True)
        print(f"    saved {row['result_path']}", flush=True)
        print(f"    elapsed {spent / 60:.1f} min, "
              f"~{remaining / 60:.1f} min remaining", flush=True)
        print(flush=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_path = (
        REPO_ROOT / "results" / f"linear_30node_sweep_summary_{timestamp}.json"
    )
    summary = {
        "example_name": "linear_30node_sweep_summary",
        "timestamp": timestamp,
        "purpose": (
            "Sensitivity of the method to fault POSITION on a fixed 30-node "
            "topology. Fault counts, topology, AlgorithmConfig and sample "
            "seed are all held fixed; only the positions vary. Not a "
            "comparison against the MILP -- no MILP was run."
        ),
        "topology_seed": TOPOLOGY_SEED,
        "n_nodes": len(NODES),
        "n_edges": len(EDGES),
        "sweep_seed": SWEEP_SEED,
        "sample_seed": SAMPLE_SEED,
        "tree_init_seed": TREE_INIT_SEED,
        "max_outer_iterations": max_outer_iterations,
        "outer_cap_uniform": True,
        "max_outer_iterations_by_row": {
            str(r["index"]): r["max_outer_iterations"] for r in rows
        },
        "n_faulty_nodes": N_FAULTY_NODES,
        "n_faulty_edges": N_FAULTY_EDGES,
        "n_configurations": N_CONFIGURATIONS,
        "draw_attempts": attempts,
        "n_rejected_degenerate": len(rejected),
        "rejected_draws": rejected,
        "rejection_criterion": (
            "shedding the faulty nodes leaves the remaining graph "
            "disconnected; checked on NODE removal only"
        ),
        "committed_case_for_reference": {
            "faulty_nodes": list(COMMITTED_FAULTY_NODES),
            "faulty_edges": [list(e) for e in COMMITTED_FAULTY_EDGES],
        },
        # Stated in the file, not only in the reading script: F5 and F6
        # here are evaluated on the INITIAL sample sets, which are
        # identical across every row because the sample seed is fixed --
        # so those two columns ARE comparable. `objective` is each row's
        # LAST inner solve, on whatever sample sets that solve ran on, and
        # is not guaranteed to be comparable across rows; on a row that
        # hit max_outer_iterations it is not a converged value at all.
        "f5_f6_basis": (
            "initial sample sets, identical across configurations "
            "(K_phi = K_psi = P_1 = P_2); comparable across rows"
        ),
        "objective_basis": (
            "each row's LAST inner solve, on the sample sets that solve "
            "ran on. Not guaranteed to be comparable across rows: a "
            "difference in K_phi/K_psi proves the sample arrays differ, "
            "but equal K does NOT prove they are identical -- two rows "
            "can append the same NUMBER of witnesses while appending "
            "different ones. Rows whose outer loop hit "
            "max_outer_iterations report their last permitted solve, "
            "which is not a converged value at all; see each row's own "
            "objective_basis."
        ),
        "total_runtime_seconds": time.perf_counter() - sweep_start,
        "configurations": rows,
    }
    summary_path.write_text(json.dumps(summary, indent=2))

    print(f"Rejected {len(rejected)} degenerate draw(s) of {attempts} attempts",
          flush=True)
    print(f"Converged: {sum(1 for r in rows if r['converged'])}/{len(rows)}",
          flush=True)
    runtimes = [r["runtime_seconds"] for r in rows]
    # statistics.median, not sorted(...)[n // 2]. The latter is the upper
    # of the two middle values on an even-length list, not their mean: on
    # the ten runtimes it reported 862.4 s where the median is 699.4 s,
    # the mean of 536.5 and 862.4. Ten configurations is even by design,
    # so this was wrong on every sweep, not on an unlucky one.
    print(f"Runtime per configuration: min {min(runtimes):.1f} s, "
          f"median {statistics.median(runtimes):.1f} s, "
          f"max {max(runtimes):.1f} s", flush=True)
    print(f"Saved summary to {summary_path}", flush=True)
    return summary_path


if __name__ == "__main__":
    sys.exit(0 if main() is not None or True else 1)
