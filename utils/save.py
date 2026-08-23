"""Result-saving helper: writes a JSON summary of a solved Problem.

utils depends on islanding (one-way) -- islanding must never import
utils, to keep the core algorithm free of I/O/reporting concerns.

This module stays free of example-specific knowledge: anything that needs
a case's weights (lambda_min per island, the faulty node/edge lists, the
cost terms) is computed by the run script and passed in as plain data.
save_result only serializes.
"""

import json
import math
import os
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from islanding.cost import canon
from islanding.search import build_reconnected_components

# utils/save.py -> utils/ -> repo root.
REPO_ROOT = Path(__file__).resolve().parent.parent

# The caller passes these flat; the payload nests them into terms plus the
# two totals, each stamped with its basis. F2 is absent: it was dropped from
# the formulation, not muted, so there is nothing to record.
COST_KEYS = (
    "f1", "f3", "f4", "f5", "f6",
    "economic_cost", "total_with_penalties", "K_phi", "K_psi",
)
_TERM_KEYS = ("f1", "f3", "f4", "f5", "f6")


LOCAL_SEARCH = "local search"
MILP = "MILP"

# What the outer loop's `reason` does and does not tell you. Recorded on
# every run because the two words invite exactly one confusion: "converged"
# is a statement about the STABILITY ORACLE (the returned partition has no
# counterexample), not about the inner optimisation. A MILP run can be
# `reason = "converged"` and `status = "TIME_LIMIT"` at the same time --
# a real, expected combination meaning the incumbent that passed the oracle
# was never proved optimal.
_REASON_SCOPE_NOTE = (
    "`reason` describes the OUTER loop only: 'converged' means the returned "
    "partition passed the stability oracle with no counterexample. It says "
    "nothing about inner-solver optimality -- reason='converged' with "
    "status='TIME_LIMIT' is a valid combination."
)


# Local search's exit reasons, and what each one licenses. Only
# "zero_objective" is a proof: local_search asserts F_hat >= 0 on every
# iteration (islanding/local_search.py), so a zero objective attains the
# minimum of a nonnegative function and no better partition exists for
# that sampled problem. The other two are silence, not evidence.
_LOCAL_SEARCH_PROOF = {
    "zero_objective": "objective is nonnegative; zero attained",
    "local_optimum": None,
    "max_iterations": None,
}


def _optimality_scope(method, result):
    """What a proved_optimal flag on this run does and does not cover."""
    if method == MILP:
        # Gurobi's OPTIMAL means optimal to its tolerances (MIPGap,
        # feasibility, integrality), not exact arithmetic optimality. The
        # requested and attained gaps are recorded alongside.
        solver = "sampled MILP, optimal within the configured Gurobi tolerances,"
    else:
        solver = "sampled inner objective at"
    return (
        f"the FINAL inner solve only: the {solver} "
        f"K_phi={result.final_K_phi}, K_psi={result.final_K_psi}. NOT a "
        "claim that the outer loop converged, that the returned partition "
        "has no stability counterexample, or that the underlying "
        "(unsampled) problem was solved. On the cap path "
        "proved_optimal=true coexists validly with "
        'reason="max_outer_iterations".'
    )


KNOWN_METHODS = (LOCAL_SEARCH, MILP)


def _check_method(method):
    """Reject a method string the schema has no definition for.

    Not permissiveness for its own sake -- an unknown string is silently
    MISHANDLED rather than merely unvalidated. `_method_block` dispatches
    on `method == MILP` and treats everything else as local search, so a
    typo like "MILP " (trailing space) would slip past the seed
    consistency check AND be written out with a local-search method block:
    `final_reason` and `final_iterations` read off a MILP info dict that
    has neither, and no status, bound or gap recorded at all.

    A genuinely new solver is welcome, but it needs its own branch in
    `_method_block` and its own row in the seed contract first. Failing
    here is what forces that.
    """
    if method not in KNOWN_METHODS:
        raise ValueError(
            f"unknown method {method!r}; expected one of "
            f"{list(KNOWN_METHODS)}. A new solver needs a method block and "
            "a tree-seed contract defined in utils/save.py before its "
            "results can be recorded -- writing it out as local search "
            "would silently mislabel every diagnostic."
        )


def _check_seed_consistency(method, tree_init_seed):
    """Reject a method/tree-seed combination the schema cannot mean.

    The schema documents tree_init_seed as None for a solver that
    initialises no tree, and as the actual start for one that does.
    Accepting the contradictions silently would put a fabricated fact in
    an artifact that outlives the run: a MILP result stamped with a tree
    seed names a start that never happened, and a local-search result
    stamped None hides the single-start degree of freedom the field
    exists to expose.

    Both known methods are covered; `_check_method` has already rejected
    anything else, so there is no unchecked third case.
    """
    if method == MILP and tree_init_seed is not None:
        raise ValueError(
            f"method={MILP!r} with tree_init_seed={tree_init_seed!r}: the "
            "MILP initialises no spanning tree, so there is no start to "
            "record. Pass None."
        )
    if method == LOCAL_SEARCH and tree_init_seed is None:
        raise ValueError(
            f"method={LOCAL_SEARCH!r} with tree_init_seed=None: local "
            "search is initialised from exactly one tree, and which one "
            "is part of what the result means. Pass the seed it used "
            "(islanding.search.DEFAULT_TREE_SEED unless overridden)."
        )


def _method_block(method, result):
    """Solver identity plus the diagnostics that make a run interpretable.

    Without this a time-limited MILP result and a proven-optimal one
    serialize identically: same partition shape, same cost keys, and the
    status/gap/bound that separate them live only in the terminal.

    Reads `result.info`, which is the FINAL pass's diagnostics. Earlier
    passes are in the top-level pass_history -- for a MILP that matters,
    since a time limit hit on an early pass shaped that pass's partition
    and every pass after it.
    """
    info = result.info
    block = {"name": method, "reason_scope": _REASON_SCOPE_NOTE}

    if method == MILP:
        status = info.get("status")
        block.update({
            "status": status,
            "requested_mip_gap": _plain(info.get("requested_mip_gap")),
            "attained_mip_gap": _plain(info.get("attained_mip_gap")),
            "attained_mip_gap_unavailable_reason":
                info.get("attained_mip_gap_unavailable_reason"),
            # Stays meaningful exactly when the relative gap does not.
            "absolute_gap": _plain(info.get("absolute_gap")),
            "best_bound": {
                "value": _plain(info.get("best_bound")),
                # The bound brackets the FINAL INNER objective, nothing
                # wider: [best_bound, objective] is an interval for that
                # sampled problem's optimum only.
                "scope": _optimality_scope(method, result),
            },
            "time_limit": _plain(info.get("time_limit")),
            "final_pass_solve_time": _plain(info.get("solve_time")),
            "n_vars": _plain(info.get("n_vars")),
            "n_constrs": _plain(info.get("n_constrs")),
            "final_inner_solve": {
                "proved_optimal": status == "OPTIMAL",
                "optimality_scope": _optimality_scope(method, result),
            },
        })
    else:
        # LOCAL_SEARCH: the only remaining possibility, since _check_method
        # rejects anything outside KNOWN_METHODS before we get here.
        reason = info.get("reason")
        # Derived, not hard-coded: local search DOES prove optimality when
        # it exits on a zero objective. The evidence was already in
        # pass_history; this states it as a flag.
        #
        # But derive it from the reason ALONE and a mislabelled record
        # would be stamped "proved optimal" on a nonzero objective. The
        # reason and the objective come from the same InnerResult, so they
        # cannot legitimately disagree -- if they do, something upstream is
        # wrong and recording a proof would launder the inconsistency.
        # EXACTLY zero, and finite. A tolerance here would be looser than
        # the code it is checking: local_search selects this exit with
        # `objective == 0` (islanding/local_search.py), so demanding
        # exactly 0.0 rejects no legitimate result today. The old `> 1e-12`
        # window admitted 5e-13 -- and NaN, since every comparison against
        # NaN is False -- both of which would then be persisted as
        # proved_optimal: true.
        #
        # If tolerance-based termination is ever added, it needs its own
        # reason and an epsilon-optimality claim recording the tolerance.
        # Widening this guard would misreport it as exact global optimality.
        if reason == "zero_objective":
            objective = result.objective
            if not math.isfinite(objective) or objective != 0.0:
                raise ValueError(
                    f'reason="zero_objective" with objective '
                    f"{objective!r}: the optimality proof requires the "
                    "objective to be finite and exactly 0.0. Refusing to "
                    "record a contradictory result."
                )
        proof_basis = _LOCAL_SEARCH_PROOF.get(reason)
        block.update({
            "final_reason": reason,
            "final_iterations": _plain(info.get("iterations")),
            "final_inner_solve": {
                "proved_optimal": proof_basis is not None,
                "proof_basis": proof_basis,
                "optimality_scope": _optimality_scope(method, result),
            },
        })

    return block


def save_result(
    result,
    problem,
    example_name,
    *,
    method,
    sample_seed,
    tree_init_seed,
    max_outer_iterations,
    runtime_seconds,
    setup_name,
    faulty_nodes,
    faulty_edges,
    island_lambda_min,
    component_lambda_min,
    cost_terms,
    final_island_witnesses,
    final_component_witnesses,
    results_dir=None,
):
    """Write a JSON summary to <results_dir>/<example_name>_<timestamp>.json.
    Creates results_dir if missing. Returns the path written.

    Everything the terminal report prints is recorded, so a saved result
    can be interpreted months later without re-running anything:

    - `runtime_seconds`: wall clock for the solve, measured by the caller
      with time.perf_counter(). It cannot be measured here -- save_result
      runs after the solve -- so it must be passed in.
    - `method`: which inner solver produced this, LOCAL_SEARCH or MILP.
      Drives the method block, which carries the status/gap/bound a MILP
      run cannot be interpreted without.
    - `sample_seed`, `tree_init_seed`, `max_outer_iterations`: the run
      parameters not already captured by the AlgorithmConfig, all needed to
      reproduce the run. The two seeds are SEPARATE knobs and were never
      one: `sample_seed` drives `build_problem`'s sample draw, while
      `tree_init_seed` is what `initial_spanning_tree` uses to pick the
      local search's starting tree (`islanding.search.DEFAULT_TREE_SEED`).
      They happen to both be 0 today, which is exactly why recording one
      number would look sufficient and quietly hide a whole degree of
      freedom -- every local-search result is a SINGLE-START result at its
      recorded `tree_init_seed`. Pass `tree_init_seed=None` for a solver
      that initialises no tree (the MILP).
    - `setup_name`, `faulty_nodes`, `faulty_edges`: which instance this is.
    - `island_lambda_min` / `component_lambda_min`: sequences of floats
      aligned by INDEX with result.stable_islands and with the components
      rebuilt below, in that order. Computed by the caller via build_M,
      since only the caller knows the case's weights.
    - `cost_terms`: the five-term breakdown (F1, F3, F4, F5, F6 -- there
      is no F2; it was dropped from the formulation) plus BOTH totals,
      keyed as COST_KEYS. The two totals are recorded separately and
      stamped with their basis, because they support different
      comparisons. `economic_cost` (F1 + F3 - F4) involves no sample set
      at all, so it is unconditionally comparable across methods.
      `total_with_penalties` adds F5/F6 and is comparable only as a score
      on an identical common sample set -- hence the `K_phi`/`K_psi`
      stamped alongside it, without which the number cannot be
      interpreted at all. Neither total measures an optimality gap. Those
      K's are the INITIAL sample-set sizes; the solver's own `objective`
      carries its own, larger K's wherever the run augmented.
    - the loop's two work counts (`inner_solve_count`,
      `augmentation_rounds`) and the sample-set sizes the objective was
      computed on (`final_K_phi`, `final_K_psi`), all read off the
      SolveResult -- see islanding/outer_loop.py for why one iteration
      count could not stand for both.
    - the full AlgorithmConfig, without which the numbers above cannot be
      interpreted (alpha and P in particular change what F5/F6 mean).

    results_dir defaults to <repo_root>/results, anchored to this file's
    location rather than the current working directory, so results always
    land inside the project regardless of where the script is launched
    from. Pass an explicit results_dir to override.
    """
    _check_method(method)
    _check_seed_consistency(method, tree_init_seed)

    missing = [key for key in COST_KEYS if key not in cost_terms]
    if missing:
        raise ValueError(f"cost_terms is missing {missing}; expected {list(COST_KEYS)}")

    if results_dir is None:
        results_dir = REPO_ROOT / "results"
    os.makedirs(results_dir, exist_ok=True)

    topology = problem.topology

    # Read the partition off the result rather than decoding a tree: the
    # inner solver may not have produced one (a MILP does not).
    stable = result.stable_islands
    unstable = result.unstable_islands
    reconnected = result.reconnected_edges
    components = build_reconnected_components(stable, reconnected, topology)

    # Guard the index alignment: a mis-wired call site would otherwise
    # attach the wrong lambda_min to an island, silently.
    if len(island_lambda_min) != len(stable):
        raise ValueError(
            f"island_lambda_min has {len(island_lambda_min)} entries but there "
            f"are {len(stable)} stable islands"
        )
    if len(component_lambda_min) != len(components):
        raise ValueError(
            f"component_lambda_min has {len(component_lambda_min)} entries but "
            f"there are {len(components)} reconnected components"
        )

    # Original edges not inside any (stable or unstable) island -- the cut
    # set. Do NOT serialize the raw tree: it's over augmented nodes
    # (supernodes/relays), not the original graph.
    within_island_edges = set()
    for island in list(stable) + list(unstable):
        within_island_edges |= {
            canon((u, v)) for u, v in topology.edges()
            if u in island and v in island
        }
    all_edges = {canon((u, v)) for u, v in topology.edges()}
    cut_edges = sorted(all_edges - within_island_edges)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{example_name}_{timestamp}.json"
    path = os.path.join(results_dir, filename)

    payload = {
        "example_name": example_name,
        "timestamp": timestamp,
        # Discriminates this schema from save_failure_result's, which has
        # no partition, no cost terms and no lambda_min lists.
        "outcome": "solved",
        "runtime_seconds": float(runtime_seconds),
        "method": _method_block(method, result),
        "sample_seed": _plain(sample_seed),
        # None for a solver with no tree to initialise. Recorded separately
        # from sample_seed because they are independent knobs.
        "tree_init_seed": _plain(tree_init_seed),
        "max_outer_iterations": _plain(max_outer_iterations),
        "setup": {
            "name": setup_name,
            "faulty_nodes": [int(node) for node in faulty_nodes],
            "faulty_edges": [list(edge) for edge in faulty_edges],
        },
        "problem_size": {
            "n_nodes": len(list(topology.nodes())),
            "n_edges": len(list(topology.edges())),
        },
        "config": {key: _plain(value) for key, value in asdict(problem.config).items()},
        "reason": result.reason,
        # Stamped like the two totals below, so its sample basis cannot be
        # misread: this is the inner solver's own value, on the sample sets
        # the FINAL inner solve ran on -- not the initial ones carried by
        # cost.total_with_penalties.
        "objective": {
            "value": float(result.objective),
            "sample_dependent": True,
            "K_phi": result.final_K_phi,
            "K_psi": result.final_K_psi,
        },
        "inner_solve_count": result.inner_solve_count,
        "augmentation_rounds": result.augmentation_rounds,
        "total_witnesses_added": result.total_witnesses_added,
        "total_island_witnesses_added": result.total_island_witnesses_added,
        "total_component_witnesses_added": result.total_component_witnesses_added,
        # Every inner solve, in order -- not just the final one. An early
        # pass that hit a MILP time limit is visible here and nowhere else,
        # and it shaped that pass's partition and the whole trajectory
        # after it.
        "pass_history": [
            {key: _plain(value) for key, value in entry.items()}
            for entry in result.pass_history
        ],
        # Sample-set sizes the recorded `objective` was computed on. Do not
        # reconstruct these as initial P + total_witnesses_added: on the cap
        # path the last pass's witnesses were added after the final solve.
        "final_K_phi": result.final_K_phi,
        "final_K_psi": result.final_K_psi,
        "cost": {
            "terms": {key: _plain(cost_terms[key]) for key in _TERM_KEYS},
            "economic_cost": {
                "value": _plain(cost_terms["economic_cost"]),
                "sample_dependent": False,
            },
            "total_with_penalties": {
                "value": _plain(cost_terms["total_with_penalties"]),
                "sample_dependent": True,
                "K_phi": _plain(cost_terms["K_phi"]),
                "K_psi": _plain(cost_terms["K_psi"]),
            },
        },
        "stable_islands": [
            {"nodes": sorted(island), "lambda_min": float(lam)}
            for island, lam in zip(stable, island_lambda_min)
        ],
        "unstable_islands": [sorted(island) for island in unstable],
        "cut_edges": [list(edge) for edge in cut_edges],
        "reconnected_edges": [list(edge) for edge in sorted(reconnected)],
        "reconnected_components": [
            {"nodes": sorted(component.nodes()), "lambda_min": float(lam)}
            for component, lam in zip(components, component_lambda_min)
        ],
        "final_stability_check": {
            "island_witnesses": int(final_island_witnesses),
            "component_witnesses": int(final_component_witnesses),
        },
    }

    with open(path, "w") as f:
        json.dump(payload, f, indent=2)

    return path


def save_failure_result(
    example_name,
    *,
    method,
    sample_seed,
    tree_init_seed,
    max_outer_iterations,
    runtime_seconds,
    setup_name,
    faulty_nodes,
    faulty_edges,
    problem,
    info,
    error,
    outer_state=None,
    results_dir=None,
):
    """Record a run that produced no solution at all.

    The success path needs a partition; a MILP that stops before its first
    incumbent has none, and previously wrote nothing -- leaving a timed-out
    or infeasible run indistinguishable in results/ from a job that was
    never launched. That is the gap this closes, and it matters most on the
    larger cases, where the MILP is the one likely to hit its limit.

    `info` is the diagnostics dict off the InnerSolveFailure, so status,
    model size, requested limits and any available bound survive.
    `incumbent` is null by construction: the run had none.

    `outer_state` is what islanding.outer_loop.solve attached: which pass
    failed, the K's that pass was solving against, the history behind it,
    and the work completed. Pass it. `problem` supplies the topology and
    the INITIAL config, whose P_1/P_2 are the sample counts the run
    STARTED with -- after any augmentation those are not the counts the
    failed solve saw, which is why the at-failure K's are recorded
    separately and the bound is scoped to that pass.

    Deliberately NOT a mode of save_result. Every field save_result
    promises (the partition, the five live cost terms, the lambda_min lists) is
    absent here, and a schema where half the keys may be null is worse
    than two schemas that each mean one thing. `outcome` tells them
    apart -- "solved" there, "no_incumbent" here.
    """
    _check_method(method)
    _check_seed_consistency(method, tree_init_seed)

    if results_dir is None:
        results_dir = REPO_ROOT / "results"
    os.makedirs(results_dir, exist_ok=True)

    topology = problem.topology
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(results_dir, f"{example_name}_{timestamp}_no_incumbent.json")

    state = dict(outer_state or {})
    failed_pass = state.get("failed_pass")
    K_phi, K_psi = state.get("K_phi"), state.get("K_psi")
    if K_phi is None or K_psi is None:
        # No outer state (the solver was called directly, outside the
        # loop), so the failure can only have been against what was passed.
        K_phi, K_psi = len(problem.samples_1), len(problem.samples_2)
    bound_scope = (
        f"the sampled MILP of the FAILED pass only (pass {failed_pass}, "
        f"K_phi={K_phi}, K_psi={K_psi}). Not a bound on the initial "
        "problem, on any other pass, or on the run as a whole."
    )

    payload = {
        "example_name": example_name,
        "timestamp": timestamp,
        "outcome": "no_incumbent",
        "error": str(error),
        "runtime_seconds": float(runtime_seconds),
        "method": {
            "name": method,
            "status": info.get("status"),
            "requested_mip_gap": _plain(info.get("requested_mip_gap")),
            "attained_mip_gap": _plain(info.get("attained_mip_gap")),
            "attained_mip_gap_unavailable_reason":
                info.get("attained_mip_gap_unavailable_reason"),
            "absolute_gap": _plain(info.get("absolute_gap")),
            "best_bound": {
                # May be null: an INFEASIBLE model has no bound to report.
                "value": _plain(info.get("best_bound")),
                "scope": bound_scope,
            },
            "time_limit": _plain(info.get("time_limit")),
            "solve_time": _plain(info.get("solve_time")),
            "n_vars": _plain(info.get("n_vars")),
            "n_constrs": _plain(info.get("n_constrs")),
        },
        # No solution was found, so there is nothing to be optimal about.
        "incumbent": None,
        "objective": None,
        # The loop state around the failure. Without this the record is
        # internally contradictory after any augmentation: `config` below
        # carries the INITIAL P_1/P_2 while the status, model size and
        # bound above came from a larger augmented problem.
        "solve_state_at_failure": {
            "failed_pass": _plain(failed_pass),
            "K_phi": _plain(K_phi),
            "K_psi": _plain(K_psi),
            "completed_inner_solves": _plain(state.get("inner_solve_count")),
            "augmentation_rounds": _plain(state.get("augmentation_rounds")),
            "total_witnesses_added": _plain(state.get("total_witnesses_added")),
            "total_island_witnesses_added":
                _plain(state.get("total_island_witnesses_added")),
            "total_component_witnesses_added":
                _plain(state.get("total_component_witnesses_added")),
            "config_P_note": (
                "`config.P_1`/`config.P_2` below are the INITIAL sample "
                "counts. The failed pass solved against K_phi/K_psi here."
            ),
        },
        "pass_history": [
            {key: _plain(value) for key, value in entry.items()}
            for entry in state.get("pass_history", ())
        ],
        "sample_seed": _plain(sample_seed),
        # None for a solver with no tree to initialise. Recorded separately
        # from sample_seed because they are independent knobs.
        "tree_init_seed": _plain(tree_init_seed),
        "max_outer_iterations": _plain(max_outer_iterations),
        "setup": {
            "name": setup_name,
            "faulty_nodes": [int(node) for node in faulty_nodes],
            "faulty_edges": [list(edge) for edge in faulty_edges],
        },
        "problem_size": {
            "n_nodes": len(list(topology.nodes())),
            "n_edges": len(list(topology.edges())),
        },
        "config": {key: _plain(value) for key, value in asdict(problem.config).items()},
    }

    with open(path, "w") as f:
        json.dump(payload, f, indent=2)

    return path


def _plain(value):
    """Coerce numpy scalars to built-ins so json.dump accepts them.

    np.float64 subclasses float and would serialize anyway, but np.int64
    and np.bool_ do not subclass their Python counterparts and raise. Every
    numpy scalar exposes .item(); no built-in scalar does.
    """
    return value.item() if hasattr(value, "item") else value
