"""Constraint-generation outer loop (Algorithm 1, lines 13-19). See
docs/outer_loop_spec.md for the full specification.

The inner solver is a parameter, defaulting to local_search. This layer
only grows the sample sets and re-runs the inner solver; it works with the
partition the solver returns (an InnerResult) and never with a spanning
tree, which is local search's private encoding rather than part of the
inner problem.
"""

from dataclasses import dataclass, field, replace

import numpy as np

from islanding.inner import InnerSolveFailure
from islanding.local_search import local_search
from islanding.counterexample import find_counterexamples_for_partition


@dataclass(frozen=True)
class SolveResult:
    objective: float
    # How much work the loop did, as two separate counts. One number
    # cannot serve both exit paths: the loop does N inner solves and
    # either N-1 augmentations (converged -- the last solve found no
    # counterexample, so nothing followed it) or N (cap reached -- the
    # last solve's witnesses were added, and then the loop ran out of
    # passes before re-solving). Reporting a single "iterations" figure
    # therefore means different things depending on how the loop exited.
    inner_solve_count: int      # times inner_solver was actually called
    augmentation_rounds: int    # times the sample sets were grown
    # Running totals over every outer pass, split by side so it is visible
    # whether the island side (F5) or the component side (F6) is driving
    # the iteration. These are cumulative counts of witnesses ADDED to the
    # sample sets -- not the final check on the returned partition, which
    # is zero on both sides whenever reason == "converged".
    #
    # On the cap path these totals OVERCOUNT what `objective` saw: the
    # final allowed pass finds witnesses and appends them, then the loop
    # exits without re-solving. Use final_K_phi/final_K_psi below for the
    # sample sets the returned objective was actually computed on.
    total_witnesses_added: int
    total_island_witnesses_added: int
    total_component_witnesses_added: int
    # Sizes of the sample sets the RETURNED objective was computed on:
    # len(samples_1) and len(samples_2) of the Problem handed to the final
    # inner solve. Recorded rather than left to be reconstructed, because
    # on the cap path "initial P + total_witnesses_added" overshoots by the
    # last pass's witnesses, which were added but never solved against.
    # K_phi is the F5/Phi side (samples_1), K_psi the F6/Psi side
    # (samples_2); the two grow independently, so they differ in general.
    final_K_phi: int
    final_K_psi: int
    reason: str  # "converged" | "max_outer_iterations"
    stable_islands: list       # list[frozenset] -- the set S
    unstable_islands: list     # list[frozenset] -- the set U
    reconnected_edges: set     # E^rec, canonical (min, max) tuples
    # Optional: only inner solvers that use the matroid encoding have one.
    # A MILP will leave this None, so read the partition fields instead.
    tree: object = None
    # The FINAL pass's diagnostics only. An earlier pass may have exited
    # differently -- a MILP that hit its time limit on pass 0 shaped that
    # pass's partition, its witnesses, and every pass after it -- and none
    # of that is visible here. Read pass_history for the whole run.
    info: dict = field(default_factory=dict)  # the inner solver's diagnostics
    # One entry per inner solve, in order: the pass index, that pass's
    # scalar diagnostics, and how many witnesses it produced. Non-scalar
    # entries in a solver's info (local search puts its tree there) are
    # dropped, so this stays serializable.
    pass_history: tuple = ()


def _augment(problem, island_witnesses, component_witnesses):
    """Grow the sample sets with newly found witnesses (spec section 4).

    P_1/P_2 must track the array lengths exactly -- f5/f6 assert
    equality, so this is the one invariant _augment must preserve.
    """
    samples_1 = problem.samples_1
    if island_witnesses:
        samples_1 = np.concatenate([samples_1, np.stack(island_witnesses)], axis=0)

    samples_2 = problem.samples_2
    if component_witnesses:
        samples_2 = np.concatenate([samples_2, np.stack(component_witnesses)], axis=0)

    config = replace(problem.config, P_1=len(samples_1), P_2=len(samples_2))
    return replace(problem, samples_1=samples_1, samples_2=samples_2, config=config)


def _scalar_info(info):
    """The JSON-safe part of an inner solver's info dict.

    Solvers may park arbitrary objects in `info` -- local_search puts its
    networkx tree there -- so keep only scalars. Dropping silently is the
    right call: the history is a diagnostic record, and a solver adding a
    new non-scalar diagnostic should not break the outer loop.
    """
    return {
        key: value for key, value in info.items()
        if isinstance(value, (int, float, str, bool, type(None)))
    }


def _solve_result(
    inner, solved_on, inner_solve_count, augmentation_rounds,
    island_added, component_added, reason, pass_history,
):
    """Lift an inner solver's InnerResult into a SolveResult.

    `solved_on` is the Problem the final inner solve ran on -- the basis of
    `inner.objective`. It must be the PRE-augmentation problem on the cap
    path, so the caller passes it explicitly rather than letting this
    reach for whatever the loop variable happens to hold.
    """
    return SolveResult(
        objective=inner.objective,
        inner_solve_count=inner_solve_count,
        augmentation_rounds=augmentation_rounds,
        total_witnesses_added=island_added + component_added,
        total_island_witnesses_added=island_added,
        total_component_witnesses_added=component_added,
        final_K_phi=len(solved_on.samples_1),
        final_K_psi=len(solved_on.samples_2),
        pass_history=tuple(pass_history),
        reason=reason,
        stable_islands=inner.stable_islands,
        unstable_islands=inner.unstable_islands,
        reconnected_edges=inner.reconnected_edges,
        tree=inner.info.get("tree"),
        info=inner.info,
    )


def solve(problem, max_outer_iterations, inner_solver=local_search, initial_tree=None):
    """Repeatedly run `inner_solver`, then grow the sample sets with any
    counterexample found, until convergence or the outer-iteration cap.

    `inner_solver(problem)` returns an InnerResult; the loop uses that
    partition directly and never decodes a tree.

    `initial_tree` is a local-search-specific warm start, so it is
    forwarded only when actually supplied, and only on the first pass. An
    inner solver that takes no such argument (a MILP) is therefore never
    handed one. local_search re-initializes its own tree on every later
    pass, so nothing but the (growing) sample sets carries between passes.

    `max_outer_iterations` caps the number of INNER SOLVES, which is what
    the loop variable counts -- not the number of augmentations, which is
    one fewer whenever the loop converges. See SolveResult for why the two
    are reported separately.

    An InnerSolveFailure from any pass propagates, but not before the loop
    attaches its own state to it (which pass, at which K's, with what
    history behind it). A caller recording that failure must use the
    attached `outer_state`, not the problem it originally passed in: after
    even one augmentation the two describe different sample sets.

    `max_outer_iterations` must be at least 1. Rejected rather than
    accommodated: with a cap of zero the loop body never runs, so there is
    no partition, no objective and no sample basis to report, and any
    SolveResult invented for that case would be a record of a run that
    never happened. (This used to fall through to the cap-return path and
    raise UnboundLocalError on the unbound `result`.)
    """
    if max_outer_iterations < 1:
        raise ValueError(
            f"max_outer_iterations must be at least 1; got "
            f"{max_outer_iterations!r}. The loop has nothing to return "
            "without at least one inner solve."
        )

    current = problem
    island_added = 0
    component_added = 0
    inner_solve_count = 0
    augmentation_rounds = 0
    pass_history = []

    for outer in range(max_outer_iterations):
        warm_start = (
            {"initial_tree": initial_tree}
            if outer == 0 and initial_tree is not None
            else {}
        )
        try:
            result = inner_solver(current, **warm_start)
        except InnerSolveFailure as failure:
            # The solver has its own diagnostics, but only this loop knows
            # WHICH pass failed and what sample sets that pass was solving
            # against. Without this, a failure on any pass after the first
            # is recorded against the ORIGINAL problem -- config.P_1/P_2
            # from the initial sample sets next to a model size and bound
            # from a larger augmented one, an internally contradictory
            # record. Everything accumulated so far would be lost too.
            #
            # `current` is the problem this pass was handed, the same
            # capture `solved_on` makes on the success paths.
            raise failure.attach_outer_state(
                failed_pass=outer,
                K_phi=len(current.samples_1),
                K_psi=len(current.samples_2),
                # Completed solves, not counting the one that just failed.
                inner_solve_count=inner_solve_count,
                augmentation_rounds=augmentation_rounds,
                total_island_witnesses_added=island_added,
                total_component_witnesses_added=component_added,
                total_witnesses_added=island_added + component_added,
                pass_history=tuple(pass_history),
            )
        inner_solve_count += 1
        # The problem `result.objective` was computed on. Captured before
        # the augmentation below, because on the cap path the loop exits
        # AFTER growing the sample sets one last time and that growth is
        # not reflected in the objective being returned.
        solved_on = current

        # .get() rather than []: an inner solver need not report a reason
        # (a MILP's info carries different keys).
        if result.info.get("reason") == "max_iterations":
            print(
                f"WARNING: inner solver hit max_iterations on outer pass "
                f"{outer}; objective = {result.objective}",
                flush=True,
            )

        island_witnesses, component_witnesses = find_counterexamples_for_partition(
            result.stable_islands, result.reconnected_edges, current
        )

        pass_history.append({
            "pass": outer,
            "K_phi": len(current.samples_1),
            "K_psi": len(current.samples_2),
            "objective": result.objective,
            "island_witnesses_found": len(island_witnesses),
            "component_witnesses_found": len(component_witnesses),
            **_scalar_info(result.info),
        })

        if not island_witnesses and not component_witnesses:
            # This solve found nothing to add, so no augmentation follows
            # it: the two counts differ by one on this path.
            return _solve_result(
                result, solved_on, inner_solve_count, augmentation_rounds,
                island_added, component_added, "converged", pass_history,
            )

        island_added += len(island_witnesses)
        component_added += len(component_witnesses)
        augmentation_rounds += 1
        current = _augment(current, island_witnesses, component_witnesses)

    # Cap path. The final pass augmented and then ran out of iterations, so
    # inner_solve_count == augmentation_rounds == max_outer_iterations, and
    # the witnesses from that last pass were never solved against -- which
    # is exactly why `solved_on` is the pre-augmentation problem.
    return _solve_result(
        result, solved_on, inner_solve_count, augmentation_rounds,
        island_added, component_added, "max_outer_iterations", pass_history,
    )
