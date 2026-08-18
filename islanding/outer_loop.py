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

from islanding.local_search import local_search
from islanding.counterexample import find_counterexamples_for_partition


@dataclass(frozen=True)
class SolveResult:
    objective: float
    outer_iterations: int
    witnesses_added: int
    reason: str  # "converged" | "max_outer_iterations"
    stable_islands: list       # list[frozenset] -- the set S
    unstable_islands: list     # list[frozenset] -- the set U
    reconnected_edges: set     # E^rec, canonical (min, max) tuples
    # Optional: only inner solvers that use the matroid encoding have one.
    # A MILP will leave this None, so read the partition fields instead.
    tree: object = None
    info: dict = field(default_factory=dict)  # the inner solver's diagnostics


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


def _solve_result(inner, outer_iterations, witnesses_added, reason):
    """Lift an inner solver's InnerResult into a SolveResult."""
    return SolveResult(
        objective=inner.objective,
        outer_iterations=outer_iterations,
        witnesses_added=witnesses_added,
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
    """
    current = problem
    total_added = 0

    for outer in range(max_outer_iterations):
        warm_start = (
            {"initial_tree": initial_tree}
            if outer == 0 and initial_tree is not None
            else {}
        )
        result = inner_solver(current, **warm_start)

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

        if not island_witnesses and not component_witnesses:
            return _solve_result(result, outer, total_added, "converged")

        total_added += len(island_witnesses) + len(component_witnesses)
        current = _augment(current, island_witnesses, component_witnesses)

    return _solve_result(
        result, max_outer_iterations, total_added, "max_outer_iterations"
    )
