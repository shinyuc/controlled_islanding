"""Constraint-generation outer loop wrapping local_search (Algorithm 1,
lines 13-19). See docs/outer_loop_spec.md for the full specification.

The algorithm itself (decode, six terms, local search, eigenvector
counterexamples) is unchanged -- this layer only grows the sample sets
and re-runs local_search.
"""

from dataclasses import dataclass, replace

import numpy as np

from islanding.local_search import local_search
from islanding.counterexample import find_counterexamples


@dataclass(frozen=True)
class SolveResult:
    tree: object
    objective: float
    outer_iterations: int
    witnesses_added: int
    reason: str  # "converged" | "max_outer_iterations"


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


def solve(problem, max_outer_iterations, initial_tree=None):
    """Repeatedly local_search, then grow the sample sets with any
    counterexample found, until convergence or the outer-iteration cap.

    Only the first pass uses `initial_tree`; local_search re-initializes
    its own tree on every later pass, so nothing but the (growing) sample
    sets carries between passes.
    """
    current = problem
    total_added = 0

    for outer in range(max_outer_iterations):
        result = local_search(current, initial_tree=initial_tree if outer == 0 else None)
        island_witnesses, component_witnesses = find_counterexamples(result.tree, current)

        if not island_witnesses and not component_witnesses:
            return SolveResult(
                result.tree, result.objective, outer, total_added, "converged",
            )

        total_added += len(island_witnesses) + len(component_witnesses)
        current = _augment(current, island_witnesses, component_witnesses)

    return SolveResult(
        result.tree, result.objective, max_outer_iterations, total_added,
        "max_outer_iterations",
    )
