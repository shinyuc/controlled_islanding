"""Local search over spanning trees of the augmented graph (Algorithm 1,
lines 4-12). See docs/local_search_spec.md for the full specification.
"""

from dataclasses import dataclass

import networkx as nx

from islanding.augmented import SOURCE, SINK, augmented_graph
from islanding.search import initial_spanning_tree
from islanding.evaluate import evaluate


@dataclass(frozen=True)
class SearchResult:
    tree: object
    objective: float
    reason: str  # "zero_objective" | "local_optimum" | "max_iterations"
    iterations: int


def _is_supernode(node):
    return node == SOURCE or node == SINK


def _removal_candidates(tree, swap_supernode_edges):
    candidates = []
    for u, v in tree.edges():
        if {u, v} == {SOURCE, SINK}:
            continue
        if not swap_supernode_edges and (_is_supernode(u) or _is_supernode(v)):
            continue
        candidates.append((u, v))
    return candidates


def _relay_addition_candidates(tree, topology):
    """Family 1: for each cut original edge, the single missing relay edge.

    Enumerated by walking original edges (topology.edges()), not by
    scanning Gbar \\ A. Per the spec's four-configuration table, a relay
    path can never have both of a relay node's edges absent, so a cut
    edge always has exactly one missing edge to add back.
    """
    candidates = []
    for u, v in topology.edges():
        i, j = (u, v) if u < v else (v, u)
        relay_i, relay_j = ("n", i, j, i), ("n", i, j, j)
        edge_i, middle, edge_j = (i, relay_i), (relay_i, relay_j), (relay_j, j)

        has_i = tree.has_edge(*edge_i)
        has_mid = tree.has_edge(*middle)
        has_j = tree.has_edge(*edge_j)

        if has_i and has_mid and has_j:
            continue  # through -- nothing to add
        if has_i and has_j:
            candidates.append(middle)  # cut in middle
        elif has_mid and has_j:
            candidates.append(edge_i)  # cut near i
        elif has_i and has_mid:
            candidates.append(edge_j)  # cut near j
    return candidates


def _supernode_addition_candidates(topology, swap_supernode_edges):
    """Family 2: (SOURCE, i) or (SINK, i) for any node i, when enabled."""
    if not swap_supernode_edges:
        return []
    candidates = []
    for node in topology.nodes():
        candidates.append((SOURCE, node))
        candidates.append((SINK, node))
    return candidates


def _feasible_swaps(tree, topology, swap_supernode_edges):
    """All (removal, addition) pairs that keep the result a spanning tree.

    For each removal candidate, removing it from the tree splits it into
    exactly two components (a tree property). An addition is feasible for
    that removal iff its endpoints land in different components -- this
    also automatically excludes any addition edge already present in the
    tree, since such an edge's endpoints are trivially still connected
    (through itself) after removing a different edge.
    """
    removals = _removal_candidates(tree, swap_supernode_edges)
    additions = (
        _relay_addition_candidates(tree, topology)
        + _supernode_addition_candidates(topology, swap_supernode_edges)
    )

    swaps = []
    for removal in removals:
        remainder = tree.copy()
        remainder.remove_edge(*removal)
        component_of = {}
        for index, component in enumerate(nx.connected_components(remainder)):
            for node in component:
                component_of[node] = index

        for addition in additions:
            u, v = addition
            if component_of.get(u) != component_of.get(v):
                swaps.append((removal, addition))

    return swaps


def local_search(problem, initial_tree=None):
    """Locally optimal spanning tree of Gbar for the given (fixed-sample) Problem."""
    if initial_tree is None:
        tree = initial_spanning_tree(augmented_graph(problem.topology))
    else:
        tree = initial_tree

    config = problem.config
    iterations = 0

    while True:
        objective = evaluate(tree, problem)
        assert objective >= -1e-9, (
            f"F_hat must never be negative (got {objective}); this indicates "
            "a broken invariant (e.g. kappa_op <= kappa_rec, or a negative cost)."
        )

        if objective == 0:
            return SearchResult(
                tree=tree, objective=objective,
                reason="zero_objective", iterations=iterations,
            )

        if iterations >= config.max_iterations:
            return SearchResult(
                tree=tree, objective=objective,
                reason="max_iterations", iterations=iterations,
            )

        swaps = _feasible_swaps(tree, problem.topology, config.swap_supernode_edges)

        best_swap = None
        best_value = None
        for removal, addition in swaps:
            candidate = tree.copy()
            candidate.remove_edge(*removal)
            candidate.add_edge(*addition)
            value = evaluate(candidate, problem)
            if best_value is None or value < best_value:
                best_value = value
                best_swap = candidate

        if best_swap is None or not (objective > (1 + config.epsilon) * best_value):
            return SearchResult(
                tree=tree, objective=objective,
                reason="local_optimum", iterations=iterations,
            )

        tree = best_swap
        iterations += 1
