"""Cost function F = F1 + F2 + F3 - F4 for a decoded islanding strategy.

Operates on decoded original-graph quantities (stable islands, unstable
islands, reconnected edges) plus per-instance cost data and algorithm
parameters. See docs/cost_function_spec.md for the full math.
"""

from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass(frozen=True)
class CostData:
    node_costs: dict
    edge_costs: dict


@dataclass(frozen=True)
class AlgorithmConfig:
    kappa_op: float
    kappa_rec: float
    alpha_1: float = 1.0
    beta: float = 1.0
    k_1: float = 1.0
    P_1: int = 1
    alpha_2: float = 1.0
    gamma: float = 1.0
    k_2: float = 1.0
    P_2: int = 1
    epsilon: float = 0.01
    max_iterations: int = 100
    swap_supernode_edges: bool = True

    def __post_init__(self):
        assert self.kappa_op > self.kappa_rec
        assert self.alpha_1 > 0
        assert self.beta > 0
        assert self.k_1 > 0
        assert self.P_1 > 0
        assert self.alpha_2 > 0
        assert self.gamma > 0
        assert self.k_2 > 0
        assert self.P_2 > 0
        assert self.epsilon > 0
        # max_iterations == 0 is a legal, meaningful config (immediate stop
        # with reason="max_iterations"), exercised by local_search_spec.md
        # section 7 assertion 4 -- so this is >= 0, not strictly positive.
        assert self.max_iterations >= 0


def canon(edge):
    i, j = edge
    return (i, j) if i < j else (j, i)


def _induced_edges(nodes, topology):
    return {
        canon((u, v))
        for u, v in topology.edges()
        if u in nodes and v in nodes
    }


def f1_unstable_nodes(unstable_islands, cost_data):
    failed_nodes = frozenset().union(*unstable_islands) if unstable_islands else frozenset()
    return sum(cost_data.node_costs[i] for i in failed_nodes)


def f2_unstable_edges(unstable_islands, topology, cost_data):
    total = 0.0
    for island in unstable_islands:
        for edge in _induced_edges(island, topology):
            total += cost_data.edge_costs[edge]
    return total


def f3_operation_edges(stable_islands, unstable_islands, topology, cost_data, config):
    all_edges = {canon((u, v)) for u, v in topology.edges()}

    within_island_edges = set()
    for island in list(stable_islands) + list(unstable_islands):
        within_island_edges |= _induced_edges(island, topology)

    cut_edges = all_edges - within_island_edges
    return config.kappa_op * sum(cost_data.edge_costs[edge] for edge in cut_edges)


def f4_reconnection(reconnected_edges, cost_data, config):
    return config.kappa_rec * sum(
        cost_data.edge_costs[canon(edge)] for edge in reconnected_edges
    )


def F(f1, f2, f3, f4, f5=0.0, f6=0.0):
    return f1 + f2 + f3 - f4 + f5 + f6


@dataclass(frozen=True)
class SystemModel:
    """Bundles the seven callables F5 needs (spec section 10).

    Signatures (all node/edge-level callables take identifiers, not just
    state, so a model's per-node/per-edge weights can be looked up
    consistently -- unlike the plain (x_i) / (x_i, x_j) forms in the spec
    table, which omit them):
        f_i(i, x_i) -> vector                     node dynamics
        f_ij(i, j, x_i, x_j) -> vector             coupling dynamics
        W_i(i, x_i) -> scalar                      Lyapunov, node term
        W_ij(i, j, x_i, x_j) -> scalar              Lyapunov, edge term
        grad_W_i(i, x_i) -> vector                 gradient, node term
        grad_W_ij(i, j, x_i, x_j) -> (d/dx_i, d/dx_j)   gradient, both endpoints
        sampler(threshold, P1, n, d) -> (P1, n, d) array
        inflation(topology) -> float

    Linear consensus's Lyapunov terms (W_i, W_ij, grad_W_i, grad_W_ij) are
    weight-independent (constant or zero) and simply ignore i, j; only
    f_i and f_ij actually use them, to look up w_i / w_ij. f_ij(i, j, ...)
    and W_ij/grad_W_ij(i, j, ...) are called with i as "self" and j as
    "neighbour" in whatever order the caller needs (the underlying weight
    lookup is symmetric via canon()), so callers never need to sign-flip
    based on canonical (min, max) order.

    inflation(topology) returns max_i{d_i * alpha_bar_i}, where d_i is
    node i's degree in the original graph -- an inflation factor applied
    to BOTH sublevel-set radii before sampling:
        Omega     = {x : W(x) <= (1 + inflation) * beta}
        Omega_hat = {x : W(x) <= (1 + inflation) * gamma}
    sampler itself stays inflation-agnostic: it always takes a plain
    threshold (already inflated by the caller), never beta/gamma directly.
    """

    f_i: Callable
    f_ij: Callable
    W_i: Callable
    W_ij: Callable
    grad_W_i: Callable
    grad_W_ij: Callable
    sampler: Callable
    inflation: Callable


def linear_consensus_model(node_weights, edge_weights):
    """SystemModel for grounded linear consensus (spec section 10).

    x_dot_i = -w_i * x_i - sum_j w_ij * (x_i - x_j), w_i, w_ij > 0.
    Lyapunov function W_i(x_i) = 0.5 * x_i . x_i is weight-independent;
    W_ij and grad_W_ij are identically zero for this model.
    """

    def f_i(i, x_i):
        return -node_weights[i] * x_i

    def f_ij(i, j, x_i, x_j):
        w_ij = edge_weights[canon((i, j))]
        return -w_ij * (x_i - x_j)

    def W_i(i, x_i):
        return 0.5 * float(np.dot(x_i, x_i))

    def W_ij(i, j, x_i, x_j):
        return 0.0

    def grad_W_i(i, x_i):
        return x_i

    def grad_W_ij(i, j, x_i, x_j):
        return (np.zeros_like(x_i), np.zeros_like(x_j))

    def sampler(threshold, P1, n, d):
        dim = n * d
        directions = np.random.normal(size=(P1, dim))
        directions /= np.linalg.norm(directions, axis=1, keepdims=True)
        radial_fractions = np.random.uniform(size=(P1, 1)) ** (1.0 / dim)
        radius = (2.0 * threshold) ** 0.5
        points = directions * radial_fractions * radius
        return points.reshape(P1, n, d)

    def inflation(topology):
        return 0.0

    return SystemModel(
        f_i=f_i, f_ij=f_ij, W_i=W_i, W_ij=W_ij,
        grad_W_i=grad_W_i, grad_W_ij=grad_W_ij, sampler=sampler,
        inflation=inflation,
    )


def _stability_bracket(nodes, edges, neighbors, states, model, k):
    """Shared bracket computation for F5 and F6 (spec sections 9 and 14).

    `nodes` is the unit's node set (an island for F5, a reconnected
    component for F6), `edges` its within-unit edge set, `neighbors[i]`
    the set of within-unit neighbours of i, and `states` a dict of
    node -> x_i for this sample. Returns max(bracket, 0.0).
    """
    grad_W = {i: model.grad_W_i(i, states[i]) for i in nodes}
    for u, v in edges:
        grad_u, grad_v = model.grad_W_ij(u, v, states[u], states[v])
        grad_W[u] = grad_W[u] + grad_u
        grad_W[v] = grad_W[v] + grad_v

    dyn = {}
    for i in nodes:
        value = model.f_i(i, states[i])
        for j in neighbors[i]:
            value = value + model.f_ij(i, j, states[i], states[j])
        dyn[i] = value

    W = sum(model.W_i(i, states[i]) for i in nodes)
    W += sum(model.W_ij(u, v, states[u], states[v]) for u, v in edges)

    bracket = sum(float(np.dot(grad_W[i], dyn[i])) for i in nodes) + k * W
    return max(bracket, 0.0)


def f5_stability_penalty(stable_islands, topology, model, samples, config):
    """Stability-penalty term F5 (spec section 9).

    Evaluated per sample and per stable island, using ONLY within-island
    edges for both the dynamics sum and the Lyapunov gradient/value (the
    island is treated as if fully disconnected from the rest of the
    system). Node order along the samples' middle axis is `sorted(topology
    .nodes())`.
    """
    if len(samples) != config.P_1:
        raise ValueError(
            f"len(samples) ({len(samples)}) != config.P_1 ({config.P_1}); "
            "F5 would silently average by the wrong denominator."
        )

    nodes = sorted(topology.nodes())
    node_index = {node: idx for idx, node in enumerate(nodes)}

    total = 0.0
    for sample in samples:
        for island in stable_islands:
            edges_r = _induced_edges(island, topology)

            neighbors = {i: set() for i in island}
            for u, v in edges_r:
                neighbors[u].add(v)
                neighbors[v].add(u)

            states = {i: sample[node_index[i]] for i in island}

            total += _stability_bracket(
                island, edges_r, neighbors, states, model, config.k_1
            )

    return config.alpha_1 / config.P_1 * total


def f6_reconnection_stability(components, topology, model, samples, config):
    """Stability-penalty term F6 (spec section 14).

    Evaluated per sample and per reconnected component. Each component is
    an nx.Graph already carrying exactly V_s^rec and E_s^rec (per
    decode_reconnection's construction), so component.edges() and
    component.neighbors(i) are used directly -- no within-unit filtering
    needed here, unlike F5.

    topology is used ONLY for node ordering (identical to
    f5_stability_penalty): node order along the samples' middle axis is
    `sorted(topology.nodes())`, matching how `samples` was generated over
    the full node set.
    """
    if len(samples) != config.P_2:
        raise ValueError(
            f"len(samples) ({len(samples)}) != config.P_2 ({config.P_2}); "
            "F6 would silently average by the wrong denominator."
        )

    nodes_all = sorted(topology.nodes())
    node_index = {node: idx for idx, node in enumerate(nodes_all)}

    total = 0.0
    for sample in samples:
        for component in components:
            nodes = list(component.nodes())
            edges = list(component.edges())
            neighbors = {i: set(component.neighbors(i)) for i in nodes}
            states = {i: sample[node_index[i]] for i in nodes}

            total += _stability_bracket(
                nodes, edges, neighbors, states, model, config.k_2
            )

    return config.alpha_2 / config.P_2 * total
