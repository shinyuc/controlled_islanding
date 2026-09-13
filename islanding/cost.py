"""Cost function F for a decoded islanding strategy.

Operates on decoded original-graph quantities (stable islands, unstable
islands, reconnected edges) plus per-instance cost data and algorithm
parameters. See docs/cost_function_spec.md for the full math.

The production formulation is F = F1 + F3 - F4 + F5 + F6. F2 (the cost of
edges inside unstable islands) was DROPPED from it -- not disabled pending
a decision. `F` keeps a generic six-argument signature, f2 slot included,
for compatibility with the spec's published form and with the tests that
still exercise the term; every production caller passes 0.0 there.
`f2_unstable_edges` likewise remains here, computed by nothing in the
pipeline.
"""

import math
import weakref
from dataclasses import dataclass
from typing import Callable, NamedTuple

import numpy as np

from islanding.counterexample import linear_counterexample



# Validation below raises ValueError, never `assert`. These checks are what
# make utils.save's persisted `proof_basis` -- "objective is nonnegative;
# zero attained" -- a true statement rather than an aspiration, and
# `python -O` strips assert statements entirely. A claim that survives into
# a saved artifact must not rest on a check that disappears under an
# interpreter flag.


def _require_finite(name, value):
    """`value` as a float, rejecting NaN and infinities.

    Nonfinite parameters do not merely produce odd numbers: inf * 0 is NaN,
    so a single infinite coefficient makes F undefined rather than large,
    and every downstream comparison (including "is the objective zero?")
    silently returns False.
    """
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a real number; got {value!r}") from None
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite; got {number!r}")
    return number


def _validate_costs(kind, costs):
    """Every cost must be finite and nonnegative.

    NaN needs the explicit finiteness check: `nan < 0` is False, so a NaN
    cost passes any comparison-based test and then poisons every sum it
    enters.
    """
    for key, value in costs.items():
        number = _require_finite(f"{kind} cost for {key!r}", value)
        if number < 0:
            raise ValueError(
                f"{kind} costs must be nonnegative; got {number!r} for "
                f"{key!r}. A negative cost breaks F_hat >= 0, on which the "
                "zero-objective optimality proof depends."
            )


@dataclass(frozen=True)
class CostData:
    """Per-instance node and edge costs. Both must be NONNEGATIVE.

    Not a stylistic constraint. F1 and F3 are nonnegative sums of these,
    and F3 - F4 >= (kappa_op - kappa_rec) * (reconnected cost) >= 0 only
    when the costs are, so nonnegativity is what makes F_hat >= 0 -- and
    therefore what makes local_search's "zero_objective" exit an actual
    proof of global optimality rather than a claim about the trees it
    happened to visit. With a negative edge cost a two-node instance can
    report zero while a one-swap partition reaches -1.

    Validated here rather than trusted, so utils.save.save_result can
    record that proof as a flag. See also AlgorithmConfig's kappa_op > 0.
    """

    node_costs: dict
    edge_costs: dict

    def __post_init__(self):
        _validate_costs("node", self.node_costs)
        _validate_costs("edge", self.edge_costs)


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
    # Eigenvalues <= this count as instabilities in the counterexample
    # search (islanding/counterexample.py). The mathematical criterion is
    # lambda <= 0, but eigh returns a structurally zero eigenvalue as a
    # small value of either sign, and Laplacian-like M_u matrices have such
    # zeros by construction -- so an exact `lambda > 0` test silently
    # classifies a marginally unstable unit as stable, and the outer loop
    # then reports "converged" on a false stability claim.
    #
    # PER-CASE, like the time limit: the conditioning of M_u worsens with
    # unit size and weight spread, so the size of the numerical noise
    # around a true zero is instance-dependent. 1e-9 is a starting point
    # for the small cases here, not a universal constant. Appended at the
    # end of the field list so no positional construction shifts.
    stability_tolerance: float = 1e-9

    def __post_init__(self):
        # Finiteness first: every comparison below returns False on a NaN,
        # so an unchecked NaN would pass the whole block. The six the
        # objective is linear in (kappa_op, kappa_rec, alpha_1, alpha_2,
        # k_1, k_2) matter most, but beta/gamma/epsilon are the same class
        # -- an infinite sampling radius or improvement threshold makes the
        # objective just as undefined -- so all nine are checked here.
        for name in ("kappa_op", "kappa_rec", "alpha_1", "alpha_2",
                     "beta", "gamma", "k_1", "k_2", "epsilon",
                     "stability_tolerance"):
            _require_finite(name, getattr(self, name))

        if not self.kappa_op > self.kappa_rec:
            raise ValueError(
                f"kappa_op must exceed kappa_rec; got kappa_op="
                f"{self.kappa_op!r}, kappa_rec={self.kappa_rec!r}"
            )
        # kappa_op > kappa_rec alone permits a negative kappa_op, which
        # would make F3 (a nonnegative cut cost scaled by kappa_op)
        # negative and break F_hat >= 0. Together with nonnegative
        # CostData this is what the zero-objective optimality proof needs:
        # F3 - F4 >= (kappa_op - kappa_rec) * (reconnected cost) >= 0,
        # since E^rec is a subset of the cut edges.
        if not self.kappa_op > 0:
            raise ValueError(
                f"kappa_op must be positive; got {self.kappa_op!r}. A "
                "nonpositive kappa_op makes F3 negative and breaks "
                "F_hat >= 0, on which the zero-objective optimality proof "
                "depends."
            )
        # kappa_rec > 0 is the two-stage reading (spec section 7): F3
        # charges kappa_op for cutting an edge and F4 refunds kappa_rec for
        # restoring it, so a reconnected edge nets
        # (kappa_op - kappa_rec) * c_ij. kappa_op > kappa_rec keeps that
        # net positive -- reconnection is never free -- and kappa_rec > 0
        # keeps it a REFUND: at kappa_rec <= 0 restoring an edge would cost
        # at least as much as leaving it cut, which inverts the term's
        # meaning rather than merely retuning it.
        #
        # Checked after kappa_op > 0 on purpose. Once this holds,
        # kappa_op > kappa_rec > 0 implies kappa_op > 0, so the check above
        # can no longer fire on its own; keeping it ordered first preserves
        # its specific message for a config that violates only it.
        if not self.kappa_rec > 0:
            raise ValueError(
                f"kappa_rec must be positive; got {self.kappa_rec!r}. A "
                "nonpositive reconnection weight makes restoring an edge "
                "cost at least as much as leaving it cut, contradicting "
                "the two-stage reading of F3/F4."
            )
        # Cross-constraint from the revised F6 theory (spec section 15):
        # k_1 >= k_2 > 0. Note >=, not strict -- k_1 == k_2 is allowed, and
        # makes F6's Part 4 vanish.
        if not self.k_1 >= self.k_2 > 0:
            raise ValueError(
                f"require k_1 >= k_2 > 0; got k_1={self.k_1!r}, "
                f"k_2={self.k_2!r}"
            )
        for name in ("alpha_1", "beta", "k_1", "alpha_2", "gamma", "k_2",
                     "P_1", "P_2", "epsilon"):
            value = getattr(self, name)
            if not value > 0:
                raise ValueError(f"{name} must be positive; got {value!r}")
        # max_iterations == 0 is a legal, meaningful config (immediate stop
        # with reason="max_iterations"), exercised by local_search_spec.md
        # section 7 assertion 4 -- so this is >= 0, not strictly positive.
        # >= 0, not > 0: zero is meaningful and recovers the exact
        # `lambda <= 0` criterion for a caller that wants it.
        if not self.stability_tolerance >= 0:
            raise ValueError(
                f"stability_tolerance must be nonnegative; got "
                f"{self.stability_tolerance!r}"
            )
        if not self.max_iterations >= 0:
            raise ValueError(
                f"max_iterations must be nonnegative; got "
                f"{self.max_iterations!r}"
            )


def canon(edge):
    i, j = edge
    return (i, j) if i < j else (j, i)


def _induced_edges(nodes, topology):
    """Within-island edges of `nodes`, cached per (topology, node set).

    For a fixed topology the result depends only on the node set, but this
    was recomputed over the FULL edge list on every call: 7.7M calls and
    75.4 s of the 20-node profile's 596 s.

    The cache lives on the Topology instance, so its lifetime is exactly
    the topology's and it cannot leak between problems. A module-level
    dict keyed by id(topology) would be wrong -- CPython reuses ids after
    garbage collection, so a freed topology's id could be handed to a new
    one and silently serve it another graph's edges. Keying by the
    topology object itself would need Topology to be hashable and would
    keep every topology alive for the process's lifetime.

    Returns a TUPLE, so the cached value cannot be corrupted. A shared
    mutable set would leave one stray `.pop()` or `.clear()` silently
    wrong for every later evaluation of that island -- and "no caller
    mutates it" was only a convention, not a guarantee.

    The tuple is built from the set's OWN iteration order and then frozen.
    That is what keeps results bit-identical rather than merely equal:
    iteration order feeds floating-point summation order in
    _stability_bracket (grad accumulation and the W_ij sum) and in
    _psi_reconnection, and float addition is not associative. The edges are
    deliberately NOT sorted -- sorting would also be deterministic, but a
    DIFFERENT order, and would change historical results.

    A tuple is free here because no consumer does a membership test on the
    return value; every use is iteration (`for u, v in edges_r`) or, in
    f3_operation_edges, a set `.update()`. Were there an `edge in edges`
    test, this would have to stay a set: O(1) lookups becoming O(n) could
    cost more than the caching saves.
    """
    key = frozenset(nodes)
    cache = topology._induced_edge_cache
    edges = cache.get(key)
    if edges is None:
        # The original comprehension, unchanged, so the order captured
        # below is exactly the order this function always produced.
        edges = tuple({
            canon((u, v))
            for u, v in topology.edges()
            if u in nodes and v in nodes
        })
        cache[key] = edges
    return edges


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
        # .update(), not |=: _induced_edges returns a tuple now. Same
        # insertion order, so within_island_edges is built identically --
        # and cut_edges' own iteration order comes from all_edges anyway,
        # since set difference iterates the left operand.
        within_island_edges.update(_induced_edges(island, topology))

    cut_edges = all_edges - within_island_edges
    return config.kappa_op * sum(cost_data.edge_costs[edge] for edge in cut_edges)


def f4_reconnection(reconnected_edges, cost_data, config):
    return config.kappa_rec * sum(
        cost_data.edge_costs[canon(edge)] for edge in reconnected_edges
    )


def F(f1, f2, f3, f4, f5=0.0, f6=0.0):
    return f1 + f2 + f3 - f4 + f5 + f6


class _StateDomain(NamedTuple):
    """The (shape, dtype) signature a set of node states shares.

    Every verification registry in this file is keyed on a structural
    identity PLUS the domain the check ran over, and the domain half has
    now been got wrong three times: the F5 closed form's key omitted d,
    then omitted dtype, and _verify_zero_pairwise's key omitted both. The
    shape of the failure is identical each time -- an entry recorded under
    one domain suppresses a later, materially different check -- so the
    domain is built by one shared helper rather than assembled by hand at
    each site.

    Only the domain is shared. The rest of each key is genuinely local:
    F5's is (island nodes, induced edges, k_1), and
    _verify_zero_pairwise's is the edge. Factoring those together would
    force a common shape on keys that have no reason to share one.
    """

    shape: tuple
    dtype: np.dtype


def _state_domain(states, where):
    """The domain `states` share, validated rather than assumed.

    `states` is either a sequence of per-node 1-D state vectors, or a
    single 2-D ndarray whose rows are those vectors.

    VALIDATES EQUALITY ACROSS STATES, which is the point. Reading the
    domain off the FIRST state alone is what makes a mixed call
    dangerous: _verify_zero_pairwise takes a generic dict, so a call whose
    two endpoints differ in shape or dtype would register under the first
    endpoint's signature and suppress a later check on a materially
    different domain. Every production caller passes slices of one
    homogeneous samples array, so this cannot happen today -- which is
    exactly why it would go unnoticed if it ever could.
    """
    if isinstance(states, np.ndarray) and states.ndim == 2:
        # A single array cannot hold rows of differing shape or dtype, so
        # the equality check below would be vacuous. Shape is validated,
        # which is the part that can still be wrong.
        return _StateDomain((states.shape[1],), states.dtype)

    arrays = [np.asarray(x) for x in states]
    if not arrays:
        raise ValueError(f"{where}: no states to take a domain from")

    first = arrays[0]
    if first.ndim != 1:
        raise ValueError(
            f"{where}: states must be 1-D vectors, got shape {first.shape}. "
            "The probe set and the registry key are both built from this "
            "signature, so a state of another rank would be checked in a "
            "domain it does not belong to."
        )
    for other in arrays[1:]:
        if other.shape != first.shape or other.dtype != first.dtype:
            raise ValueError(
                f"{where}: states do not share a domain -- "
                f"{first.shape}/{first.dtype} against "
                f"{other.shape}/{other.dtype}. One signature cannot "
                "describe both, and recording the check under either "
                "would suppress a later one on the other."
            )
    return _StateDomain(first.shape, first.dtype)


# Shared read-only zero arrays, keyed by (shape, dtype). Module level
# rather than per model: the arrays are immutable and carry no model
# state, and _psi_reconnection needs one for its reconnection-flow
# accumulator without having a model closure to reach into.
_zero_arrays = {}


def _shared_zero(like):
    """A cached read-only zeros array with `like`'s shape and dtype.

    THE CONTRACT, stated narrowly. This is a correct substitute for
    np.zeros_like on ORDINARY ndarrays whose (shape, dtype) determine the
    result -- which is what every caller passes: a 1-D slice of a samples
    array. NOT float64 only. The key carries dtype precisely so a float32
    state gets a float32 zero, which _verify_zero_pairwise now depends on:
    it checks declarations in whatever precision the run supplies rather
    than restricting to one.

    It is NOT a general equivalent, because the key omits two things
    zeros_like honours:
      - memory order. A C-contiguous (2,2) followed by an F-contiguous
        (2,2) gets the cached C-contiguous zero, where zeros_like would
        return F-contiguous.
      - subclass. An ndarray followed by an np.matrix of the same shape
        and dtype gets an ndarray, where zeros_like(matrix) returns a
        matrix.
    Neither reaches the current callers, and neither would change a value
    -- only layout and type. Widen the key before using this with
    F-ordered or subclassed states.

    READ-ONLY, so it is only ever a legitimate STARTING VALUE for an
    accumulator that REBINDS (`acc = acc + term`), never one that
    accumulates in place (`acc += term`) -- the latter would raise, which
    is the point of writeable=False.
    """
    key = (like.shape, like.dtype)
    zero = _zero_arrays.get(key)
    if zero is None:
        zero = np.zeros_like(like)
        zero.flags.writeable = False
        _zero_arrays[key] = zero
    return zero


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
        counterexample(unit_nodes, unit_edges, node_order, d,
                       threshold_radius, stability_tolerance)
            -> list of (n, d) witness arrays

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
    counterexample: Callable
    # OPTIONAL fast path, following the same pattern as sampler/inflation/
    # counterexample: a model that can express its F5 bracket as a
    # quadratic form supplies one, and f5_stability_penalty evaluates every
    # sample in a single einsum instead of a per-node, per-edge Python
    # loop. None means "no closed form" and the loop runs unchanged.
    #
    # Only valid where the bracket really IS quadratic in the island state.
    # Linear consensus qualifies because W_ij == 0 and the dynamics are
    # linear; Kuramoto and any model with nonzero edge energy do not, and
    # must leave this None.
    #
    # Verified at the point of USE, in f5_stability_penalty, against the
    # model's own callables -- per island, so an island that happens to
    # agree cannot vouch for the others. A DIRECT call to
    # model.island_quadratic_form(...) bypasses that check entirely. That
    # is accepted rather than closed: no solver path makes one -- the
    # MILP, the counterexample search and F6 all ignore this field -- so
    # the only direct callers are tests, which is where a raw matrix is
    # legitimately wanted.
    #
    # WARNING -- dataclasses.replace INHERITS this field. Deriving a model
    # by swapping W_ij, grad_W_ij, f_i or f_ij while leaving this set
    # gives a fast path whose closed form no longer describes the model,
    # and F5 then returns a different number with no error and no type
    # mismatch. Anything that changes the energy or the dynamics must pass
    # island_quadratic_form=None. The synthetic edge-energy model in
    # tests/conftest.py is the worked example.
    #
    #   island_quadratic_form(nodes, edges, k) -> (members, B)
    #
    # `members` is the island's nodes in the row/column order of `B`, and
    # the bracket for a state X (rows ordered by `members`) is
    # trace(X.T @ B @ X).
    island_quadratic_form: Callable = None

    # OPTIONAL declaration, same opt-in pattern as the fields above: True
    # asserts that this model's PAIRWISE Lyapunov terms are identically
    # zero -- W_ij(i, j, x_i, x_j) == 0.0 and grad_W_ij(...) == (0, 0) for
    # every edge and every state.
    #
    # What it buys. The within-island gradient
    #     g_i = grad_W_i(x_i) + sum_{j in N_i cap V_r} d/dx_i grad_W_ij
    # is built by a Python loop over E_r in both _stability_bracket and
    # _psi_reconnection. Under this declaration every summand is a zero
    # vector, so the loop adds nothing and is skipped. On the 20-node
    # benchmark that loop ran grad_W_ij 30.0 million times for 17.8 s of
    # pure arithmetic on zeros.
    #
    # A BOOLEAN, not "W_ij is None". Deriving the fact from a None
    # callable would change what the field MEANS -- from "a function you
    # can call" to "a function you must first test for None" -- and W_ij
    # and grad_W_ij are called directly from _stability_bracket's W_r sum,
    # _psi_reconnection's Parts 3 and 4, baseline/milp.py's certificate,
    # and tests. Every one of those call sites would need a guard, and a
    # missed guard is a TypeError at a random sample rather than a
    # declaration that is simply absent. The boolean adds a claim without
    # removing a capability, and defaults to False so every existing model
    # keeps the loop.
    #
    # WARNING -- dataclasses.replace INHERITS this, exactly as it inherits
    # island_quadratic_form. A model derived from linear consensus by
    # giving it a real edge energy keeps the declaration unless it passes
    # zero_pairwise_energy=False, and would then silently compute g_i
    # without its edge terms. tests/conftest.py's synthetic edge-energy
    # model is the worked example, and _verify_zero_pairwise is the
    # runtime backstop.
    zero_pairwise_energy: bool = False


def linear_consensus_model(node_weights, edge_weights):
    """SystemModel for grounded linear consensus (spec section 10).

    x_dot_i = -w_i * x_i - sum_j w_ij * (x_i - x_j), w_i, w_ij > 0.
    Lyapunov function W_i(x_i) = 0.5 * x_i . x_i is weight-independent;
    W_ij and grad_W_ij are identically zero for this model.
    """

    # DEFENSIVE COPIES. Everything below closes over these dictionaries,
    # and island_quadratic_form CACHES matrices built from them. Captured
    # by reference, a later mutation of the caller's dict would be seen
    # immediately by f_i and f_ij while the cache kept returning the
    # matrix built from the old weights -- the two paths then disagree
    # with no error. Reproduced: changing one node weight after a matrix
    # was cached gave 0.0 from the fast path against 16.25 from the loop.
    #
    # Copying rather than versioning the cache key: a version would have
    # to be bumped by whoever mutates, which is exactly the discipline
    # that failed. Copying makes the model's weights immutable from the
    # outside, which is how every caller already treats them -- nothing in
    # islanding/, baseline/ or examples/ mutates a weight dict after
    # construction, so this changes no current behaviour. The copies are
    # shallow, which is enough: the values are floats.
    node_weights = dict(node_weights)
    edge_weights = dict(edge_weights)

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

    # W_ij is identically zero for this model, so grad_W_ij returns a pair
    # of zero vectors. Allocating them fresh on every call dominated the
    # 20-node profile: 51.9M calls to grad_W_ij, 120.5M np.zeros_like
    # calls, 93.3 s of pure allocation out of 596 s total.
    #
    # The arrays are shared and READ-ONLY. Sharing is safe because no
    # consumer mutates the return value -- audited at every call site:
    #   cost.py _stability_bracket:  grad_W[u] = grad_W[u] + grad_u
    #   cost.py _psi_reconnection:   g[u] = g[u] + grad_u
    #     both REBIND to a new array; neither uses +=.
    #   cost.py _psi_reconnection:   np.dot(grad_ij_i, dyn_i)   -- read
    #   baseline/milp.py _sample_cache: stored, then only np.dot'd -- read
    # writeable=False turns any future in-place write into a loud
    # ValueError rather than silent cross-call corruption.
    #
    # Keyed by (shape, dtype) rather than one array per model: a single
    # model may legitimately be evaluated at more than one state dimension
    # d (the d=1 and d=2 F6 tests do exactly that), and the pair's two
    # halves are built from x_i and x_j independently.
    def grad_W_ij(i, j, x_i, x_j):
        return (_shared_zero(x_i), _shared_zero(x_j))

    # Cache of island quadratic forms, on the MODEL rather than the
    # Topology: B is built from the weights, which the topology does not
    # know. Keyed by the induced edge set as well as the node set -- one
    # model can legitimately be evaluated against more than one topology
    # (tests do exactly that), and two topologies can share an island's
    # nodes while inducing different edges.
    _quadratic_forms = {}

    def island_quadratic_form(nodes, edges, k):
        """B with trace(X.T B X) equal to F5's bracket for this island.

        For linear consensus the bracket has a closed form. With
        W_i = 0.5 * x_i.x_i, W_ij = 0, f_i = -w_i x_i and
        f_ij = -w_ij (x_i - x_j):

            dW_r/dx_i = x_i
            f_i + sum_{j in N_i cap V_r} f_ij = -(M_r x_r)_i
            W_r = 0.5 * ||x_r||^2

        with M_r = D_r + L_r, so

            bracket = x_r.(-M_r x_r) + (k/2) ||x_r||^2
                    = x_r^T (-M_r + (k/2) I) x_r.

        That is an identity, not an approximation: M_r is the same matrix
        build_M and linear_counterexample construct. For d > 1 the
        coordinates decouple (the weights are scalars), so the bracket is
        trace(X.T B X) over the (n_r, d) state.
        """
        key = (frozenset(nodes), frozenset(edges), k)
        cached = _quadratic_forms.get(key)
        if cached is None:
            members = sorted(nodes)
            index = {node: i for i, node in enumerate(members)}
            B = np.zeros((len(members), len(members)))
            # -M_r = -(D_r + L_r), built directly.
            for node in members:
                B[index[node], index[node]] -= node_weights[node]
            for u, v in edges:
                w = edge_weights[canon((u, v))]
                a, b = index[u], index[v]
                B[a, b] += w
                B[b, a] += w
                B[a, a] -= w
                B[b, b] -= w
            B[np.diag_indices(len(members))] += k / 2.0
            # Guarded against ACCIDENTAL mutation before caching, for the
            # same reason the induced-edge tuples and the shared zero
            # gradients are: this pair is handed to every later caller for
            # this key, so a reordered `members` or a written-to `B` would
            # silently corrupt every subsequent F5 evaluation. Free -- all
            # consumers only read.
            #
            # This is a guard, NOT immutability, and the difference is
            # verified rather than assumed. B owns its buffer, so
            # `B.setflags(write=True)` re-enables writing and a subsequent
            # store does corrupt the cache; the cache dict itself is
            # reachable through `island_quadratic_form.__closure__` and can
            # be cleared from outside. Both take deliberate effort, which
            # is the point -- the same protection level the shared zero
            # gradients have, and appropriate for internal research code.
            B.flags.writeable = False
            cached = (tuple(members), B)
            _quadratic_forms[key] = cached
        return cached

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
        island_quadratic_form=island_quadratic_form,
        # W_ij and grad_W_ij above are identically zero for this model.
        zero_pairwise_energy=True,
        inflation=inflation,
        counterexample=linear_counterexample(node_weights, edge_weights),
    )


# Edges already checked against a model's zero_pairwise_energy claim.
# WeakKeyDictionary for the same reason as _verified_quadratic_forms: the
# entry dies with the model rather than pinning seven callables and their
# closures for the life of the process.
_verified_zero_pairwise = weakref.WeakKeyDictionary()


def _verify_zero_pairwise(model, edges, states):
    """Check a model's zero_pairwise_energy claim against its callables.

    THE HAZARD, and it is the same one island_quadratic_form has:
    `dataclasses.replace` INHERITS the declaration. A model derived from
    linear consensus by swapping in a real edge energy keeps
    zero_pairwise_energy=True unless it says otherwise, and g_i is then
    computed without its edge terms -- silently, with no signature
    mismatch, in both F5's fallback path and every part of F6.

    PER (EDGE, STATE DOMAIN), not once per model. The claim is about every
    edge, and a model whose pairwise terms vanish on some edges and not
    others -- a weight that happens to be zero, a lookup keyed on the
    pair -- would otherwise be blessed by the first edge that agrees.
    That is the same escape a single blessed island opened for the closed
    form. d and dtype join it for the reasons given at the key below. The
    cost is paid once per combination: |E| edges per topology at the one
    or two dimensions a run uses, against the millions of skipped calls
    the declaration buys.

    CHEAPER THAN THE CLOSED FORM'S CHECK, deliberately. There is no
    matrix to reconstruct here, only a claim that two functions return
    zero, so the check evaluates them and compares against zero exactly
    -- no tolerance, because bit-identity of the skip depends on the term
    being exactly zero, and "nearly zero" would not license dropping it.

    WHAT IT PROBES. The real state this call was given, plus the
    deterministic probe set used for the closed form, read as (x_u, x_v)
    pairs. The real state is what the run actually evaluates; the probes
    are what stop a model that happens to vanish on the states a
    particular run drew. As with the closed form, this is not a proof:
    grad_W_ij is an arbitrary function of (x_u, x_v), and no finite probe
    set certifies one. It closes accidental inheritance, which is the
    failure that actually happens.
    """
    try:
        seen = _verified_zero_pairwise.setdefault(model, set())
    except TypeError:
        # Unhashable or non-weak-referenceable model: verify every time
        # rather than crash, exactly as _verify_quadratic_form does.
        seen = None

    for edge in edges:
        u, v = canon(edge)

        # THE KEY IS (edge, domain), where the domain is the (shape,
        # dtype) signature BOTH endpoints share -- taken from the shared
        # helper rather than read off states[u], which is what let the
        # third instance of this bug in. See _StateDomain for why the
        # domain half is factored out and the edge half is not.
        #
        # Dropping the shape component reproduces the closed form's
        # escape exactly: an edge whose pairwise terms vanish at d = 1 is
        # marked verified, the same model and edge are presented at d = 2
        # where they are nonzero, the second check is skipped, and F5
        # returned 11.468133054826271 where the honest value with the
        # declaration withheld is 4.4724463550769995. The probe set
        # itself depends on d, so a d = 1 verification has not looked at
        # cross-coordinate behaviour at all.
        #
        # dtype is KEYED rather than restricted away, which is the
        # OPPOSITE of what F5's fast path does -- and deliberately. There
        # the probes are compared under a 1e-9 tolerance, which sits below
        # float32's own precision, so an honest float32 model would fail
        # on rounding alone and rejection was the only coherent option.
        # This check compares to EXACTLY zero, so there is no rounding
        # regime to get wrong: a function that returns zero returns it in
        # every precision, and one that does not is caught in whichever
        # precision it is called in -- PROVIDED it is nonzero on the real
        # state or one of the finite probes, which is the same limit
        # every probe-based check here carries.
        #
        # The probes are cast to the state's dtype so the callables see
        # what the run will actually hand them. The cast cannot itself
        # manufacture a discrepancy: it only rounds the probe VALUES, and
        # a model whose pairwise terms vanish identically returns zero on
        # every input, rounded or not.
        domain = _state_domain(
            (states[u], states[v]), where=f"edge ({u}, {v})"
        )
        if seen is not None and (u, v, domain) in seen:
            continue

        pairs = [(np.asarray(states[u]), np.asarray(states[v]))]
        pairs += [
            (probe[0].astype(domain.dtype, copy=False),
             probe[1].astype(domain.dtype, copy=False))
            for probe in _quadratic_form_probes(2, domain.shape[0])
        ]

        for a, b in pairs:
            energy = model.W_ij(u, v, a, b)
            grad_u, grad_v = model.grad_W_ij(u, v, a, b)
            if (
                energy != 0.0
                or not np.all(np.asarray(grad_u) == 0.0)
                or not np.all(np.asarray(grad_v) == 0.0)
            ):
                raise ValueError(
                    f"this model declares zero_pairwise_energy=True, but on "
                    f"edge ({u}, {v}) its own callables give "
                    f"W_ij = {energy!r} and grad_W_ij = "
                    f"({grad_u!r}, {grad_v!r}), not zero.\n"
                    "  The declaration lets F5's fallback path and F6 skip "
                    "the within-island gradient loop entirely, so a model "
                    "that declares it wrongly computes g_i without its edge "
                    "terms and returns a different number with no error.\n"
                    "  dataclasses.replace INHERITS the declaration: a "
                    "model derived by swapping W_ij or grad_W_ij must pass "
                    "zero_pairwise_energy=False. See "
                    "tests/conftest.py's synthetic edge-energy model."
                )

        if seen is not None:
            seen.add((u, v, domain))


def _stability_bracket(
    nodes, edges, neighbors, states, model, k, hinge=True,
    trust_zero_pairwise=True,
):
    """Shared bracket computation for F5 and F6 (spec sections 9 and 14).

    `nodes` is the unit's node set (an island for F5, a reconnected
    component for F6), `edges` its within-unit edge set, `neighbors[i]`
    the set of within-unit neighbours of i, and `states` a dict of
    node -> x_i for this sample. Returns max(bracket, 0.0).
    """
    grad_W = {i: model.grad_W_i(i, states[i]) for i in nodes}
    # Skipped under the model's zero_pairwise_energy declaration: every
    # summand would be a zero vector. See SystemModel for what the
    # declaration asserts and _verify_zero_pairwise for the check;
    # trust_zero_pairwise=False forces the full loop, which is what
    # _verify_quadratic_form needs -- a reference that trusted the same
    # declaration as the closed form would agree with it for the wrong
    # reason.
    if not (trust_zero_pairwise and model.zero_pairwise_energy):
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
    # hinge=False gives the RAW bracket. The verification below needs it:
    # comparing hinged values hides any disagreement where both sides are
    # negative, which the clip flattens to 0.0 == 0.0.
    return max(bracket, 0.0) if hinge else bracket




# Structural keys already verified, per model. A WeakKeyDictionary rather
# than a module-level set of callable tuples: the old form held strong
# references to seven callables per variant, and island_quadratic_form's
# closure retains the weight dictionaries and every cached island matrix,
# so a parameter sweep -- each linear_consensus_model call making fresh
# function objects -- grew memory with every construction. Keyed by the
# model, the whole entry is collected when the model is.
_verified_quadratic_forms = weakref.WeakKeyDictionary()


# How many dense fractional probes to add. Four is enough that a
# perturbation would have to be simultaneously orthogonal to all of them
# to hide; each costs one bracket evaluation per island.
_DENSE_PROBES = 4

# The float64 nearest 1/phi = 0.6180..., used as a multiplicative stride
# mod 1. It is a RATIONAL number like every IEEE float, so the sequence
# does have a period -- it is a fixed rotation of the unit interval, and
# successive slots are structurally related, not "unrelated". What it
# gives is spread: the low-discrepancy property of the golden-ratio
# rotation keeps consecutive values from clustering, so the dense probes
# sample distinct directions rather than nearly parallel ones. It is
# deterministic, which is the other half of why it is used -- no RNG
# state to depend on.
_PROBE_STRIDE = 0.6180339887498949


def _quadratic_form_probes(size, d):
    """States for checking a claimed quadratic form.

    WHAT THESE DO AND DO NOT ESTABLISH. A state here is a point in
    R^(size x d), so a general symmetric quadratic bracket over it has
    (n d)(n d + 1) / 2 coefficients, where n = size. Basis vectors and
    pairwise sums over all n*d COORDINATE SLOTS determine every one of
    them -- e_s gives the diagonal coefficient at slot s, and e_s + e_t
    gives that pair's diagonal terms plus twice their cross term -- so
    for a bracket already known to be a symmetric quadratic form, this
    set is complete.

    It was NOT complete when the basis and pairwise probes ranged over
    MEMBERS at coordinate 0 only. That set pins the n(n+1)/2 coefficients
    of a coordinate-DECOUPLED form, which is what the closed form
    asserts, but a replaced callable need not produce one. A singleton
    island at d = 4 has 10 general coefficients and received 6 probes;
    the nullspace is nonzero, and a purely LINEAR replacement -- bracket
    still quadratic, so not the nonlinear limitation below -- was found
    in it that agreed with the closed form on every probe to 6.2e-15
    while the two disagreed by 8.888310483792408 on an ordinary state.
    Probing all n*d slots removes that nullspace.

    What remains open is arbitrary callable replacement, which is what
    can make the true bracket non-quadratic in the first place: a
    perturbation proportional to sum_c x_c^2 (x_c - 1)^2, with its
    gradient, vanishes whenever every coordinate is 0 or 1, so it passes
    every unit probe while leaving the closed form wrong on ordinary
    fractional states.

    So the unit probes are supplemented rather than relied on:

      * basis and pairwise sums over all n*d slots at unit amplitude --
        pin every coefficient of a general symmetric quadratic bracket;
      * one all-coordinates state per member -- redundant for a genuine
        quadratic form, but not for a nonlinear one: it puts every
        coordinate of one member simultaneously at unit amplitude, which
        no basis or pairwise probe does, and a nonlinear coupling can be
        constructed that hides from all of those and from the dense
        probes yet fires here (see the test of that name);
      * dense deterministic vectors with fractional, mixed-sign entries in
        every (member, coordinate) slot -- these are where a perturbation
        that vanishes on {0, 1} coordinates shows up.

    That is a strengthening, not a completeness proof: no finite set of
    probes can certify an arbitrary nonlinear callable. It closes the
    quadratic case entirely, and the perturbations that unit-amplitude
    probes structurally cannot see.

    WHAT IT COSTS. Probing all slots is O((n d)^2) bracket evaluations
    per structural key, against O(n^2) before -- about d^2 = 4x here, and
    each bracket is itself O(n + |E_r|). Measured, as a share of total
    run time spent inside _verify_quadratic_form: 1.6% -> 4.4% on the
    10-node case, 2.6% -> 8.7% on the 20-node, 5.3% -> 17.8% on the
    30-node.

    The n^3 growth is ANALYTIC, not extrapolated from a measurement:
    O((n d)^2) bracket evaluations, each itself O(n + |E_r|), gives
    O(n^3) at fixed d. The probe counts are likewise derivable from the
    construction rather than observed -- a whole-graph island at n = 50,
    d = 2 takes n*d + C(n*d, 2) + n + 4 = 5104 probes, against
    n + C(n, 2) + n + 4 = 1329 under member-only probing. Both figures
    follow from _quadratic_form_probes and can be checked by calling it.

    The corresponding WALL-CLOCK times, 2.15 s and 0.57 s, are
    historical observations on a 50-node case no longer in this
    repository; they cannot be reproduced from here, and nothing above
    depends on them. The share grows like n^3 against the rest of the
    run, which it did before this change too -- the change multiplies
    it by a constant. If it ever becomes the dominant cost at a size
    larger than the 40 nodes now present, the alternative is to go back
    to member-only probes and narrow the claim to coordinate-decoupled
    brackets; it is not acceptable to keep the wider claim on the
    narrower probe set.

    Only the FIRST call for a structural key pays this. Repeat calls hit
    the registry and cost about 1.2 us: on the 20-node run, 368 keys took
    9.92 s of probing and the other 139,525 calls took 0.17 s in total.

    AMPLITUDE AND TOLERANCE. The probes run at amplitude ~1 and the
    caller applies a ~1e-9 tolerance. Quadratic error scales with
    ||x||^2, so a coefficient error just under the threshold here can
    exceed it on a larger state: the effective bar on real samples is
    looser than 1e-9 by the square of the sample radius. Samples are drawn
    on ||x|| <= sqrt(2 (1 + inflation) beta), which is about 1.41 at the
    configs used here, so that factor is roughly 2 -- immaterial. It would
    stop being immaterial for a much larger beta, and the probe amplitude
    or the tolerance would need to move with it.
    """
    probes = []

    # Over SLOTS (member, coordinate), not members: a general symmetric
    # quadratic form couples slots, and probing only coordinate 0 leaves
    # every cross-coordinate coefficient unconstrained.
    slots = [(a, c) for a in range(size) for c in range(d)]

    for a, c in slots:
        x = np.zeros((size, d))
        x[a, c] = 1.0
        probes.append(x)

    for index, (a, c) in enumerate(slots):
        for b, e in slots[index + 1:]:
            x = np.zeros((size, d))
            x[a, c] = 1.0
            x[b, e] = 1.0
            probes.append(x)

    if d > 1:
        for a in range(size):
            x = np.zeros((size, d))
            x[a, :] = 1.0
            probes.append(x)

    for offset in range(_DENSE_PROBES):
        x = np.empty((size, d))
        for a in range(size):
            for c in range(d):
                slot = (a * d + c) + offset * 7919
                x[a, c] = 2.0 * (((slot + 1) * _PROBE_STRIDE) % 1.0) - 1.0
        probes.append(x)

    return probes


def _verify_quadratic_form(model, island, edges_r, config, domain):
    """Check a model's closed form against its own callables, per island.

    THE HAZARD. `island_quadratic_form` is derived from W_i, W_ij, f_i and
    f_ij, but `dataclasses.replace` INHERITS it when any of those is
    swapped, leaving a form that no longer describes the model. F5 then
    returns a different number with no error and no signature mismatch.

    PER STRUCTURAL KEY, not once per model. An earlier version verified
    only the first island of the first call and then trusted the model
    forever. That let the historical stale form through: checked on the
    singleton island {1}, which has no internal edges, the stale linear
    form and the modified model agree -- and a later call on {1,2,3}
    silently returned 2.6530981659254214 where the truth is
    9.806501205737511. Keying on (island nodes, induced edges, k_1, d) --
    d included, see the comment on `key` below -- means each newly
    encountered island is checked on its own merits.

    WHY PROBES RATHER THAN THE CALL'S SAMPLES. Samples make a blind
    verifier: an all-zero set agrees trivially, a set whose errors cancel
    across samples agrees after summation, and a set landing where both
    raw brackets are negative agrees after the hinge. The probes are
    deterministic and independent of whatever states a run happens to
    hold. See _quadratic_form_probes for what they do and do not
    establish -- they are complete for a symmetric QUADRATIC bracket, not
    a proof against arbitrary nonlinear replacement.

    THE PROBES ARE float64. `f5_stability_penalty` rejects any other
    sample dtype before reaching here, so there is only one to verify
    against -- but the domain's dtype is still part of the key, because a
    key that silently describes only part of what a check covered is the
    failure this file has now hit three times. Keying dtype alone would
    NOT have been enough had the restriction not been there: the probes
    are built as float64 regardless, so a float32 model would still have
    been checked in the wrong precision. _verify_zero_pairwise, which has
    no such restriction, casts its probes to the domain's dtype instead.

    WHY __post_init__ CANNOT DO THIS. `replace()` does call __post_init__,
    including for frozen dataclasses (verified, contrary to a common
    assumption), so a check there would run -- but the callables are keyed
    by REAL node identifiers (`node_weights[i]`), so it has no data to
    probe with until an actual island exists.
    """
    # The domain belongs in the key, and comes from the shared
    # _state_domain helper so that the shape and dtype components cannot
    # be assembled inconsistently here and in _verify_zero_pairwise.
    #
    # d belongs in it: the probe SET depends on it (the
    # all-coordinates probes exist only for d > 1), so a model verified
    # at d = 1 has not been checked for cross-coordinate behaviour at
    # all. Omitting it let a model whose f_i couples coordinates only
    # when d > 1 pass at d = 1 and then run unchecked at d = 2. This
    # project evaluates one model at more than one dimension -- the F6
    # d=1 and d=2 tests do exactly that.
    key = (frozenset(island), frozenset(edges_r), config.k_1, domain)
    try:
        seen = _verified_quadratic_forms.setdefault(model, set())
        if key in seen:
            return
    except TypeError:
        # A model carrying an unhashable or non-weak-referenceable
        # callable cannot be registered. Verify on every call rather than
        # crash: slower, never wrong.
        seen = None

    members, B = model.island_quadratic_form(island, edges_r, config.k_1)

    neighbors = {i: set() for i in island}
    for u, v in edges_r:
        neighbors[u].add(v)
        neighbors[v].add(u)

    for probe in _quadratic_form_probes(len(members), domain.shape[0]):
        closed_form = float(np.einsum("ic,ij,jc->", probe, B, probe))
        states = {node: probe[index] for index, node in enumerate(members)}
        # RAW, and one probe at a time: summing first would let positive
        # and negative discrepancies cancel, and hinging first would hide
        # any disagreement below zero entirely.
        from_loop = _stability_bracket(
            island, edges_r, neighbors, states, model, config.k_1,
            hinge=False, trust_zero_pairwise=False,
        )

        # Explicitly, before the tolerance: abs(nan - nan) > tol is False,
        # so two NaNs -- or two identical infinities -- would otherwise be
        # read as agreement and mark the island verified.
        finite = math.isfinite(closed_form) and math.isfinite(from_loop)
        if not finite or (
            abs(closed_form - from_loop) > 1e-9 + 1e-9 * abs(from_loop)
        ):
            raise ValueError(
                "island_quadratic_form does not match this model's own "
                f"callables on island {sorted(island)}: the closed form "
                f"gives {closed_form!r} where evaluating "
                f"W_i/W_ij/f_i/f_ij directly gives {from_loop!r}"
                + ("" if finite else " (one or both is not finite)") + ".\n"
                "  The closed form is derived from those callables, but "
                "dataclasses.replace INHERITS it, so a model built by "
                "swapping any of them keeps a form that no longer "
                "describes it.\n"
                "  Pass island_quadratic_form=None when deriving a model "
                "whose energy or dynamics differ (see SystemModel), or "
                "supply a closed form valid for the new model. A model "
                "with a nonzero W_ij must pass zero_pairwise_energy=False "
                "as well -- that field is inherited by the same route and "
                "would drop the edge terms from g_i."
            )

    if seen is not None:
        seen.add(key)


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

    # FAST PATH, when the model can express the bracket as a quadratic
    # form (linear consensus can; see SystemModel.island_quadratic_form).
    # Evaluates every sample for an island in one einsum instead of a
    # Python loop over nodes and edges per sample.
    #
    # NOT bit-identical to the loop below, and cannot be: the matrix
    # product accumulates in a different order and BLAS may block or
    # thread.
    #
    # ACCEPTANCE CRITERION, and its scope. The two paths were required to
    # agree to 1e-12 ABSOLUTE (see tests/test_cost_caches.py, which
    # enforces it with rel=0.0). Measured disagreement on the benchmark
    # cases is at most 1.78e-15 on F5 values of order 30, so the bar has
    # roughly three orders of margin -- and it sits thirteen orders below
    # the epsilon = 0.01 threshold the local search accepts swaps at, so
    # it cannot change which swap is taken.
    #
    # That absolute bar is calibrated to THESE magnitudes. Relative error
    # is what the arithmetic actually bounds (~1e-16 per operation), so on
    # an instance whose F5 ran to 1e6 an absolute 1e-12 would be tighter
    # than float64 can deliver and would have to be revisited. Nothing in
    # the 10- to 40-node cases is near that.
    quadratic_form = model.island_quadratic_form
    if quadratic_form is not None:
        # DTYPE IS PART OF WHAT THE CLOSED FORM CLAIMS. The verification
        # below probes the model's callables in float64. A model whose
        # callables agree with the closed form in float64 and diverge in
        # another precision -- a cast, a lookup keyed on dtype, an
        # accumulation that rounds differently -- would be verified with
        # the wrong probes and then evaluated with the wrong number: a
        # float32 case gave a fast-path 0.0 against a loop value of 1.15.
        #
        # Rejected rather than verified per dtype, because verifying in
        # float32 is not merely more code: the 1e-9 tolerance below is
        # roughly float64 epsilon times a small factor, and float32 holds
        # about 1e-7 relative precision, so honest float32 evaluation
        # would fail the check on rounding alone. Every sampler in this
        # codebase produces float64 (utils.sampling draws from
        # np.random.Generator.normal), so this rejects nothing that is
        # actually generated here.
        if samples.dtype != np.float64:
            raise ValueError(
                f"F5's fast path requires float64 samples, got "
                f"{samples.dtype}. The closed form is checked against "
                "this model's own callables using float64 probes, so a "
                "model that agrees in float64 but differs in "
                f"{samples.dtype} would pass verification and then be "
                "evaluated wrongly. Verifying in a lower precision is "
                "not a workaround: the check's 1e-9 tolerance is below "
                "what float32 can deliver. Cast the samples to float64, "
                "or clear island_quadratic_form to use the direct loop, "
                "which evaluates the callables in whatever precision it "
                "is given."
            )

        # Once per call, not once per island: samples is one array, so
        # every island reads the same domain out of it.
        domain = _state_domain(samples[0], where="F5 samples")

        total = 0.0
        for island in stable_islands:
            edges_r = _induced_edges(island, topology)
            # Per island, not once per model: a first island that happens
            # to agree must not bless the rest of this call, let alone
            # every later one. See _verify_quadratic_form.
            _verify_quadratic_form(model, island, edges_r, config, domain)
            members, B = quadratic_form(island, edges_r, config.k_1)
            rows = [node_index[node] for node in members]
            X = samples[:, rows, :]                     # (P, n_r, d)
            brackets = np.einsum("pic,ij,pjc->p", X, B, X)
            total += float(np.maximum(brackets, 0.0).sum())
        return config.alpha_1 / config.P_1 * total

    # OUTSIDE the sample loop. _stability_bracket runs per sample per
    # island, so a check inside it would be paid millions of times to
    # protect a claim that cannot change between samples. Here it costs
    # one pass over the islands, on the first sample's real states.
    if model.zero_pairwise_energy and len(samples):
        first = {i: samples[0][node_index[i]] for i in nodes}
        for island in stable_islands:
            _verify_zero_pairwise(model, _induced_edges(island, topology), first)

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


def _psi_reconnection(nodes, edges_r, neighbors, rec_neighbors, states, model, config):
    """Psi_r for one stable island and one sample (spec section 14).

    Sums the four parts of Psi_r and returns the raw (unhinged) value; the
    caller applies the per-island hinge.

    `nodes` is the island's node set V_r, `edges_r` its within-island edge
    set E_r, `neighbors[i]` the within-island neighbours of i,
    `rec_neighbors[i]` the reconnected neighbours N_i^rec (which lie in
    OTHER islands), and `states` maps every topology node -> x_i for this
    sample -- not just the island's, since Parts 1-3 read the far endpoint
    of each reconnected edge.

    Psi_r does NOT fit _stability_bracket's grad.dyn + k*W shape, so this
    is a separate routine and F5's bracket stays untouched (spec section
    14, "Relation to F5 and the shared bracket"). g_i and W_r below
    duplicate the bracket's first two blocks deliberately, rather than
    refactoring them out of a function F5 depends on.

    All four parts are computed unconditionally. On models with W_ij == 0
    (e.g. linear consensus) Parts 2 and 3 come out zero on their own --
    that is arithmetic, not a special case to branch on. The one place
    such a model IS branched on is the g_i loop below, which adds only
    zero vectors; that is skipped, under a declaration the caller has
    verified, because it is pure cost rather than an arithmetic result.
    """
    # g_i -- within-island gradient, identical to F5's dW_r/dx_i.
    g = {i: model.grad_W_i(i, states[i]) for i in nodes}
    # Skipped under zero_pairwise_energy, for the same reason and with the
    # same declaration as _stability_bracket's copy of this loop: every
    # summand is a zero vector. This was F6's single largest cost --
    # 30.0 million grad_W_ij calls, 17.8 s on the 20-node benchmark, all
    # of it adding zeros.
    #
    # Whether the sum vanishes is a property of the MODEL, not of the
    # state, so it cannot be read off the returned values without first
    # computing them -- which is the work being avoided. Hence a
    # declaration, checked once per edge by _verify_zero_pairwise from
    # the callers that own the island loop.
    #
    # NOT bit-identical in one corner, stated rather than glossed:
    # `g[u] + zeros` flattens a -0.0 component to +0.0, and skipping it
    # preserves the -0.0. Every consumer is np.dot into an accumulator
    # that starts at +0.0, where +0.0 and -0.0 contribute identically, so
    # no reachable value changes -- and the 10-, 20- and 30-node runs were
    # compared with float.hex() to confirm it.
    if not model.zero_pairwise_energy:
        for u, v in edges_r:
            grad_u, grad_v = model.grad_W_ij(u, v, states[u], states[v])
            g[u] = g[u] + grad_u
            g[v] = g[v] + grad_v

    # W_r -- within-island energy, identical to F5's.
    W_r = sum(model.W_i(i, states[i]) for i in nodes)
    W_r += sum(model.W_ij(u, v, states[u], states[v]) for u, v in edges_r)

    part_1 = 0.0
    part_2 = 0.0
    part_3 = 0.0
    for i in nodes:
        x_i = states[i]
        rec_i = rec_neighbors.get(i, ())

        # Reconnection dynamics at i: sum over N_i^rec only.
        #
        # The accumulator starts at zero whatever the node, so it is the
        # SHARED read-only zero rather than a fresh allocation: on the
        # 20-node benchmark np.zeros_like ran 16.8 million times here for
        # 13.9 s, about 4.5 per call. Every node still calls _shared_zero
        # -- Part 1 needs the array whether or not rec_i is empty -- but
        # no node makes a FRESH per-node allocation: the first call for a
        # given (shape, dtype) fills the cache and every later one is a
        # dict lookup.
        #
        # SAFE ONLY BECAUSE THIS LOOP REBINDS. `rec_flow = rec_flow +
        # term` builds a new array and leaves the shared zero untouched;
        # `rec_flow += term` would write into it in place and corrupt
        # every later caller. The array is writeable=False, so that
        # mistake raises rather than silently spreading -- but the reason
        # this substitution is legitimate at all is the rebinding form
        # above, not the guard.
        rec_flow = _shared_zero(x_i)
        for k in rec_i:
            rec_flow = rec_flow + model.f_ij(i, k, x_i, states[k])

        part_1 += float(np.dot(g[i], rec_flow))

        # SHORT-CIRCUIT. dyn_i feeds Part 2 and nothing else, and Part 2's
        # inner sum runs over N_i^rec (spec section 14) -- so for a node
        # with no reconnected neighbours it is computed and then thrown
        # away. E^rec is typically 0-3 edges on these benchmarks and often
        # empty, so that was most of F6's f_ij calls.
        #
        # Skipping DISCARDED work only, so results are bit-identical
        # rather than merely close: no floating-point sum loses or gains a
        # term. Note what is deliberately NOT skipped --
        #
        #   part_1 += float(np.dot(g[i], rec_flow)) above stays outside
        #   the guard. With rec_i empty it adds exactly 0.0, so hoisting
        #   it in would look free; but `x + 0.0` is not `x` when x is
        #   -0.0, and np.dot(g_i, zeros) is NaN rather than 0.0 if g_i
        #   ever held a NaN. Leaving it unconditional keeps bit-identity
        #   by construction instead of by argument.
        #
        #   Part 4 is untouched: it is -(k_1 - k_2) * W_r, involves no
        #   N_i^rec, and is computed from W_r outside this loop.
        #
        # ASSUMPTION: the model callables are PURE. Bit-identity here means
        # "the surviving arithmetic is unchanged", which is only the whole
        # story if a call that no longer happens had no other effect. A
        # stateful f_i or f_ij -- one counting invocations, memoising
        # against a mutable store, or logging -- would observe the reduced
        # call count, and a call that previously raised would no longer do
        # so. Purity is the natural reading of the SystemModel contract and
        # every model in this repository satisfies it, but nothing enforces
        # it, so a future model with side effects would need this revisited.
        # tests/test_psi_short_circuit.py relies on exactly that
        # observability -- deliberately, with a recording model -- to prove
        # the guard is present.
        if rec_i:
            # Full dynamics at i: node term plus couplings over
            # (N_i cap V_r) union N_i^rec. The two neighbour sets are
            # disjoint (a reconnected edge joins distinct islands), so
            # adding rec_flow to the within-island sum needs no dedup.
            dyn_i = model.f_i(i, x_i)
            for k in neighbors[i]:
                dyn_i = dyn_i + model.f_ij(i, k, x_i, states[k])
            dyn_i = dyn_i + rec_flow

            for j in rec_i:
                grad_ij_i, _ = model.grad_W_ij(i, j, x_i, states[j])
                part_2 += float(np.dot(grad_ij_i, dyn_i))
                part_3 += model.W_ij(i, j, x_i, states[j])

    part_3 *= config.k_2 / 2.0
    part_4 = -(config.k_1 - config.k_2) * W_r

    return part_1 + part_2 + part_3 + part_4


def f6_reconnection_stability(
    stable_islands, reconnected_edges, topology, model, samples, config
):
    """Reconnection stability-penalty term F6 (spec section 14, revised).

    Sums over stable islands r in S -- NOT over reconnected components.
    The revised theory derives component stability from a per-island
    condition Psi_r, so no quantity here refers to a component.

    `reconnected_edges` is the E^rec set of canonical (min, max) tuples
    (decode_reconnection's third return value); it supplies N_i^rec.
    `topology` is used for BOTH within-island edge derivation (via
    _induced_edges, exactly as F5 does) and the global node ordering
    `sorted(topology.nodes())` along the samples' middle axis.

    The hinge is applied PER ISLAND, before summing over islands: islands
    with Psi_r < 0 clip to zero rather than offsetting a positive one.
    """
    if len(samples) != config.P_2:
        raise ValueError(
            f"len(samples) ({len(samples)}) != config.P_2 ({config.P_2}); "
            "F6 would silently average by the wrong denominator."
        )

    nodes_all = sorted(topology.nodes())
    node_index = {node: idx for idx, node in enumerate(nodes_all)}

    # N_i^rec, read straight off E^rec. No intersection with N_i is needed:
    # E^rec is a subset of E, so every reconnected edge is already an edge
    # of the original graph.
    rec_neighbors = {}
    for edge in reconnected_edges:
        u, v = canon(edge)
        rec_neighbors.setdefault(u, set()).add(v)
        rec_neighbors.setdefault(v, set()).add(u)

    # OUTSIDE the sample loop, as in F5's fallback path: the declaration
    # is a property of the model, so one pass over the islands on the
    # first sample's real states is the whole check.
    if model.zero_pairwise_energy and len(samples):
        first = {node: samples[0][node_index[node]] for node in nodes_all}
        for island in stable_islands:
            _verify_zero_pairwise(model, _induced_edges(island, topology), first)

    total = 0.0
    for sample in samples:
        # Built over the full node set, not per island: a reconnected
        # neighbour of i lies in a different island.
        states = {node: sample[node_index[node]] for node in nodes_all}

        for island in stable_islands:
            edges_r = _induced_edges(island, topology)

            neighbors = {i: set() for i in island}
            for u, v in edges_r:
                neighbors[u].add(v)
                neighbors[v].add(u)

            psi = _psi_reconnection(
                island, edges_r, neighbors, rec_neighbors, states, model, config
            )
            total += max(psi, 0.0)

    return config.alpha_2 / config.P_2 * total
