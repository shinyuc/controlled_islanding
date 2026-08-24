"""Regression tests for the caches and the vectorised F5 fast path.

All three are performance work, but they do NOT share an acceptance bar.

The two CACHES must be bit-identical: they only avoid rebuilding objects,
so any difference in the last bits would mean something was actually
reordered or corrupted. They are:

  - `linear_consensus_model.grad_W_ij` returns shared read-only zero
    arrays instead of allocating a pair per call (W_ij is identically zero
    for that model);
  - `_induced_edges` memoises an island's within-island edge tuple on the
    Topology instance.

Both trade freshly-built objects for shared ones, so both can go wrong in
a way that ordinary tests would not see: a mutated shared array corrupts
every later caller, and a reordered edge sequence changes floating-point
summation order and therefore the last bits of the objective. Nothing
covered either until these.

HOW BIT-IDENTITY WAS VERIFIED WHEN THE CACHES LANDED, so the procedure is
reproducible: the 20-node case was run before and after, and the two saved
JSON results were compared field by field, recursively, using repr() on
every leaf. Two fields were EXCLUDED because they differ between any two
runs of identical code:

    timestamp          -- wall-clock time of the run
    runtime_seconds    -- 329.863 s before, 246.018 s after

Everything else matched exactly, including the full-precision per-pass
objectives in pass_history (8.679957181448122, 14.803490849992968, ...)
and the islands' lambda_min. The terminal output was diffed too and was
byte-identical once the runtime and saved-path lines were filtered out.

The float.hex() test below is the automated form of that check: it
compares an objective computed with the caches active against one from a
reference implementation with both caches defeated, so a single-bit
difference fails rather than passing an approximate compare.

The VECTORISED F5 FAST PATH is held to a weaker bar, deliberately. It
replaces a per-sample Python loop with one einsum over all samples, and a
matrix product accumulates in a different order than sequential Python
addition -- BLAS may also block or thread. Agreement is ~1e-15, so those
tests assert 1e-12 rather than equality. That tolerance is thirteen orders
of magnitude below the search's epsilon = 0.01 acceptance threshold, so it
cannot change which swap the local search takes; the acceptance check for
that change was that the returned PARTITION is identical on the 10-, 20-
and 30-node cases, with cost terms agreeing to 1e-12.
"""

from dataclasses import replace

import numpy as np
import pytest

import islanding.cost as cost_module
from islanding.augmented import augmented_graph
from islanding.cost import (
    AlgorithmConfig, linear_consensus_model, canon, _induced_edges,
)
from islanding.evaluate import evaluate
from islanding.search import initial_spanning_tree
from islanding.topology import Topology
from examples.linear_10node.linear_10node import (
    NODE_AND_EDGE_FAULTS, build_topology, build_problem,
)
from examples.linear_10node.config import build_config


# ---------------------------------------------------------------------------
# The shared zero gradient
# ---------------------------------------------------------------------------

def _model():
    return linear_consensus_model({1: 1.0, 2: 1.0}, {(1, 2): 1.0})


def test_the_zero_gradient_is_shared_across_calls():
    model = _model()
    x = np.zeros(3)

    first_i, first_j = model.grad_W_ij(1, 2, x, x)
    second_i, second_j = model.grad_W_ij(1, 2, x, x)

    assert first_i is second_i
    assert first_j is second_j
    # ...and it is still the right value.
    np.testing.assert_array_equal(first_i, np.zeros_like(x))


def test_the_shared_zero_gradient_rejects_in_place_writes():
    # THE HAZARD SHARING CREATES. Without read-only, a single `grad += ...`
    # anywhere would silently rewrite the zero vector every other caller
    # is about to read -- corrupting F5, F6 and every MILP certificate
    # constant, with no error and no obviously wrong output.
    model = _model()
    gradient, _ = model.grad_W_ij(1, 2, np.zeros(3), np.zeros(3))

    assert not gradient.flags.writeable
    with pytest.raises(ValueError):
        gradient[0] = 1.0
    with pytest.raises(ValueError):
        gradient += 1.0

    np.testing.assert_array_equal(gradient, np.zeros(3))


def test_each_state_dimension_gets_its_own_zero_gradient():
    # A single cached array per model would return a wrongly-shaped zero
    # once the same model is evaluated at another d -- which the d=1 and
    # d=2 F6 tests do.
    model = _model()

    one_d, _ = model.grad_W_ij(1, 2, np.zeros(1), np.zeros(1))
    two_d, _ = model.grad_W_ij(1, 2, np.zeros(2), np.zeros(2))

    assert one_d.shape == (1,)
    assert two_d.shape == (2,)
    assert one_d is not two_d


# ---------------------------------------------------------------------------
# The induced-edge cache
# ---------------------------------------------------------------------------

def _topology():
    return Topology(
        nodes=[1, 2, 3, 4],
        edges=[(1, 2), (2, 3), (1, 3), (3, 4)],
    )


def test_induced_edges_returns_the_same_order_every_call():
    # Iteration order feeds floating-point summation order in
    # _stability_bracket and _psi_reconnection, so a cache that returned
    # an equal-but-reordered sequence would change the last bits of the
    # objective while every equality test still passed.
    topology = _topology()
    island = frozenset({1, 2, 3})

    first = _induced_edges(island, topology)
    repeated = [_induced_edges(island, topology) for _ in range(5)]

    for later in repeated:
        assert later == first          # same order, not just same members
        assert list(later) == list(first)


def test_induced_edges_matches_the_uncached_computation():
    topology = _topology()
    island = frozenset({1, 2, 3})

    cached = _induced_edges(island, topology)
    fresh = tuple({
        canon((u, v))
        for u, v in topology.edges()
        if u in island and v in island
    })

    assert set(cached) == set(fresh) == {(1, 2), (1, 3), (2, 3)}
    # Order too: this is what bit-identity rests on.
    assert cached == fresh


def test_a_caller_cannot_corrupt_the_cached_edges():
    # The reason the return type is a tuple. As a shared mutable set, one
    # stray .pop() would leave that island permanently missing an edge --
    # silently changing F3, F5 and F6 for the rest of the process.
    topology = _topology()
    island = frozenset({1, 2, 3})

    edges = _induced_edges(island, topology)

    assert isinstance(edges, tuple)
    with pytest.raises(AttributeError):
        edges.pop()
    with pytest.raises(TypeError):
        edges[0] = ("x", "y")

    assert _induced_edges(island, topology) == edges


def test_the_cache_does_not_leak_between_topologies():
    # Same node labels, DIFFERENT edges. A module-level cache keyed by
    # id(topology) would collide here once an id were reused after garbage
    # collection, serving one graph's edges for another.
    triangle = Topology(nodes=[1, 2, 3], edges=[(1, 2), (2, 3), (1, 3)])
    path = Topology(nodes=[1, 2, 3], edges=[(1, 2), (2, 3)])
    island = frozenset({1, 2, 3})

    assert set(_induced_edges(island, triangle)) == {(1, 2), (2, 3), (1, 3)}
    assert set(_induced_edges(island, path)) == {(1, 2), (2, 3)}


def test_the_cache_lives_on_the_topology_and_is_private():
    topology = _topology()
    assert topology._induced_edge_cache == {}

    _induced_edges(frozenset({1, 2, 3}), topology)

    assert frozenset({1, 2, 3}) in topology._induced_edge_cache
    # A fresh topology starts empty -- the cache is per-instance, so its
    # lifetime matches the topology's exactly.
    assert _topology()._induced_edge_cache == {}


# ---------------------------------------------------------------------------
# The strongest check: cached vs uncached, to the last bit
# ---------------------------------------------------------------------------

def _uncached_induced_edges(nodes, topology):
    """_induced_edges as it was before the cache: rebuilt every call."""
    return tuple({
        canon((u, v))
        for u, v in topology.edges()
        if u in nodes and v in nodes
    })


def test_the_objective_is_bit_identical_with_and_without_the_caches(monkeypatch):
    # The automated form of the before/after run recorded in this module's
    # docstring. Compared with float.hex(), not == : a last-bit difference
    # would slip through pytest.approx and would even survive a plain ==
    # print, but shows immediately as a different hex mantissa.
    topology = build_topology()
    problem = build_problem(
        NODE_AND_EDGE_FAULTS, topology, seed=0, config=build_config()
    )
    tree = initial_spanning_tree(augmented_graph(topology))

    with_caches = evaluate(tree, problem)

    # Defeat BOTH caches: a fresh pair of zero arrays per grad_W_ij call,
    # and a rebuilt edge tuple per _induced_edges call, on a fresh
    # topology so nothing is memoised from the run above.
    def fresh_zero_gradients(i, j, x_i, x_j):
        return (np.zeros_like(x_i), np.zeros_like(x_j))

    reference_topology = build_topology()
    reference = build_problem(
        NODE_AND_EDGE_FAULTS, reference_topology, seed=0, config=build_config()
    )
    reference = replace(
        reference, model=replace(reference.model, grad_W_ij=fresh_zero_gradients)
    )
    monkeypatch.setattr(cost_module, "_induced_edges", _uncached_induced_edges)
    reference_tree = initial_spanning_tree(augmented_graph(reference_topology))

    without_caches = evaluate(reference_tree, reference)

    assert float.hex(with_caches) == float.hex(without_caches), (
        f"caching changed the objective: {with_caches!r} "
        f"({float.hex(with_caches)}) vs {without_caches!r} "
        f"({float.hex(without_caches)})"
    )
    # The comparison is only meaningful on a non-trivial value: an
    # objective of 0.0 would match for reasons unrelated to the caches.
    assert with_caches != 0.0


# ---------------------------------------------------------------------------
# The vectorised F5 fast path
# ---------------------------------------------------------------------------

def _f5_via_loop(stable_islands, topology, model, samples, config):
    """F5 with the fast path disabled -- the original per-sample loop."""
    return cost_module.f5_stability_penalty(
        stable_islands, topology,
        replace(model, island_quadratic_form=None),
        samples, config,
    )


def _twenty_node():
    from examples.linear_20node.builder import (
        LINEAR_20, build_topology as topo20, build_problem as problem20,
    )
    from examples.linear_20node.config import build_config as config20

    topology = topo20()
    return topology, problem20(LINEAR_20, topology, seed=0, config=config20())


@pytest.mark.parametrize("size", [1, 5, 9, 20], ids=lambda n: f"island_of_{n}")
def test_the_vectorised_f5_matches_the_loop(size):
    # ABSOLUTE 1e-12 -- rel=0.0 is deliberate. pytest.approx takes the
    # LARGER of its two bounds, so the previous abs=1e-12, rel=1e-12 quietly
    # permitted ~1e-10 of absolute error on a result of magnitude 100. The
    # criterion this change was accepted against is absolute agreement to
    # 1e-12, so the test now enforces exactly that.
    #
    # Not exact equality, because the einsum accumulates in a different
    # order than the Python loop and BLAS may block or thread. Measured
    # error is <= 1.8e-15 absolute across these islands (values up to 30),
    # so 1e-12 leaves three orders of margin while staying thirteen orders
    # below the search's epsilon = 0.01. If F5 ever reached ~1e6 an
    # absolute bar would become tighter than float64 can deliver and would
    # need revisiting; nothing here is near that.
    topology, problem = _twenty_node()
    nodes = sorted(topology.nodes())
    islands = [frozenset(nodes[:size])]

    fast = cost_module.f5_stability_penalty(
        islands, topology, problem.model, problem.samples_1, problem.config
    )
    loop = _f5_via_loop(
        islands, topology, problem.model, problem.samples_1, problem.config
    )

    assert fast == pytest.approx(loop, abs=1e-12, rel=0.0)


def test_the_vectorised_f5_matches_the_loop_on_a_multi_island_partition():
    topology, problem = _twenty_node()
    nodes = sorted(topology.nodes())
    islands = [
        frozenset(nodes[:7]), frozenset(nodes[7:13]), frozenset(nodes[13:]),
    ]

    fast = cost_module.f5_stability_penalty(
        islands, topology, problem.model, problem.samples_1, problem.config
    )
    loop = _f5_via_loop(
        islands, topology, problem.model, problem.samples_1, problem.config
    )

    assert fast > 0, "a zero penalty would make this comparison vacuous"
    assert fast == pytest.approx(loop, abs=1e-12, rel=0.0)


def test_the_vectorised_f5_matches_the_loop_for_d_greater_than_one():
    # The coordinates decouple because the weights are scalars, so the
    # bracket is trace(X.T B X) over the (n_r, d) state. d = 2 is what the
    # 20-node case actually uses.
    topology, problem = _twenty_node()
    assert problem.samples_1.shape[2] == 2
    islands = [frozenset(sorted(topology.nodes())[:9])]

    fast = cost_module.f5_stability_penalty(
        islands, topology, problem.model, problem.samples_1, problem.config
    )
    loop = _f5_via_loop(
        islands, topology, problem.model, problem.samples_1, problem.config
    )

    assert fast == pytest.approx(loop, abs=1e-12, rel=0.0)


def test_a_model_without_a_quadratic_form_uses_the_loop():
    # The fast path is OPTIONAL. A model that cannot express the bracket
    # as a quadratic form must fall through to the loop, unchanged.
    topology, problem = _twenty_node()
    islands = [frozenset(sorted(topology.nodes())[:9])]
    plain = replace(problem.model, island_quadratic_form=None)

    assert plain.island_quadratic_form is None
    value = cost_module.f5_stability_penalty(
        islands, topology, plain, problem.samples_1, problem.config
    )

    assert value == pytest.approx(
        cost_module.f5_stability_penalty(
            islands, topology, problem.model, problem.samples_1, problem.config
        ),
        abs=1e-12, rel=0.0,
    )


def test_the_synthetic_edge_energy_model_has_no_quadratic_form(
    synthetic_edge_energy_model
):
    # THE TRAP. dataclasses.replace inherits island_quadratic_form, so a
    # model derived by swapping W_ij/grad_W_ij keeps a closed form that
    # assumed W_ij == 0. Left set, F5 returned 82.58 on the 10-node
    # fixture where the true value is 30.16 -- no error, no type
    # mismatch, only the MILP-vs-cost equality in tests/test_baseline.py
    # noticed.
    base = linear_consensus_model({1: 1.0, 2: 1.0}, {(1, 2): 1.0})

    assert base.island_quadratic_form is not None
    assert synthetic_edge_energy_model(base).island_quadratic_form is None


def test_the_quadratic_form_is_cached_per_island_and_edge_set():
    # Cached on the MODEL, not the topology: B is built from the weights,
    # which a Topology does not know. Keyed by the edge set too, since one
    # model can be evaluated against more than one topology.
    model = linear_consensus_model(
        {1: 1.0, 2: 2.0, 3: 3.0}, {(1, 2): 4.0, (2, 3): 5.0, (1, 3): 6.0}
    )
    island = frozenset({1, 2, 3})
    triangle = ((1, 2), (2, 3), (1, 3))
    path = ((1, 2), (2, 3))

    first = model.island_quadratic_form(island, triangle, 0.3)
    assert model.island_quadratic_form(island, triangle, 0.3) is first

    # A different induced edge set must NOT hit the same entry.
    other = model.island_quadratic_form(island, path, 0.3)
    assert other is not first
    assert not np.array_equal(first[1], other[1])

    # ...nor a different decay rate.
    assert model.island_quadratic_form(island, triangle, 0.5) is not first


def test_both_paths_are_nonzero_on_a_penalised_island():
    # Several of the comparisons above land on islands where F5 is 0 on
    # BOTH paths -- the hinge clips everything -- so they would pass even
    # if the two computed wildly different negative brackets. This one
    # pins a case where both values are genuinely positive, so the
    # agreement is about the arithmetic rather than about the clip.
    topology, problem = _twenty_node()
    islands = [frozenset(sorted(topology.nodes())[:5])]

    fast = cost_module.f5_stability_penalty(
        islands, topology, problem.model, problem.samples_1, problem.config
    )
    loop = _f5_via_loop(
        islands, topology, problem.model, problem.samples_1, problem.config
    )

    assert fast > 1.0
    assert loop > 1.0
    assert fast == pytest.approx(loop, abs=1e-12, rel=0.0)


@pytest.mark.parametrize(
    "offset",
    [-1e-9, -1e-13, 0.0, 1e-13, 1e-9],
    ids=["clearly_positive", "barely_positive", "exactly_zero",
         "barely_negative", "clearly_negative"],
)
def test_the_two_paths_agree_where_the_hinge_could_flip(offset):
    # A single-node island has bracket = (k/2 - w_1) * ||x||^2, so setting
    # w_1 = k/2 + offset puts the bracket exactly `offset * ||x||^2` from
    # the hinge's knee. Straddling it checks the region where a ~1e-15
    # disagreement between the two paths could land them on opposite sides
    # of max(., 0) -- the one place a tiny arithmetic difference could
    # produce a qualitatively different answer.
    #
    # It cannot produce a LARGE one: near the knee the bracket itself is
    # tiny, so a flipped clip changes the result by at most that tiny
    # amount. The assertions are the agreement and the hinge's floor.
    config = AlgorithmConfig(
        kappa_op=1.0, kappa_rec=0.7, k_1=0.3, P_1=1, k_2=0.1, P_2=1
    )
    weight = config.k_1 / 2.0 + offset
    model = linear_consensus_model({1: weight}, {})
    topology = Topology(nodes=[1], edges=[])
    samples = np.array([[[2.0]]])
    islands = [frozenset({1})]

    fast = cost_module.f5_stability_penalty(
        islands, topology, model, samples, config
    )
    loop = cost_module.f5_stability_penalty(
        islands, topology, replace(model, island_quadratic_form=None),
        samples, config,
    )

    assert fast == pytest.approx(loop, abs=1e-12, rel=0.0)
    assert fast >= 0.0 and loop >= 0.0          # the hinge holds on both

    # The two clear cases must also land on the side the maths says.
    if offset == -1e-9:
        assert fast > 0.0
    if offset == 1e-9:
        assert fast == 0.0


# ---------------------------------------------------------------------------
# The quadratic-form cache cannot go stale or be corrupted
# ---------------------------------------------------------------------------

def test_mutating_the_callers_weights_desynchronises_nothing():
    # linear_consensus_model used to capture the weight dictionaries BY
    # REFERENCE. A later mutation was seen immediately by f_i and f_ij
    # while island_quadratic_form kept returning the matrix built from the
    # old weights, so the two paths silently disagreed: 0.0 from the fast
    # path against 16.25 from the loop. The model now copies both dicts at
    # construction, so a caller's later edits cannot reach either path.
    #
    # A TWO-NODE island with a real (1, 2) edge, so the EDGE-weight copy is
    # exercised too. An earlier version of this test used a single node
    # with no edges and mutated edge_weights[(1, 2)] -- an entry neither
    # path ever read, so it passed with `edge_weights = dict(edge_weights)`
    # deleted.
    # Weights chosen so F5 is genuinely NONZERO (14.7) and so that each
    # mutation below would individually drive it to 0.0 -- see
    # test_the_mutations_would_each_have_mattered.
    node_weights = {1: -1.0, 2: 1.0}
    edge_weights = {(1, 2): 0.1}
    model = linear_consensus_model(node_weights, edge_weights)
    topology = Topology(nodes=[1, 2], edges=[(1, 2)])
    config = AlgorithmConfig(
        kappa_op=1.0, kappa_rec=0.7, k_1=0.3, P_1=1, k_2=0.1, P_2=1
    )
    # Asymmetric state, so the edge term contributes: a uniform state makes
    # (x_i - x_j) vanish and the edge weight would drop out regardless.
    samples = np.array([[[5.0], [-3.0]]])
    islands = [frozenset({1, 2})]

    def both_paths():
        return (
            cost_module.f5_stability_penalty(
                islands, topology, model, samples, config
            ),
            cost_module.f5_stability_penalty(
                islands, topology, replace(model, island_quadratic_form=None),
                samples, config,
            ),
        )

    before_fast, before_loop = both_paths()
    assert before_fast == pytest.approx(before_loop, abs=1e-12, rel=0.0)
    assert before_fast > 1.0, "a zero penalty would make this vacuous"

    node_weights[1] = 0.5           # would make the island stable
    edge_weights[(1, 2)] = 99.0     # would dominate the bracket

    after_fast, after_loop = both_paths()

    # Neither path moved, and they still agree with each other.
    assert after_fast == before_fast
    assert after_loop == before_loop
    assert after_fast == pytest.approx(after_loop, abs=1e-12, rel=0.0)


def test_the_mutations_would_each_have_mattered():
    # Guards the test above against going vacuous, one dictionary at a
    # time. An earlier version used a single node with no edges and
    # mutated edge_weights[(1, 2)] -- an entry neither path ever read, so
    # it passed with `edge_weights = dict(edge_weights)` deleted from
    # linear_consensus_model. Each mutation is now shown to change F5 on
    # its own, so "unchanged" in that test is a real assertion.
    topology = Topology(nodes=[1, 2], edges=[(1, 2)])
    config = AlgorithmConfig(
        kappa_op=1.0, kappa_rec=0.7, k_1=0.3, P_1=1, k_2=0.1, P_2=1
    )
    # Asymmetric: a uniform state makes (x_i - x_j) vanish and the edge
    # weight would drop out however large it was.
    samples = np.array([[[5.0], [-3.0]]])
    islands = [frozenset({1, 2})]

    def f5(node_weights, edge_weights):
        return cost_module.f5_stability_penalty(
            islands, topology,
            linear_consensus_model(node_weights, edge_weights),
            samples, config,
        )

    original = f5({1: -1.0, 2: 1.0}, {(1, 2): 0.1})
    assert original == pytest.approx(14.7, abs=1e-12, rel=0.0)

    # The EDGE-weight mutation alone.
    assert f5({1: -1.0, 2: 1.0}, {(1, 2): 99.0}) == 0.0
    # The NODE-weight mutation alone.
    assert f5({1: 0.5, 2: 1.0}, {(1, 2): 0.1}) == 0.0


def test_the_counterexample_also_sees_the_snapshot():
    # linear_consensus_model builds its counterexample from the SAME
    # private dictionaries as f_i, f_ij and the quadratic form, because
    # the copies are taken before any closure is created. Only F5 was
    # covered above, so a regression that passed the caller's originals to
    # linear_counterexample would have gone unnoticed.
    #
    # A single node with no edges has M_u = [w_1], so the stability
    # verdict is just the sign of w_1 against the tolerance. Starting
    # safely positive and mutating to negative crosses that boundary: with
    # the snapshot the unit stays stable and no witness is produced;
    # without it, the mutated weight would make it unstable.
    #
    # NOTE linear_counterexample called DIRECTLY still captures its
    # arguments by reference -- it holds no cache of its own, so there is
    # nothing to desynchronise. The snapshot semantics tested here hold
    # via linear_consensus_model, not of linear_counterexample itself.
    node_weights = {1: 1.0}
    edge_weights = {}
    model = linear_consensus_model(node_weights, edge_weights)

    def witnesses():
        return model.counterexample(
            [1], [], node_order=[1], d=1, threshold_radius=1.0,
            stability_tolerance=1e-9,
        )

    assert witnesses() == []            # w_1 = 1.0 > 0: stable

    node_weights[1] = -1.0              # would make it unstable

    assert witnesses() == [], (
        "the counterexample search saw the caller's mutated weight, so it "
        "was not built from the model's private snapshot"
    )

    # ...and a model genuinely built with the negative weight DOES find
    # one, so the assertion above is not vacuous.
    unstable = linear_consensus_model({1: -1.0}, {})
    assert len(unstable.counterexample(
        [1], [], node_order=[1], d=1, threshold_radius=1.0,
        stability_tolerance=1e-9,
    )) == 1


def test_the_cached_quadratic_form_rejects_mutation():
    # Same hazard already closed for the induced-edge tuples and the
    # shared zero gradients: the pair is handed to every later caller for
    # this key, so a reordered `members` or a written-to `B` would corrupt
    # every subsequent F5 evaluation.
    model = linear_consensus_model({1: 1.0, 2: 2.0}, {(1, 2): 3.0})
    members, B = model.island_quadratic_form(frozenset({1, 2}), ((1, 2),), 0.3)

    assert isinstance(members, tuple)
    with pytest.raises(AttributeError):
        members.reverse()
    with pytest.raises(TypeError):
        members[0] = 99

    assert not B.flags.writeable
    with pytest.raises(ValueError):
        B[0, 0] = 1.0
    with pytest.raises(ValueError):
        B += 1.0

    again = model.island_quadratic_form(frozenset({1, 2}), ((1, 2),), 0.3)
    assert again[0] == members
    np.testing.assert_array_equal(again[1], B)
