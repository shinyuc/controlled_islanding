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

import re
from dataclasses import replace

import weakref

import numpy as np
import pytest

import islanding.cost as cost_module
from islanding.augmented import augmented_graph
from islanding.cost import (
    AlgorithmConfig, linear_consensus_model, canon, _induced_edges,
    _quadratic_form_probes,
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

    # zero_pairwise_energy WITHHELD on both sides. Linear consensus
    # declares it, which makes the g_i loops in _stability_bracket and
    # _psi_reconnection skip grad_W_ij entirely -- and grad_W_ij is where
    # the shared-zero cache lives, so with the declaration in place this
    # test would only reach it through Psi_r's Part 2 across E^rec. The
    # cache still serves every model that does not declare, so the old
    # path is what this test needs to exercise. Withheld on the cached
    # side too, or the two runs would differ in which code path they take
    # rather than only in whether the caches are active.
    problem = replace(
        problem, model=replace(problem.model, zero_pairwise_energy=False)
    )

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
        reference,
        model=replace(
            reference.model, grad_W_ij=fresh_zero_gradients,
            zero_pairwise_energy=False,
        ),
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


# ---------------------------------------------------------------------------
# The dataclasses.replace hazard, closed at runtime
# ---------------------------------------------------------------------------

def _ten_node():
    from examples.linear_10node.linear_10node import (
        NODE_AND_EDGE_FAULTS, build_topology as topo10,
        build_problem as problem10,
    )
    from examples.linear_10node.config import build_config as config10

    topology = topo10()
    return topology, problem10(
        NODE_AND_EDGE_FAULTS, topology, seed=0, config=config10()
    )


def _inherits_the_stale_form(base):
    """The synthetic edge-energy model WITHOUT clearing the closed form.

    Exactly the mistake that shipped once: swapping W_ij and grad_W_ij via
    dataclasses.replace, which inherits island_quadratic_form -- a form
    derived on the assumption W_ij == 0, which this model breaks. It now
    inherits zero_pairwise_energy=True by the same route, which is wrong
    for the same reason; both are left in place here deliberately, since
    this helper's job is to be the un-fixed model.
    """
    def W_ij(i, j, x_i, x_j):
        delta = x_i - x_j
        return 0.5 * float(np.dot(delta, delta))

    def grad_W_ij(i, j, x_i, x_j):
        delta = x_i - x_j
        return (delta, -delta)

    return replace(base, W_ij=W_ij, grad_W_ij=grad_W_ij)


def test_an_inherited_quadratic_form_is_rejected():
    # THE HAZARD. Before this check, F5 returned 82.58 on the 10-node
    # fixture where the true value was 30.16 -- no error, no signature
    # mismatch, and only the MILP-vs-cost equality in tests/test_baseline.py
    # noticed. Kuramoto would hit the same thing directly, its W_ij being
    # nonzero.
    topology, problem = _ten_node()
    stable = [frozenset({1, 2, 3}), frozenset({5, 8, 9}), frozenset({6, 7, 10})]

    with pytest.raises(ValueError, match="does not match this model's own"):
        cost_module.f5_stability_penalty(
            stable, topology, _inherits_the_stale_form(problem.model),
            problem.samples_1, problem.config,
        )


def test_the_rejection_message_says_what_to_do():
    # An error naming only the symptom would leave the reader to rediscover
    # the replace() inheritance for themselves.
    topology, problem = _ten_node()
    stable = [frozenset({1, 2, 3})]

    with pytest.raises(ValueError) as caught:
        cost_module.f5_stability_penalty(
            stable, topology, _inherits_the_stale_form(problem.model),
            problem.samples_1, problem.config,
        )

    message = str(caught.value)
    assert "dataclasses.replace" in message
    assert "island_quadratic_form=None" in message
    # BOTH numeric values must appear, so the size of the discrepancy is
    # visible. `message.count(".") > 2` used to stand in for this, which
    # ordinary punctuation and "dataclasses.replace" satisfy on their own.
    numbers = re.findall(r"-?\d+\.\d+(?:e[+-]\d+)?", message)
    assert len(numbers) >= 2, f"expected two values in: {message}"
    assert float(numbers[0]) != float(numbers[1])


def test_the_check_passes_on_the_unmodified_model():
    topology, problem = _ten_node()
    stable = [frozenset({1, 2, 3}), frozenset({5, 8, 9}), frozenset({6, 7, 10})]

    value = cost_module.f5_stability_penalty(
        stable, topology, problem.model, problem.samples_1, problem.config
    )

    assert np.isfinite(value)


def test_clearing_the_form_is_the_documented_escape():
    # The fix the message prescribes must actually work: with BOTH
    # inherited declarations cleared, the same derived model runs on the
    # loop path. zero_pairwise_energy is inherited by exactly the same
    # route as the closed form and is just as wrong on this model, so the
    # message names both -- clearing only island_quadratic_form leaves a
    # model that skips its own edge gradients.
    topology, problem = _ten_node()
    stable = [frozenset({1, 2, 3})]
    derived = replace(
        _inherits_the_stale_form(problem.model),
        island_quadratic_form=None, zero_pairwise_energy=False,
    )

    value = cost_module.f5_stability_penalty(
        stable, topology, derived, problem.samples_1, problem.config
    )

    assert np.isfinite(value)


@pytest.mark.parametrize("case", ["10node", "20node", "30node", "40node"])
def test_the_check_does_not_fire_on_any_benchmark_case(case):
    # A false positive would break every run, so the tolerance must be
    # loose enough for ordinary floating-point disagreement (~1e-15 on
    # values of order 30) while still catching a genuine mismatch (82.58
    # against 30.16).
    import importlib

    # The 10-node case predates the builder.py convention and keeps its
    # setup in linear_10node.py under a different name.
    module_name = (
        "examples.linear_10node.linear_10node" if case == "10node"
        else f"examples.linear_{case}.builder"
    )
    builder = importlib.import_module(module_name)
    config_module = importlib.import_module(f"examples.linear_{case}.config")
    setup = (
        builder.NODE_AND_EDGE_FAULTS if case == "10node"
        else getattr(builder, f"LINEAR_{case[:2]}")
    )

    topology = builder.build_topology()
    problem = builder.build_problem(
        setup, topology, seed=0, config=config_module.build_config()
    )
    nodes = sorted(topology.nodes())
    half = len(nodes) // 2
    stable = [frozenset(nodes[:half]), frozenset(nodes[half:])]

    value = cost_module.f5_stability_penalty(
        stable, topology, problem.model, problem.samples_1, problem.config
    )

    assert np.isfinite(value)


def test_the_check_runs_once_per_model_variant():
    # Amortised to nothing: the loop-path evaluation happens on first use
    # and never again for that combination of callables.
    topology, problem = _ten_node()
    stable = [frozenset({1, 2, 3})]
    calls = []

    base = problem.model

    def counting_form(nodes, edges, k):
        calls.append(1)
        return base.island_quadratic_form(nodes, edges, k)

    model = replace(base, island_quadratic_form=counting_form)

    for _ in range(5):
        cost_module.f5_stability_penalty(
            stable, topology, model, problem.samples_1, problem.config
        )

    # 5 fast-path calls; the verification adds exactly one extra on the
    # first, not one per call.
    assert len(calls) == 6


# ---------------------------------------------------------------------------
# The escapes a once-per-model check left open
# ---------------------------------------------------------------------------

def _stale(base):
    """A model whose inherited closed form no longer describes it."""
    def W_ij(i, j, x_i, x_j):
        delta = x_i - x_j
        return 0.5 * float(np.dot(delta, delta))

    def grad_W_ij(i, j, x_i, x_j):
        delta = x_i - x_j
        return (delta, -delta)

    return replace(base, W_ij=W_ij, grad_W_ij=grad_W_ij)


def test_a_singleton_island_does_not_bless_the_rest():
    # THE ESCAPE. Verifying only the first island and then trusting the
    # model let this through: island {1} has no internal edges, so the
    # stale form and the modified model genuinely agree there -- correctly
    # not flagged -- and the variant was marked verified. A later call on
    # {1,2,3} then returned 2.6530981659254214 where the truth is
    # 9.806501205737511, silently.
    topology, problem = _ten_node()
    model = _stale(problem.model)

    # Agrees on the singleton, so no error -- a true negative, not a miss.
    cost_module.f5_stability_penalty(
        [frozenset({1})], topology, model, problem.samples_1, problem.config
    )

    # ...and the blessing must not carry to an island that does disagree.
    with pytest.raises(ValueError, match="does not match"):
        cost_module.f5_stability_penalty(
            [frozenset({1, 2, 3})], topology, model,
            problem.samples_1, problem.config,
        )


def test_a_first_island_that_agrees_does_not_bless_the_same_call():
    # The same escape WITHIN one F5 call: stable_islands[0] agreeing must
    # not let the remaining islands of that call through unchecked.
    topology, problem = _ten_node()

    with pytest.raises(ValueError, match="does not match"):
        cost_module.f5_stability_penalty(
            [frozenset({1}), frozenset({1, 2, 3})], topology,
            _stale(problem.model), problem.samples_1, problem.config,
        )


def test_all_zero_samples_do_not_hide_a_mismatch():
    # A sample set of zeros makes both hinged results 0.0, so a
    # sample-driven check agrees trivially. The probes do not depend on
    # the samples at all.
    topology, problem = _ten_node()

    with pytest.raises(ValueError, match="does not match"):
        cost_module.f5_stability_penalty(
            [frozenset({1, 2, 3})], topology, _stale(problem.model),
            np.zeros_like(problem.samples_1), problem.config,
        )


def test_a_mismatch_below_the_hinge_is_still_caught():
    # Comparing hinged values hides any disagreement where both sides are
    # negative: the clip flattens both to 0.0 and they "agree".
    #
    # The singleton island {1} is chosen precisely so EVERY probe stays
    # below the hinge -- node 1 is healthy, so its only probe bracket is
    # B[0,0] = k_1/2 - w_1 = -0.85, and the d=2 mixed probe is twice that.
    # A larger island will not do: nodes 2 and 3 are faulty, so their
    # basis probes come out positive and a hinged comparison would catch
    # the mismatch for the wrong reason.
    topology, problem = _ten_node()
    island = frozenset({1})

    members, B = problem.model.island_quadratic_form(
        island, _induced_edges(island, topology), problem.config.k_1
    )
    assert B[0, 0] < 0, "fixture must sit below the hinge to be meaningful"

    shifted = replace(
        problem.model,
        island_quadratic_form=lambda nodes, edges, k: (
            lambda pair: (pair[0], pair[1] - 1.0)
        )(problem.model.island_quadratic_form(nodes, edges, k)),
    )
    # Still negative, so the hinge clips both sides to 0.0 and only a RAW
    # comparison can see the difference.
    assert B[0, 0] - 1.0 < 0

    with pytest.raises(ValueError, match="does not match"):
        cost_module.f5_stability_penalty(
            [island], topology, shifted, problem.samples_1, problem.config
        )


def test_a_later_k1_is_verified_separately():
    # k_1 is part of the structural key. The earlier version of this test
    # changed k_1 on a singleton and then checked a DIFFERENT island --
    # which was unverified anyway, so it passed with k_1 removed from the
    # key. This uses the SAME island throughout, with a form deliberately
    # correct at one k_1 and wrong at another, so only the key can save it.
    topology, problem = _ten_node()
    island = frozenset({1, 2, 3})
    base = problem.model

    def form(nodes, edges, k):
        members, B = base.island_quadratic_form(nodes, edges, k)
        return (members, B * 2.0) if k > 0.5 else (members, B)

    model = replace(base, island_quadratic_form=form)

    # Correct at k_1 = 0.3: accepted, and the key is recorded.
    cost_module.f5_stability_penalty(
        [island], topology, model, problem.samples_1, problem.config
    )

    # Wrong at k_1 = 0.9, on the very same island. Without k_1 in the key
    # the recorded entry would suppress this check entirely.
    with pytest.raises(ValueError, match="does not match"):
        cost_module.f5_stability_penalty(
            [island], topology, model, problem.samples_1,
            replace(problem.config, k_1=0.9),
        )


def test_a_later_d_is_verified_separately():
    # THE BUG. d belongs in the key because the probe SET depends on it:
    # the all-coordinates probes exist only for d > 1, so a model verified
    # at d = 1 has not been checked for cross-coordinate behaviour at all.
    # Reproduced before the fix as fast = 0.0 against loop = 0.3 at d = 2,
    # with no exception, after a clean d = 1 verification.
    #
    # This project really does evaluate one model at more than one
    # dimension -- the F6 d=1 and d=2 tests do exactly that.
    from islanding.cost import AlgorithmConfig
    from islanding.topology import Topology

    base = linear_consensus_model({1: 1.0}, {})
    topology = Topology(nodes=[1], edges=[])
    config = AlgorithmConfig(
        kappa_op=1.0, kappa_rec=0.7, k_1=0.3, P_1=1, k_2=0.1, P_2=1
    )

    def f_i(i, x_i):
        # Matches linear consensus exactly at d = 1; couples coordinates
        # above it, which no d = 1 probe can see.
        value = base.f_i(i, x_i)
        if len(x_i) > 1:
            return value + np.array([x_i[1], x_i[0]] + list(x_i[2:]))
        return value

    model = replace(base, f_i=f_i)
    island = [frozenset({1})]

    # Genuinely agrees at d = 1 -- correctly not flagged.
    cost_module.f5_stability_penalty(
        island, topology, model, np.array([[[1.0]]]), config
    )

    with pytest.raises(ValueError, match="does not match"):
        cost_module.f5_stability_penalty(
            island, topology, model, np.array([[[1.0, 1.0]]]), config
        )


def test_cross_coordinate_coupling_away_from_the_first_member_is_caught():
    # The old mixed probe set only x[0,0] and x[0,1], so it tested
    # decoupling at the FIRST member across coordinates 0 and 1 only. A
    # model coupling coordinates at another member passed.
    topology, problem = _ten_node()
    base = problem.model
    island = frozenset({1, 2, 3})
    members = sorted(island)
    target = members[1]                       # NOT the first member

    def f_i(i, x_i):
        value = base.f_i(i, x_i)
        if i == target and len(x_i) > 1:
            return value + np.array([x_i[1], x_i[0]] + list(x_i[2:]))
        return value

    with pytest.raises(ValueError, match="does not match"):
        cost_module.f5_stability_penalty(
            [island], topology, replace(base, f_i=f_i),
            problem.samples_1, problem.config,
        )


def test_a_perturbation_vanishing_on_binary_states_is_caught():
    # Unit basis and pairwise probes are complete only for a bracket
    # already known to be QUADRATIC, and replacing callables is exactly
    # what can break that. This perturbation -- sum_c x_c^2 (x_c - 1)^2 in
    # the energy, with its gradient -- is zero whenever every coordinate
    # is 0 or 1, so it passes every unit probe while leaving the closed
    # form wrong on ordinary fractional states.
    #
    # The dense fractional, mixed-sign probes are what catch it. Verified
    # directly: against the old unit-only set the worst discrepancy was
    # exactly 0.0; against the current set it is 8.27.
    from islanding.cost import AlgorithmConfig
    from islanding.topology import Topology

    base = linear_consensus_model({1: 1.0}, {})
    topology = Topology(nodes=[1], edges=[])
    config = AlgorithmConfig(
        kappa_op=1.0, kappa_rec=0.7, k_1=0.3, P_1=1, k_2=0.1, P_2=1
    )

    def W_i(i, x_i):
        return base.W_i(i, x_i) + float(np.sum(x_i ** 2 * (x_i - 1.0) ** 2))

    def grad_W_i(i, x_i):
        return base.grad_W_i(i, x_i) + 2.0 * x_i * (x_i - 1.0) * (2.0 * x_i - 1.0)

    # The perturbation really does vanish on binary coordinates, or the
    # test would be catching something a unit probe could already see.
    for binary in (np.array([0.0]), np.array([1.0])):
        assert np.sum(binary ** 2 * (binary - 1.0) ** 2) == 0.0
        assert np.all(2.0 * binary * (binary - 1.0) * (2.0 * binary - 1.0) == 0.0)

    with pytest.raises(ValueError, match="does not match"):
        cost_module.f5_stability_penalty(
            [frozenset({1})], topology, replace(base, W_i=W_i, grad_W_i=grad_W_i),
            np.array([[[0.5]]]), config,
        )


def test_every_member_gets_an_all_coordinates_probe():
    # The mixed probe used to be a single state with x[0,0] and x[0,1] set,
    # testing decoupling at the FIRST member across coordinates 0 and 1
    # only. There is now one all-coordinates state per member.
    #
    # Structural, and paired with a BEHAVIOURAL test: an earlier version
    # of this comment claimed behavioural separation was impossible,
    # because the dense probes also catch cross-coordinate coupling
    # anywhere. They do -- unless the coupling is built to vanish on
    # them, which
    # test_a_nonlinear_coupling_confined_to_one_member_is_caught_there
    # does. Narrowing this back to member 0 fails that test, not only
    # this one.
    size, d = 4, 3
    probes = _quadratic_form_probes(size, d)

    for a in range(size):
        wanted = np.zeros((size, d))
        wanted[a, :] = 1.0
        assert any(np.array_equal(probe, wanted) for probe in probes), (
            f"no all-coordinates probe for member {a}"
        )


def test_no_all_coordinates_probes_when_d_is_one():
    # They would duplicate the basis probes exactly.
    from islanding.cost import _quadratic_form_probes

    probes = _quadratic_form_probes(3, 1)
    basis = sum(
        1 for probe in probes
        if np.count_nonzero(probe) == 1 and probe.max() == 1.0
    )
    assert basis == 3


def test_the_probe_set_uses_non_binary_and_signed_amplitudes():
    # Pins the property the test above depends on. If the probes ever
    # collapsed back to unit-only, that test would still pass by accident
    # on some other fixture; this one would not.
    from islanding.cost import _quadratic_form_probes

    values = np.concatenate([p.ravel() for p in _quadratic_form_probes(3, 2)])

    assert np.any((values != 0.0) & (values != 1.0)), "no non-binary amplitude"
    assert np.any(values < 0.0), "no negative amplitude"
    # ...and a dense probe: every slot occupied at once.
    assert any(np.all(p != 0.0) for p in _quadratic_form_probes(3, 2))


@pytest.mark.parametrize("value", [np.nan, np.inf])
def test_nonfinite_on_both_sides_is_rejected(value):
    # abs(nan - nan) > tol is False, and so is abs(inf - inf) > tol, so
    # both sides being nonfinite reads as agreement under a bare
    # comparison and would mark the island verified.
    topology, problem = _ten_node()
    poisoned = replace(
        problem.model,
        island_quadratic_form=lambda nodes, edges, k: (
            sorted(nodes), np.full((len(nodes), len(nodes)), value)
        ),
        W_i=lambda i, x_i: float(value),
    )

    with pytest.raises(ValueError, match="not finite"):
        cost_module.f5_stability_penalty(
            [frozenset({1, 2, 3})], topology, poisoned,
            problem.samples_1, problem.config,
        )


@pytest.mark.parametrize(
    "field",
    ["grad_W_i", "grad_W_ij", "f_i", "f_ij", "W_i", "W_ij",
     "island_quadratic_form"],
)
def test_swapping_any_contributing_callable_is_caught(field):
    # Every callable the bracket depends on. replace() inherits the closed
    # form when any of them changes, so each must be verified against it
    # independently rather than covered by the others.
    topology, problem = _ten_node()
    base = problem.model
    swaps = {
        "grad_W_i": lambda: replace(
            base, grad_W_i=lambda i, x: 2.0 * base.grad_W_i(i, x)),
        # base's grad_W_ij returns zeros, so doubling it changes nothing --
        # it has to become genuinely nonzero to perturb the bracket.
        "grad_W_ij": lambda: replace(
            base, grad_W_ij=lambda i, j, a, b: (a - b, -(a - b))),
        "f_i": lambda: replace(base, f_i=lambda i, x: 2.0 * base.f_i(i, x)),
        "f_ij": lambda: replace(
            base, f_ij=lambda i, j, a, b: 2.0 * base.f_ij(i, j, a, b)),
        "W_i": lambda: replace(base, W_i=lambda i, x: 2.0 * base.W_i(i, x)),
        "W_ij": lambda: replace(base, W_ij=lambda i, j, a, b: 1.0),
        "island_quadratic_form": lambda: replace(
            base, island_quadratic_form=lambda n, e, k: (
                lambda pair: (pair[0], pair[1] * 2.0)
            )(base.island_quadratic_form(n, e, k))),
    }

    with pytest.raises(ValueError, match="does not match"):
        cost_module.f5_stability_penalty(
            [frozenset({1, 2, 3})], topology, swaps[field](),
            problem.samples_1, problem.config,
        )


class _UnhashableCallable:
    """A callable that cannot be a dict key -- e.g. a stateful functor."""

    __hash__ = None

    def __init__(self, function):
        self._function = function

    def __call__(self, *args, **kwargs):
        return self._function(*args, **kwargs)


def test_an_unhashable_callable_does_not_break_the_check():
    # A model carrying one cannot be registered, so the registry lookup
    # raises TypeError. The check must fall back to verifying every call
    # rather than crashing -- slower, never wrong.
    topology, problem = _ten_node()
    consistent = replace(
        problem.model, f_i=_UnhashableCallable(problem.model.f_i)
    )

    value = cost_module.f5_stability_penalty(
        [frozenset({1, 2, 3})], topology, consistent,
        problem.samples_1, problem.config,
    )
    assert np.isfinite(value)


def test_an_unhashable_callable_still_gets_verified():
    # ...and the fallback must still CATCH a stale form, not merely avoid
    # the crash.
    topology, problem = _ten_node()
    stale_and_unhashable = replace(
        _stale(problem.model), f_i=_UnhashableCallable(problem.model.f_i)
    )

    with pytest.raises(ValueError, match="does not match"):
        cost_module.f5_stability_penalty(
            [frozenset({1, 2, 3})], topology, stale_and_unhashable,
            problem.samples_1, problem.config,
        )


def test_the_registry_does_not_retain_models():
    # The old module-level set held seven callables per variant, and the
    # closed form's closure retains the weight dicts and every cached
    # island matrix -- so a parameter sweep grew memory per construction.
    import gc

    topology, problem = _ten_node()
    model = replace(problem.model)          # a fresh object to watch
    cost_module.f5_stability_penalty(
        [frozenset({1, 2, 3})], topology, model,
        problem.samples_1, problem.config,
    )
    assert model in cost_module._verified_quadratic_forms

    reference = weakref.ref(model)
    del model
    gc.collect()

    assert reference() is None, "the registry kept the model alive"


def _dense_probes(size, d):
    """The dense fractional probes from the live probe set.

    Taken from the generator rather than recomputed, so the constructions
    below stay pinned to the probes actually used if _PROBE_STRIDE or the
    dense count ever changes.
    """
    from islanding.cost import _DENSE_PROBES, _quadratic_form_probes

    probes = _quadratic_form_probes(size, d)[-_DENSE_PROBES:]
    assert all(np.all(probe != 0.0) for probe in probes), "not dense"
    return probes


def _symmetric_nullspace(constraints, d, pick):
    """A symmetric d x d matrix M with q(x) = x^T M x zero on `constraints`.

    `constraints` are the states q must vanish on; `pick` scores a
    candidate so a direction that is nonzero where the test needs it can
    be chosen from a nullspace of dimension > 1.
    """
    upper = [(p, q) for p in range(d) for q in range(p, d)]

    def build(coefficients):
        M = np.zeros((d, d))
        for value, (p, q) in zip(coefficients, upper):
            M[p, q] += value
            M[q, p] += value if p != q else 0.0
        return M

    rows = []
    for x in constraints:
        rows.append([
            (x[p] * x[q] * 2.0 if p != q else x[p] * x[q]) for p, q in upper
        ])

    _, _, vt = np.linalg.svd(np.array(rows))
    basis = vt[len(rows):]
    assert len(basis) > 0, "no nullspace to draw from"

    best = max((build(v) for v in basis), key=lambda M: abs(pick(M)))
    assert abs(pick(M := best)) > 1e-3, "nullspace direction is degenerate"
    return M


def test_a_general_quadratic_coupling_across_coordinates_is_caught():
    # THE ESCAPE THIS CLOSES. A state lives in R^(n x d), so a general
    # symmetric quadratic bracket over it has (n d)(n d + 1)/2
    # coefficients -- 10 for a singleton at d = 4. The probe set used to
    # range over MEMBERS at coordinate 0, giving that island 6 probes
    # (one basis, one all-coordinates, four dense) against 10 unknowns.
    # The nullspace is nonzero, and a perturbation drawn from it agrees
    # with the closed form on every probe while disagreeing elsewhere.
    #
    # This is NOT the acknowledged arbitrary-nonlinear limitation: the
    # replacement below is LINEAR in x and the resulting bracket is
    # exactly quadratic. It is a general quadratic rather than the
    # coordinate-decoupled one the closed form asserts, which is the
    # discrepancy the check exists to find.
    import islanding.cost as cost_module
    from islanding.cost import AlgorithmConfig
    from islanding.topology import Topology

    d = 4
    base = linear_consensus_model({1: 1.0}, {})
    topology = Topology(nodes=[1], edges=[])
    config = AlgorithmConfig(
        kappa_op=1.0, kappa_rec=0.7, k_1=0.3, P_1=1, k_2=0.1, P_2=1
    )

    # Since grad_W_i(1, x) = x for this model and the island has no
    # edges, replacing f_i by f_i + M x adds exactly x^T M x to the
    # bracket. Chosen to vanish on the OLD probe set: the coordinate-0
    # basis vector, the all-coordinates state, and the four dense probes.
    old_probes = [np.zeros(d), np.ones(d)]
    old_probes[0][0] = 1.0
    old_probes += [probe[0] for probe in _dense_probes(1, d)]
    ordinary = np.array([0.4, -1.3, 0.9, 2.1])
    M = _symmetric_nullspace(
        old_probes, d, pick=lambda M: float(ordinary @ M @ ordinary)
    )

    def f_i(i, x_i):
        return base.f_i(i, x_i) + M @ x_i

    escaped = replace(base, f_i=f_i)

    # The escape is real: on every probe the old set contained, the
    # perturbation is zero to rounding.
    worst_old = max(abs(float(x @ M @ x)) for x in old_probes)
    assert worst_old < 1e-12, worst_old
    # ...while an ordinary state is wrong by a wide margin.
    assert abs(float(ordinary @ M @ ordinary)) > 1e-2

    # Probing all n*d slots pins every coefficient, so it is caught.
    with pytest.raises(ValueError, match="does not match"):
        cost_module.f5_stability_penalty(
            [frozenset({1})], topology, escaped,
            ordinary.reshape(1, 1, d), config,
        )


def test_a_nonlinear_coupling_confined_to_one_member_is_caught_there():
    # THE BEHAVIOURAL SEPARATION of the per-member all-coordinates probes.
    # An earlier version of this file claimed none was possible because
    # the dense probes catch cross-coordinate coupling anywhere. They do
    # -- unless the coupling is built to vanish on them.
    #
    # The perturbation is prod_c x_c * (x^T M x) applied to node 2's
    # dynamics at d = 3. The leading product kills it whenever any
    # coordinate of x_2 is zero, which covers every basis probe, every
    # pairwise probe (two slots set, so at d = 3 some coordinate of x_2
    # is still zero), and the all-coordinates probe at the OTHER member.
    # M is then drawn from the nullspace of the four dense probes. What
    # is left is exactly the all-coordinates probe at node 2, where the
    # perturbation is 1^T M 1.
    import islanding.cost as cost_module
    from islanding.cost import AlgorithmConfig
    from islanding.topology import Topology

    d = 3
    base = linear_consensus_model({1: 1.0, 2: 1.0}, {(1, 2): 0.5})
    topology = Topology(nodes=[1, 2], edges=[(1, 2)])
    config = AlgorithmConfig(
        kappa_op=1.0, kappa_rec=0.7, k_1=0.3, P_1=1, k_2=0.1, P_2=1
    )

    members, _ = base.island_quadratic_form(frozenset({1, 2}), ((1, 2),), 0.3)
    target = members.index(2)
    # Node 2 must NOT be the first member, or narrowing the probe back to
    # member 0 would still cover it and the mutation would not separate.
    assert target != 0

    M = _symmetric_nullspace(
        [probe[target] for probe in _dense_probes(len(members), d)],
        d, pick=lambda M: float(np.ones(d) @ M @ np.ones(d)),
    )

    def f_i(i, x_i):
        if i != 2:
            return base.f_i(i, x_i)
        return base.f_i(i, x_i) + float(np.prod(x_i)) * (M @ x_i)

    hidden = replace(base, f_i=f_i)

    # It really is invisible to every other probe: zero whenever any
    # coordinate is zero, and zero on the dense probes by construction.
    for probe in _quadratic_form_probes(len(members), d):
        x = probe[target]
        if np.count_nonzero(x) < d:
            assert float(np.prod(x)) == 0.0
    for probe in _dense_probes(len(members), d):
        x = probe[target]
        assert abs(float(np.prod(x)) * float(x @ M @ x)) < 1e-12
    # ...and visible on this member's all-coordinates probe.
    assert abs(float(np.ones(d) @ M @ np.ones(d))) > 1e-3

    with pytest.raises(ValueError, match="does not match"):
        cost_module.f5_stability_penalty(
            [frozenset({1, 2})], topology, hidden,
            np.full((1, 2, d), 0.7), config,
        )


def test_the_fast_path_rejects_non_float64_samples():
    # DTYPE IS AN UNCHECKED DIMENSION otherwise. The probes are built as
    # float64, so a model agreeing with its closed form in float64 and
    # differing in another precision is verified with the wrong probes:
    # a float32 case passed verification and then gave a fast-path 0.0
    # against a loop value of 1.15.
    #
    # Adding dtype to the verification key would not have fixed it -- the
    # probes would still be float64. Rejected instead of verified per
    # dtype because the 1e-9 tolerance is below float32's precision, so
    # an honest float32 model would fail on rounding alone.
    import islanding.cost as cost_module
    from islanding.cost import AlgorithmConfig
    from islanding.topology import Topology

    base = linear_consensus_model({1: 1.0}, {})
    topology = Topology(nodes=[1], edges=[])
    config = AlgorithmConfig(
        kappa_op=1.0, kappa_rec=0.7, k_1=0.3, P_1=1, k_2=0.1, P_2=1
    )
    samples = np.array([[[0.5]]], dtype=np.float32)

    with pytest.raises(ValueError, match="float64"):
        cost_module.f5_stability_penalty(
            [frozenset({1})], topology, base, samples, config
        )

    # The same samples in float64 are accepted, so the rejection is about
    # dtype and not about this fixture.
    cost_module.f5_stability_penalty(
        [frozenset({1})], topology, base, samples.astype(np.float64), config
    )


def test_the_probe_set_covers_every_coordinate_slot():
    # Structural counterpart to the general-quadratic test above: basis
    # and pairwise probes must range over all n*d SLOTS, since that is
    # what makes them complete for a general symmetric quadratic bracket
    # rather than only a coordinate-decoupled one.
    size, d = 2, 3
    probes = _quadratic_form_probes(size, d)
    slots = [(a, c) for a in range(size) for c in range(d)]

    for a, c in slots:
        wanted = np.zeros((size, d))
        wanted[a, c] = 1.0
        assert any(np.array_equal(probe, wanted) for probe in probes), (
            f"no basis probe for slot ({a}, {c})"
        )

    for index, (a, c) in enumerate(slots):
        for b, e in slots[index + 1:]:
            wanted = np.zeros((size, d))
            wanted[a, c] = 1.0
            wanted[b, e] = 1.0
            assert any(np.array_equal(probe, wanted) for probe in probes), (
                f"no pairwise probe for slots ({a}, {c}) and ({b}, {e})"
            )
