"""Protection for the zero_pairwise_energy declaration and the skip it licenses.

g_i = grad_W_i(x_i) + sum over the within-island neighbours of the i-side
partial of grad_W_ij. For a model whose pairwise Lyapunov terms are
identically zero -- linear consensus, where W_ij == 0 by construction --
every summand of that sum is a zero vector, so the loop is pure cost:
30.0 million grad_W_ij calls and 17.8 s on the 20-node benchmark.

Whether it vanishes is a property of the MODEL, not of the state, and
cannot be read off the return values without computing them. So it is
declared, and the declaration is checked. The declaration is inherited by
dataclasses.replace exactly as island_quadratic_form is, which is the
failure these tests are mostly about: a model derived by giving linear
consensus a real edge energy keeps a claim that is now false, and would
silently compute g_i without its edge terms.

Both loops -- _stability_bracket's (F5's fallback path) and
_psi_reconnection's (F6) -- take the same treatment, so both are covered
here.
"""

from dataclasses import replace

import numpy as np
import pytest

import islanding.cost as cost_module
from islanding.cost import (
    AlgorithmConfig, f5_stability_penalty, f6_reconnection_stability,
    linear_consensus_model,
)
from islanding.topology import Topology


# Node 1 is unstable, so the island's bracket is POSITIVE and survives
# the hinge -- a term clipped to 0.0 would make the bit-identity
# comparison below agree for an uninteresting reason.
NODE_WEIGHTS = {1: -6.0, 2: 2.5, 3: 0.5, 4: 1.25}
# (3, 4) is the reconnected edge, weighted so Psi_r's Part 1 outweighs
# Part 4 on the sample draw below and F6 survives its hinge too.
EDGE_WEIGHTS = {(1, 2): 3.0, (2, 3): 1.5, (3, 4): 5.0, (1, 3): 2.0}
EDGES = sorted(EDGE_WEIGHTS)
ISLANDS = [frozenset({1, 2, 3})]
REC_EDGES = [(3, 4)]


def _config(P=2):
    return AlgorithmConfig(
        kappa_op=1.0, kappa_rec=0.7, k_1=0.3, P_1=P, k_2=0.1, P_2=P
    )


def _topology():
    return Topology(nodes=[1, 2, 3, 4], edges=EDGES)


def _samples(P=2, d=2):
    # Asymmetric and signed, so nothing vanishes by accident.
    rng = np.random.default_rng(3)
    return rng.normal(size=(P, 4, d))


def _plain():
    """Linear consensus, which declares the pairwise terms zero truthfully."""
    return linear_consensus_model(NODE_WEIGHTS, EDGE_WEIGHTS)


def _fallback(model=None):
    """`model` (default linear consensus) forced onto F5's LOOP path.

    The fast path has no g_i loop to skip, so every F5 test here has to
    clear the closed form. Cleared ONCE and reused rather than per call:
    the verification registry is keyed by the model object, so a fresh
    replace() per call would re-verify every time and make the call
    counts below meaningless.
    """
    return replace(model or _plain(), island_quadratic_form=None)


def _edge_energy(base, declare_zero, only_edge=None):
    """`base` given a genuinely nonzero edge energy.

    `declare_zero` controls whether the (now false) declaration is left in
    place -- the dataclasses.replace hazard, reproduced. `only_edge`
    confines the nonzero terms to a single edge, so the per-edge keying of
    the check can be separated from per-model keying.
    """
    def live(i, j):
        return only_edge is None or tuple(sorted((i, j))) == only_edge

    def W_ij(i, j, x_i, x_j):
        if not live(i, j):
            return 0.0
        delta = x_i - x_j
        return 0.5 * float(np.dot(delta, delta))

    def grad_W_ij(i, j, x_i, x_j):
        if not live(i, j):
            return (np.zeros_like(x_i), np.zeros_like(x_j))
        delta = x_i - x_j
        return (delta, -delta)

    return replace(
        base, W_ij=W_ij, grad_W_ij=grad_W_ij,
        island_quadratic_form=None, zero_pairwise_energy=declare_zero,
    )


def _recording(model):
    """`model` with every grad_W_ij call logged."""
    calls = []

    def grad_W_ij(i, j, x_i, x_j):
        calls.append((i, j))
        return model.grad_W_ij(i, j, x_i, x_j)

    return replace(model, grad_W_ij=grad_W_ij), calls


def _f5(model, samples, config):
    # `model` must already be on the fallback path (see _fallback).
    return f5_stability_penalty(
        ISLANDS, _topology(), model, samples, config
    )


def _f6(model, samples, config):
    return f6_reconnection_stability(
        ISLANDS, REC_EDGES, _topology(), model, samples, config
    )


# ---------------------------------------------------------------------------
# The skip happens, and only under the declaration
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("term", [_f5, _f6])
def test_the_within_island_gradient_loop_is_skipped_when_declared(term):
    # THE ASSERTION THAT PROTECTS THE OPTIMISATION. Removing the guard
    # changes no value -- the summands are zeros -- so no correctness test
    # can see it. Only the call log can.
    model, calls = _recording(_plain())
    model = _fallback(model) if term is _f5 else model
    config = _config()
    samples = _samples(config.P_1)

    # The FIRST call also runs _verify_zero_pairwise, which legitimately
    # calls grad_W_ij once per edge per probe. Discard those: the registry
    # is keyed by the model object, so a second call on the same object
    # verifies nothing and leaves only the evaluation's own calls.
    term(model, samples, config)
    assert calls, "the declaration was never verified"
    calls.clear()

    term(model, samples, config)

    # The g_i loop would call grad_W_ij once per within-island edge per
    # sample per island. Those edges are (1,2), (2,3) and (1,3).
    for edge in ((1, 2), (2, 3), (1, 3)):
        assert edge not in calls, f"the g_i loop still ran for {edge}"


def test_the_loop_still_runs_for_a_model_that_does_not_declare():
    # The default. A model that says nothing keeps the loop, whatever its
    # pairwise terms actually do.
    model, calls = _recording(_edge_energy(_plain(), declare_zero=False))
    config = _config()

    _f5(model, _samples(config.P_1), config)

    assert (1, 2) in calls


def test_f6_part_2_still_calls_grad_W_ij_for_reconnected_edges():
    # The skip is confined to the g_i loop. Psi_r's Part 2 takes the
    # i-side partial across each RECONNECTED edge, and that call is not
    # covered by the declaration's premise -- guarding it too would drop a
    # term rather than drop a zero.
    model, calls = _recording(_plain())
    config = _config()

    _f6(model, _samples(config.P_2), config)

    assert (3, 4) in calls, "Part 2's gradient across E^rec was skipped"


# ---------------------------------------------------------------------------
# Bit-identity
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("term", [_f5, _f6])
def test_the_skip_is_bit_identical(term):
    # float.hex(), not ==: the skipped summands are exactly zero, so the
    # surviving arithmetic is untouched and anything short of exact
    # equality would let a real change through. The comparison model is
    # the SAME model with the declaration withheld, so the two differ in
    # nothing but whether the loop runs.
    config = _config()
    samples = _samples(config.P_1)
    declared = _fallback() if term is _f5 else _plain()
    looping = replace(declared, zero_pairwise_energy=False)

    fast = term(declared, samples, config)
    slow = term(looping, samples, config)

    assert float.hex(fast) == float.hex(slow), (
        f"{fast!r} ({float.hex(fast)}) != {slow!r} ({float.hex(slow)})"
    )
    # Non-vacuous: a term of 0.0 would match for uninteresting reasons.
    assert fast != 0.0


# ---------------------------------------------------------------------------
# The declaration is checked
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("term", [_f5, _f6])
def test_an_inherited_declaration_is_rejected(term):
    # THE HAZARD. replace() inherits zero_pairwise_energy, so a model
    # derived from linear consensus by giving it a real edge energy keeps
    # a claim that is now false -- and would compute g_i without the very
    # terms that model exists to add, silently and with every signature
    # still matching.
    config = _config()
    wrong = _edge_energy(_plain(), declare_zero=True)

    with pytest.raises(ValueError, match="declares zero_pairwise_energy"):
        term(wrong, _samples(config.P_1), config)


@pytest.mark.parametrize("term", [_f5, _f6])
def test_one_agreeing_edge_does_not_bless_the_others(term):
    # PER EDGE, not once per model. The claim is about every edge, so a
    # model whose pairwise terms vanish on some edges and not others must
    # not be blessed by whichever edge is seen first. This is the same
    # escape a single agreeing island opened for the closed form.
    config = _config()
    # Zero on (1,2) and (2,3), nonzero on (1,3) -- and (1,2) is checked
    # first, since _induced_edges yields them in topology order.
    wrong = _edge_energy(_plain(), declare_zero=True, only_edge=(1, 3))

    with pytest.raises(ValueError, match=r"edge \(1, 3\)"):
        term(wrong, _samples(config.P_1), config)


@pytest.mark.parametrize("term", [_f5, _f6])
def test_a_declaration_true_only_on_the_states_a_run_drew_is_rejected(term):
    # The check evaluates the real state AND a deterministic probe set,
    # for the same reason the closed form's check does: a model that
    # happens to vanish where a particular run's samples landed would
    # otherwise be blessed by that accident. Here the pairwise terms are
    # zero exactly when the two endpoints are equal, which every sample
    # state below makes true and no probe does.
    config = _config()
    base = _plain()
    samples = np.ones((config.P_1, 4, 2)) * 0.6      # every node identical

    def grad_W_ij(i, j, x_i, x_j):
        delta = x_i - x_j
        return (delta, -delta)

    def W_ij(i, j, x_i, x_j):
        delta = x_i - x_j
        return 0.5 * float(np.dot(delta, delta))

    sneaky = replace(
        base, W_ij=W_ij, grad_W_ij=grad_W_ij, island_quadratic_form=None,
    )
    assert sneaky.zero_pairwise_energy, "inherited declaration expected"
    # The premise: on the run's own states the pairwise terms really are
    # zero, so checking those alone would pass.
    x = samples[0][0]
    assert np.all(grad_W_ij(1, 2, x, x)[0] == 0.0)

    with pytest.raises(ValueError, match="declares zero_pairwise_energy"):
        term(sneaky, samples, config)


@pytest.mark.parametrize("term", [_f5, _f6])
def test_the_check_is_paid_once_per_island_not_once_per_sample(term, monkeypatch):
    # The check sits OUTSIDE the sample loop in both callers. Inside, it
    # would be paid millions of times to protect a claim that cannot
    # change between samples -- which would cost more than the skip saves.
    #
    # COUNTING _verify_zero_pairwise, not grad_W_ij. An earlier version of
    # this test counted the callable, which cannot tell the two placements
    # apart: after the first island the registry hits and the function
    # returns without calling grad_W_ij at all, so a call moved inside the
    # sample loop would show the same count. Only the entry count pins
    # where it is invoked from.
    entries = []
    real = cost_module._verify_zero_pairwise

    def counting(model, edges, states):
        entries.append(tuple(sorted(edges)))
        return real(model, edges, states)

    monkeypatch.setattr(cost_module, "_verify_zero_pairwise", counting)

    def count(P):
        entries.clear()
        config = _config(P)
        model = _fallback() if term is _f5 else _plain()
        term(model, _samples(P), config)
        return len(entries)

    # One entry per island, whatever P is. len(ISLANDS) == 1 here, so this
    # also pins that it is not once per (island, sample).
    assert count(1) == count(8) == len(ISLANDS)


def test_a_false_declaration_survives_a_direct_call_to_the_bracket():
    # Scope, stated rather than assumed: _stability_bracket trusts the
    # declaration by default, so a DIRECT call bypasses the check exactly
    # as a direct call to island_quadratic_form bypasses its own. The
    # verification lives with the callers that own the island loop, which
    # is where every solver path goes.
    wrong = _edge_energy(_plain(), declare_zero=True)
    states = {i: np.array([0.4, -0.7]) * i for i in (1, 2, 3)}
    neighbors = {1: {2, 3}, 2: {1, 3}, 3: {1, 2}}

    skipped = cost_module._stability_bracket(
        frozenset({1, 2, 3}), ((1, 2), (2, 3), (1, 3)), neighbors,
        states, wrong, 0.3, hinge=False,
    )
    honest = cost_module._stability_bracket(
        frozenset({1, 2, 3}), ((1, 2), (2, 3), (1, 3)), neighbors,
        states, wrong, 0.3, hinge=False, trust_zero_pairwise=False,
    )

    # Not equal -- which is precisely why the declaration has to be
    # checked before the skip is trusted.
    assert skipped != honest


def test_the_closed_form_check_does_not_trust_the_declaration():
    # _verify_quadratic_form compares the closed form against a loop that
    # must NOT skip the same terms the closed form assumes away. Trusting
    # the declaration there would make the two agree for the wrong reason,
    # and a swapped grad_W_ij would stop being caught.
    topology = _topology()
    config = _config()
    base = _plain()
    swapped = replace(
        base, grad_W_ij=lambda i, j, a, b: (a - b, -(a - b))
    )
    assert swapped.island_quadratic_form is not None

    with pytest.raises(ValueError, match="does not match"):
        f5_stability_penalty(
            ISLANDS, topology, swapped, _samples(config.P_1), config
        )


# ---------------------------------------------------------------------------
# The shared zero accumulator
# ---------------------------------------------------------------------------

def test_the_reconnection_accumulator_allocates_nothing():
    # rec_flow starts at zero whatever the node, so it reuses the shared
    # read-only zero rather than allocating: np.zeros_like ran 16.8
    # million times here on the 20-node benchmark, 13.9 s, roughly 4.5 per
    # _psi_reconnection call.
    config = _config()
    samples = _samples(config.P_2)
    model = _plain()

    # Warm the cache first, so a miss inside the timed region cannot be
    # mistaken for the accumulator allocating.
    cost_module._shared_zero(samples[0][0])

    calls = []
    real = np.zeros_like

    def counting(*a, **k):
        calls.append(a[0].shape if a else None)
        return real(*a, **k)

    np.zeros_like = counting
    try:
        _f6(model, samples, config)
    finally:
        np.zeros_like = real

    assert calls == [], f"{len(calls)} arrays allocated in the F6 loop"


def test_the_shared_zero_is_not_written_into_by_the_accumulator():
    # THE ONE WAY THIS SUBSTITUTION GOES WRONG. `rec_flow = rec_flow +
    # term` rebinds and leaves the shared array alone; `rec_flow += term`
    # would write into it in place, and every later caller -- including
    # linear consensus's grad_W_ij, which hands out the same arrays --
    # would see a corrupted "zero". writeable=False turns that into a
    # loud failure, and this pins both halves.
    config = _config()
    samples = _samples(config.P_2)

    _f6(_plain(), samples, config)

    assert cost_module._zero_arrays, "no shared zero was used at all"
    for zero in cost_module._zero_arrays.values():
        assert not zero.flags.writeable
        assert np.all(zero == 0.0)


def test_the_shared_zero_does_not_make_f6_nonfinite():
    # RENAMED from a claim it did not support. It used to be called
    # "an empty reconnected set still contributes through Part 1", but
    # this fixture reconnects (3, 4), and a finite total says nothing
    # about whether Part 1 ran for a node with an empty N_i^rec -- a
    # skipped zero contribution is invisible in a finite sum.
    #
    # THE REAL COVERAGE for that is
    # tests/test_psi_short_circuit.py::test_part_1_is_evaluated_even_when_
    # rec_i_is_empty, which poisons the skipped node's gradient with a NaN
    # so the dot is observable. What this one pins is narrower and still
    # worth having: swapping a fresh np.zeros_like for the shared array
    # did not make the total nonfinite or otherwise degenerate.
    config = _config()
    samples = _samples(config.P_2)

    value = _f6(_plain(), samples, config)

    assert np.isfinite(value)
    assert value > 0.0


# ---------------------------------------------------------------------------
# The key, and the comparison
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("term", [_f5, _f6])
def test_a_later_d_is_verified_separately(term):
    # THE ESCAPE, and it is the third time this exact shape has appeared:
    # once for the closed form's structural key, once for its dtype, now
    # here. A registry keyed on the edge alone marks it verified at d = 1
    # and skips the check when the SAME model and edge come back at d = 2,
    # where the pairwise terms are nonzero. Wrongly skipped, F5 returned
    # 5.824999999999999 against an honest 0.0.
    #
    # The probe set itself depends on d -- the all-coordinates probes
    # exist only for d > 1 -- so a d = 1 verification has not looked at
    # cross-coordinate behaviour at all.
    config = _config()
    base = _plain()

    def grad_W_ij(i, j, x_i, x_j):
        if np.shape(x_i)[0] == 1:
            return (np.zeros_like(x_i), np.zeros_like(x_j))
        delta = x_i - x_j
        return (delta, -delta)

    def W_ij(i, j, x_i, x_j):
        if np.shape(x_i)[0] == 1:
            return 0.0
        delta = x_i - x_j
        return 0.5 * float(np.dot(delta, delta))

    sneaky = replace(
        base, W_ij=W_ij, grad_W_ij=grad_W_ij, island_quadratic_form=None,
    )
    assert sneaky.zero_pairwise_energy, "inherited declaration expected"

    # d = 1: the declaration is TRUE, so this passes and registers.
    term(sneaky, _samples(config.P_1, d=1), config)

    # d = 2, same model object and the same edges: it is now false.
    with pytest.raises(ValueError, match="declares zero_pairwise_energy"):
        term(sneaky, _samples(config.P_1, d=2), config)


@pytest.mark.parametrize("term", [_f5, _f6])
def test_a_later_dtype_is_verified_separately(term):
    # dtype is KEYED here, not restricted away as it is on F5's fast path.
    # That path compares under a 1e-9 tolerance, which is below float32's
    # own precision, so an honest float32 model would fail on rounding
    # alone and rejection was the only coherent option. This check
    # compares to exactly zero, where no such regime exists: a function
    # that returns zero returns it in every precision.
    config = _config()
    base = _plain()

    def grad_W_ij(i, j, x_i, x_j):
        if np.asarray(x_i).dtype == np.float64:
            return (np.zeros_like(x_i), np.zeros_like(x_j))
        delta = x_i - x_j
        return (delta, -delta)

    sneaky = replace(base, grad_W_ij=grad_W_ij, island_quadratic_form=None)
    assert sneaky.zero_pairwise_energy

    term(sneaky, _samples(config.P_1), config)

    with pytest.raises(ValueError, match="declares zero_pairwise_energy"):
        term(sneaky, _samples(config.P_1).astype(np.float32), config)


@pytest.mark.parametrize("term", [_f5, _f6])
@pytest.mark.parametrize("field", ["W_ij", "grad_u", "grad_v"])
def test_the_smallest_nonzero_pairwise_term_is_rejected(term, field):
    # THE COMPARISON IS EXACT, and nothing else pins that. Relaxing it to
    # np.allclose -- or to any tolerance at all -- passes every other test
    # in this file, because every other false declaration here is wrong by
    # an ordinary amount.
    #
    # It has to be exact because the skip is licensed by the summand being
    # zero, not by it being small: g_i drops the term entirely, so an
    # error of 5e-324 per edge is dropped in full rather than rounded
    # away, and a model returning it is not one whose pairwise energy
    # vanishes.
    # PARAMETRISED OVER ALL THREE COMPARISONS. The check tests W_ij,
    # grad_W_ij's u-side and grad_W_ij's v-side separately, so a single
    # fixture perturbing one of them leaves the other two free to be
    # relaxed independently. Each parameter here is wrong in exactly one
    # place and zero in the others.
    config = _config()
    tiny = np.nextafter(0.0, 1.0)
    assert tiny != 0.0 and np.allclose(tiny, 0.0)

    def W_ij(i, j, x_i, x_j):
        return tiny if field == "W_ij" else 0.0

    def grad_W_ij(i, j, x_i, x_j):
        return (
            np.full_like(x_i, tiny) if field == "grad_u"
            else np.zeros_like(x_i),
            np.full_like(x_j, tiny) if field == "grad_v"
            else np.zeros_like(x_j),
        )

    sneaky = replace(
        _plain(), W_ij=W_ij, grad_W_ij=grad_W_ij,
        island_quadratic_form=None,
    )
    assert sneaky.zero_pairwise_energy

    with pytest.raises(ValueError, match="declares zero_pairwise_energy"):
        term(sneaky, _samples(config.P_1), config)


@pytest.mark.parametrize("term", [_f5, _f6])
def test_the_probes_are_cast_to_the_state_dtype(term):
    # KEYING dtype is not the whole fix; the probes have to be BUILT in
    # that dtype too. An earlier version of this file only tested the
    # key: its float32 state already had differing endpoints, so the real
    # state alone raised and float64 probes would have gone unnoticed.
    #
    # This fixture removes that shortcut. The real float32 state has
    # IDENTICAL endpoint values, so a difference-based callable returns
    # zero on it; the callable returns zero for float64 inputs outright;
    # and the probes are the only pairs with differing endpoints. So the
    # check fires if and only if the probes reach the callable as
    # float32. Drop the .astype and the false declaration passes.
    config = _config()
    # Every node identical, so states[u] - states[v] is exactly zero.
    samples = np.full((config.P_1, 4, 2), 0.6, dtype=np.float32)

    def grad_W_ij(i, j, x_i, x_j):
        if np.asarray(x_i).dtype == np.float64:
            return (np.zeros_like(x_i), np.zeros_like(x_j))
        delta = x_i - x_j
        return (delta, -delta)

    def W_ij(i, j, x_i, x_j):
        if np.asarray(x_i).dtype == np.float64:
            return 0.0
        delta = x_i - x_j
        return 0.5 * float(np.dot(delta, delta))

    sneaky = replace(
        _plain(), W_ij=W_ij, grad_W_ij=grad_W_ij, island_quadratic_form=None,
    )
    assert sneaky.zero_pairwise_energy

    # The premise: on the run's own float32 states the terms really are
    # zero, and float64 probes would report zero as well.
    x = samples[0][0]
    assert np.all(grad_W_ij(1, 2, x, x)[0] == 0.0)
    sixty_four = np.array([0.3, -0.9])
    assert np.all(grad_W_ij(1, 2, sixty_four, -sixty_four)[0] == 0.0)

    with pytest.raises(ValueError, match="declares zero_pairwise_energy"):
        term(sneaky, samples, config)


def test_endpoints_from_different_domains_are_rejected_not_registered():
    # THE THIRD-INSTANCE FIX. The domain used to be read off states[u]
    # alone. _verify_zero_pairwise takes a generic dict, so a call whose
    # endpoints differ in shape or dtype would have registered under the
    # first endpoint's signature and suppressed a later check on a
    # materially different domain -- the same failure as omitting d or
    # dtype, arriving by a third route.
    #
    # No production path can build such a call today: every state is a
    # slice of one homogeneous samples array. That is why it has to be
    # rejected structurally rather than left to be noticed.
    model = _plain()

    mixed_dtype = {
        1: np.array([0.5, 0.25], dtype=np.float64),
        2: np.array([0.5, 0.25], dtype=np.float32),
    }
    with pytest.raises(ValueError, match="do not share a domain"):
        cost_module._verify_zero_pairwise(model, [(1, 2)], mixed_dtype)

    mixed_shape = {
        1: np.array([0.5, 0.25]),
        2: np.array([0.5]),
    }
    with pytest.raises(ValueError, match="do not share a domain"):
        cost_module._verify_zero_pairwise(model, [(1, 2)], mixed_shape)

    # And nothing was recorded for the edge, so a later well-formed call
    # is still checked rather than waved through.
    assert not cost_module._verified_zero_pairwise.get(model)


def test_a_state_of_the_wrong_rank_is_rejected():
    # The domain feeds probe construction as well as the key, so a state
    # that is not a 1-D vector would be probed in a domain it does not
    # belong to.
    model = _plain()
    square = {i: np.zeros((2, 2)) for i in (1, 2)}

    with pytest.raises(ValueError, match="must be 1-D vectors"):
        cost_module._verify_zero_pairwise(model, [(1, 2)], square)


def test_both_registries_key_on_the_shared_state_domain():
    # STRUCTURAL, and the point of the shared helper: the domain half of
    # both keys is one object built in one place, so it cannot be
    # assembled inconsistently at the two sites. Three separate rounds of
    # this bug is what makes that worth pinning rather than trusting.
    from examples.linear_10node.linear_10node import (
        NODE_AND_EDGE_FAULTS, build_topology, build_problem,
    )
    from examples.linear_10node.config import build_config

    topology = build_topology()
    problem = build_problem(
        NODE_AND_EDGE_FAULTS, topology, seed=0, config=build_config()
    )
    islands = [frozenset({1, 2, 3})]

    # F5's fast path fills the closed-form registry...
    f5_stability_penalty(
        islands, topology, problem.model, problem.samples_1, problem.config
    )
    keys = cost_module._verified_quadratic_forms[problem.model]
    assert keys
    for key in keys:
        assert any(isinstance(part, cost_module._StateDomain) for part in key)

    # ...and the fallback path fills the declaration registry. Bound to a
    # name first: the registry holds models weakly, so a model built
    # inline in the call would be collected before it could be read back.
    fallback = replace(problem.model, island_quadratic_form=None)
    f5_stability_penalty(
        islands, topology, fallback, problem.samples_1, problem.config
    )
    edge_keys = list(cost_module._verified_zero_pairwise[fallback])
    assert edge_keys
    for key in edge_keys:
        assert any(isinstance(part, cost_module._StateDomain) for part in key)
