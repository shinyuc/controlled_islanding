"""Protection for the short-circuit in `_psi_reconnection`.

`dyn_i` -- the full dynamics at node i -- feeds only Psi_r's Part 2, whose
inner sum runs over N_i^rec (docs/cost_function_spec.md section 14). For a
node with no reconnected neighbours it was computed and thrown away, which
on the 20-node benchmark was most of F6's f_ij calls. The guard skips it.

The change is bit-identical, and every existing correctness test agrees
with that -- which is the problem. Removing the guard entirely would also
pass all of them, because skipping discarded work changes no value. So
would the subtler regression of folding the Part 1 accumulation into the
guard, which looks free (with rec_i empty it adds exactly 0.0) but is not:
np.dot(g_i, zeros) is NaN rather than 0.0 when g_i holds a NaN, and
`x + 0.0` is not `x` when x is -0.0.

These tests therefore assert the SHAPE of the computation -- which model
callables run, for which nodes -- rather than only its result, plus a
bit-exact comparison of the raw Psi_r against an unoptimised reference.
"""

from dataclasses import replace

import numpy as np
import pytest

from islanding.cost import (
    AlgorithmConfig, _psi_reconnection, linear_consensus_model,
)


# The fixture: island {1, 2} joined by edge (1, 2). Node 1 has a
# reconnected neighbour 3 (outside the island, as E^rec requires); node 2
# has none. So node 1 takes the guard's live branch and node 2 its skipped
# branch, in a single call.
ISLAND = frozenset({1, 2})
EDGES_R = ((1, 2),)
NEIGHBOURS = {1: {2}, 2: {1}}
REC_NEIGHBOURS = {1: {3}}
NODE_WEIGHTS = {1: 1.5, 2: 2.5, 3: 0.5}
# (2, 3) is present only for the alternate fixture in
# test_the_skip_is_driven_by_rec_i_not_by_node_identity; the main
# fixture never looks it up, so it is inert elsewhere.
EDGE_WEIGHTS = {(1, 2): 3.0, (1, 3): 4.0, (2, 3): 5.0}


def _config():
    return AlgorithmConfig(
        kappa_op=1.0, kappa_rec=0.7, k_1=0.3, P_1=1, k_2=0.1, P_2=1
    )


def _states():
    # Asymmetric, so no term vanishes by symmetry.
    return {
        1: np.array([2.0]),
        2: np.array([-3.0]),
        3: np.array([5.0]),
    }


def _recording_model():
    """A linear-consensus model that logs every f_i / f_ij call."""
    base = linear_consensus_model(NODE_WEIGHTS, EDGE_WEIGHTS)
    calls = []

    def f_i(i, x_i):
        calls.append(("f_i", i))
        return base.f_i(i, x_i)

    def f_ij(i, j, x_i, x_j):
        calls.append(("f_ij", i, j))
        return base.f_ij(i, j, x_i, x_j)

    return replace(base, f_i=f_i, f_ij=f_ij), calls


def _reference_psi(nodes, edges_r, neighbors, rec_neighbors, states, model, config):
    """`_psi_reconnection` as it was BEFORE the short-circuit.

    Kept verbatim so the optimised version can be compared against it
    bit-for-bit. Note dyn_i is computed unconditionally here -- that is
    precisely the work the guard removes.
    """
    g = {i: model.grad_W_i(i, states[i]) for i in nodes}
    for u, v in edges_r:
        grad_u, grad_v = model.grad_W_ij(u, v, states[u], states[v])
        g[u] = g[u] + grad_u
        g[v] = g[v] + grad_v

    W_r = sum(model.W_i(i, states[i]) for i in nodes)
    W_r += sum(model.W_ij(u, v, states[u], states[v]) for u, v in edges_r)

    part_1 = 0.0
    part_2 = 0.0
    part_3 = 0.0
    for i in nodes:
        x_i = states[i]
        rec_i = rec_neighbors.get(i, ())

        rec_flow = np.zeros_like(x_i)
        for k in rec_i:
            rec_flow = rec_flow + model.f_ij(i, k, x_i, states[k])

        part_1 += float(np.dot(g[i], rec_flow))

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


def _call(model):
    return _psi_reconnection(
        ISLAND, EDGES_R, NEIGHBOURS, REC_NEIGHBOURS, _states(), model, _config()
    )


# ---------------------------------------------------------------------------
# The shape of the computation
# ---------------------------------------------------------------------------

def test_full_dynamics_run_only_for_the_reconnected_node():
    # THE ASSERTION THAT PROTECTS THE OPTIMISATION. Every correctness test
    # in the suite passes with the guard removed, because the skipped work
    # was discarded. Only the call log can tell the two apart.
    model, calls = _recording_model()

    _call(model)

    # Node 1 HAS a reconnected neighbour: its full dynamics must run --
    # f_i(1), plus f_ij over its within-island neighbours.
    assert ("f_i", 1) in calls
    assert ("f_ij", 1, 2) in calls

    # Node 2 has NO reconnected neighbour: its full dynamics must not run.
    # These are the calls the short-circuit exists to avoid.
    assert ("f_i", 2) not in calls, "dyn_i was computed for a node with empty rec_i"
    assert ("f_ij", 2, 1) not in calls, "dyn_i was computed for a node with empty rec_i"

    # The reconnection flow itself is NOT skipped -- it feeds Part 1.
    assert ("f_ij", 1, 3) in calls


def test_the_skip_is_driven_by_rec_i_not_by_node_identity():
    # Same island and states, but node 2 now also has a reconnected
    # neighbour. Its dynamics must come back -- otherwise the guard is
    # keyed on something other than N_i^rec being empty.
    model, calls = _recording_model()

    _psi_reconnection(
        ISLAND, EDGES_R, NEIGHBOURS,
        {1: {3}, 2: {3}},                      # node 2 now reconnects too
        _states(), model, _config(),
    )

    assert ("f_i", 2) in calls
    assert ("f_ij", 2, 1) in calls


def test_the_optimised_value_matches_the_reference_bit_for_bit():
    # float.hex(), not ==: skipping DISCARDED work cannot change any
    # floating-point sum, so anything less than exact equality would let a
    # real change through. This also catches a guard widened to skip work
    # that is NOT discarded -- Part 4's W_r accumulation in particular,
    # whose effect the hinge in f6_reconnection_stability would otherwise
    # hide, since W_r >= 0 makes Part 4 nonpositive.
    plain = linear_consensus_model(NODE_WEIGHTS, EDGE_WEIGHTS)

    optimised = _call(plain)
    reference = _reference_psi(
        ISLAND, EDGES_R, NEIGHBOURS, REC_NEIGHBOURS, _states(), plain, _config()
    )

    assert float.hex(optimised) == float.hex(reference), (
        f"{optimised!r} ({float.hex(optimised)}) != "
        f"{reference!r} ({float.hex(reference)})"
    )
    # Non-vacuous: a Psi_r of 0.0 would match for uninteresting reasons.
    assert optimised != 0.0


def test_the_reference_does_the_work_the_optimised_version_skips():
    # Guards the comparison above against drifting into a copy of the
    # optimised code: the reference must still compute the dynamics the
    # guard skips, or it would agree for the wrong reason.
    model, calls = _recording_model()

    _reference_psi(
        ISLAND, EDGES_R, NEIGHBOURS, REC_NEIGHBOURS, _states(), model, _config()
    )

    assert ("f_i", 2) in calls
    assert ("f_ij", 2, 1) in calls


# ---------------------------------------------------------------------------
# Why Part 1 stays OUTSIDE the guard
# ---------------------------------------------------------------------------

def test_part_1_is_evaluated_even_when_rec_i_is_empty():
    # Part 1's contribution for a node with empty rec_i is
    # np.dot(g_i, zeros). For finite g_i that is exactly 0.0, so moving
    # the line inside the guard is invisible to every ordinary test -- and
    # would still be wrong.
    #
    # With a NaN in g_i the dot is NaN, NOT 0.0, so the whole Psi_r
    # becomes NaN. That is the one observable proving Part 1 still runs
    # for the skipped node, and it is why the line was left outside.
    base = linear_consensus_model(NODE_WEIGHTS, EDGE_WEIGHTS)

    def grad_W_i(i, x_i):
        # Poison ONLY node 2 -- the one whose dynamics are skipped.
        return np.array([np.nan]) if i == 2 else base.grad_W_i(i, x_i)

    poisoned = replace(base, grad_W_i=grad_W_i)

    result = _call(poisoned)

    assert np.isnan(result), (
        "Psi_r is finite, so Part 1 was not evaluated for the node with "
        "empty rec_i -- the accumulation has been folded into the guard"
    )


def test_a_finite_gradient_on_the_skipped_node_still_contributes_zero():
    # The complement: the NaN above must come from Part 1 running, not
    # from the fixture being broken. With a finite gradient the same node
    # contributes exactly 0.0 and Psi_r is finite.
    plain = linear_consensus_model(NODE_WEIGHTS, EDGE_WEIGHTS)

    result = _call(plain)

    assert np.isfinite(result)


def test_part_4_is_computed_for_every_node_including_skipped_ones():
    # Part 4 is -(k_1 - k_2) * W_r, and W_r sums W_i over ALL island
    # nodes -- it does not involve N_i^rec, so the guard must not reach
    # it. Detected by giving node 2 a huge energy: if its W_i were skipped
    # the result would move by (k_1 - k_2) * that amount.
    base = linear_consensus_model(NODE_WEIGHTS, EDGE_WEIGHTS)
    config = _config()

    def W_i(i, x_i):
        return 1000.0 if i == 2 else base.W_i(i, x_i)

    inflated = replace(base, W_i=W_i)

    baseline = _call(base)
    with_inflated = _call(inflated)

    # Node 2's energy reached Part 4 despite its dynamics being skipped.
    assert with_inflated < baseline
    shift = (config.k_1 - config.k_2) * (1000.0 - base.W_i(2, _states()[2]))
    assert with_inflated == pytest.approx(baseline - shift, abs=1e-9, rel=0.0)
