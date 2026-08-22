"""Acceptance tests for the MILP baseline (docs/milp_baseline_spec.md s10).

Run on the 10-node case, where the local search's answer is known
(F = 6.9, node 10 shed, three reconnected edges).

Assertion 4 -- `test_objective_matches_the_cost_functions` -- is the
critical one. If the MILP's reported objective does not equal the cost
module's F on the same partition with the same sample sets, the two
methods are not minimising the same function and every downstream
comparison is meaningless.
"""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from examples.linear_10node.linear_10node import (
    NODE_AND_EDGE_FAULTS, build_topology, build_problem,
)
from examples.linear_10node.config import build_config
from islanding.topology import Topology
from islanding.cost import (
    AlgorithmConfig, SystemModel, canon,
    f1_unstable_nodes, f3_operation_edges, f4_reconnection,
    f5_stability_penalty, f6_reconnection_stability,
)
from baseline.config import MILPConfig
from islanding.inner import InnerSolveFailure
from islanding.outer_loop import solve
from baseline.milp import (
    milp_solver, phi_certificates, psi_certificates, build_model, NoIncumbentError,
    _status_name, _finite_or_none, _gap_block, _gurobi_status_constants,
    _STATUS_NAMES,
)

# The synthetic nonzero-edge-energy model comes from tests/conftest.py as
# the `synthetic_edge_energy_model` fixture -- shared with test_cost.py.
# It is NOT imported from test_cost.py: tests/ is not a package and is not
# in setuptools package discovery, so `from tests.test_cost import ...`
# resolves only when the repo root happens to be on sys.path (true under
# `python -m pytest`, false under a bare `pytest tests/test_baseline.py`).

TIME_LIMIT = 300.0


@pytest.fixture(scope="module")
def solved():
    topology = build_topology()
    problem = build_problem(
        NODE_AND_EDGE_FAULTS, topology, build_config(), seed=0
    )
    milp_config = MILPConfig(time_limit=TIME_LIMIT, mip_gap=1e-9, verbose=False)
    result = milp_solver(problem, milp_config)
    return problem, result


def _cost_terms(problem, result):
    """F on a partition, term by term, exactly as evaluate() composes it.

    evaluate() takes a spanning tree and decodes it; the MILP has no tree,
    so the same five terms are assembled directly from the partition. F2 is
    absent from both -- the MILP has no such term (spec section 3) and
    evaluate() passes 0.0 in its place.

    Returns the terms individually as well as the total, so a test can
    assert that the term it means to exercise is actually nonzero.
    """
    topology, config = problem.topology, problem.config
    stable = result.stable_islands
    unstable = result.unstable_islands
    reconnected = result.reconnected_edges

    f1 = f1_unstable_nodes(unstable, problem.cost_data)
    f3 = f3_operation_edges(
        stable, unstable, topology, problem.cost_data, config
    )
    f4 = f4_reconnection(reconnected, problem.cost_data, config)
    f5 = f5_stability_penalty(
        stable, topology, problem.model, problem.samples_1, config
    )
    f6 = f6_reconnection_stability(
        stable, reconnected, topology, problem.model, problem.samples_2, config
    )
    return {"f1": f1, "f3": f3, "f4": f4, "f5": f5, "f6": f6,
            "total": f1 + f3 - f4 + f5 + f6}


def _cost_on_partition(problem, result):
    return _cost_terms(problem, result)["total"]


def _partition_result(stable, unstable, reconnected):
    """Minimal stand-in for an InnerResult, for hand-built partitions."""
    return SimpleNamespace(
        stable_islands=stable,
        unstable_islands=unstable,
        reconnected_edges=reconnected,
    )


def test_model_builds_and_solves(solved):
    _, result = solved

    assert result.info["status"] in ("OPTIMAL", "TIME_LIMIT")
    assert result.info["n_vars"] > 0
    assert result.info["n_constrs"] > 0
    assert result.info["solve_time"] >= 0.0
    # A time limit that expires before any incumbent raises in milp_solver,
    # so reaching here means a partition was returned.
    assert result.stable_islands or result.unstable_islands


def test_returned_partition_covers_every_node_exactly_once(solved):
    problem, result = solved
    nodes = set(problem.topology.nodes())

    islands = list(result.stable_islands) + list(result.unstable_islands)
    covered = [node for island in islands for node in island]

    assert sorted(covered) == sorted(nodes)      # nothing missing
    assert len(covered) == len(set(covered))     # nothing double-counted


def test_reconnected_edges_are_cut_edges_between_distinct_stable_islands(solved):
    problem, result = solved
    topology = problem.topology

    island_of = {}
    for index, island in enumerate(result.stable_islands):
        for node in island:
            island_of[node] = index

    all_edges = {canon((u, v)) for u, v in topology.edges()}
    within = set()
    for island in list(result.stable_islands) + list(result.unstable_islands):
        within |= {
            canon((u, v)) for u, v in topology.edges()
            if u in island and v in island
        }

    for edge in result.reconnected_edges:
        i, j = edge
        assert edge in all_edges, f"{edge} is not an original edge"
        assert edge not in within, f"{edge} is not a cut edge"
        assert i in island_of and j in island_of, f"{edge} touches an unstable island"
        assert island_of[i] != island_of[j], f"{edge} is inside one stable island"


def test_objective_matches_the_cost_functions(solved):
    # ASSERTION 4 -- the critical one. Do not relax this tolerance: a
    # mismatch means the MILP encodes a different function than the local
    # search minimises, and the comparison is void.
    problem, result = solved

    assert result.objective == pytest.approx(
        _cost_on_partition(problem, result), rel=1e-6, abs=1e-6
    )


def _pinned_model(problem, stable, unstable, reconnected):
    """Build the model, pin a/b/y to a given partition, and solve it.

    Everything else -- the products, the flow, and the sigma epigraph
    variables -- is left to the solver, so the result is the model's own
    value of F at that point rather than a reconstruction of it.
    """
    topology, config = problem.topology, problem.config
    milp_config = MILPConfig(time_limit=TIME_LIMIT, mip_gap=1e-9, verbose=False)
    phi = phi_certificates(topology, problem.model, config, problem.samples_1)
    psi = psi_certificates(topology, problem.model, config, problem.samples_2)
    built = build_model(problem, milp_config, phi, psi)

    slot_of = {i: min(island) for island in list(stable) + list(unstable) for i in island}
    stable_slots = {min(island) for island in stable}

    for (i, r), var in built.a.items():
        value = 1.0 if slot_of[i] == r else 0.0
        var.setAttr("LB", value)
        var.setAttr("UB", value)
    for r, var in built.b.items():
        value = 1.0 if r in stable_slots else 0.0
        var.setAttr("LB", value)
        var.setAttr("UB", value)
    for edge, var in built.y.items():
        value = 1.0 if edge in reconnected else 0.0
        var.setAttr("LB", value)
        var.setAttr("UB", value)

    built.model.optimize()
    assert built.model.SolCount > 0, "pinned partition is infeasible"
    return built


def _milp_objective_at(problem, stable, unstable, reconnected):
    return _pinned_model(problem, stable, unstable, reconnected).model.ObjVal


def test_objective_matches_on_a_partition_that_reconnects():
    # The MILP's own optimum on this case declines all reconnection, so
    # assertion 4 above leaves the entire Psi reconnection machinery
    # (q^r, q^sr, q^rr) untested. This pins a hand-built partition that
    # DOES reconnect and checks the same equality there.
    #
    # The partition also pins the per-family slot bound of spec section 4:
    # for slot r=2, node i=2 reconnects to k=1 with pos[1] < pos[2], so a
    # uniform "r <= min(i,k)" would delete q^r[2,(2,1)] and silently drop
    # that island's reconnection contribution. q^r is bounded by i alone.
    topology = build_topology()
    problem = build_problem(
        NODE_AND_EDGE_FAULTS, topology, build_config(), seed=0
    )

    stable = [frozenset({1, 4, 8}), frozenset({2, 3, 7})]
    unstable = [frozenset({5, 6, 9, 10})]
    reconnected = {(1, 2)}

    terms = _cost_terms(
        problem, _partition_result(stable, unstable, reconnected)
    )
    # Guard on F6 ITSELF: the total exceeding some threshold proves nothing,
    # since f1 + f3 alone already do.
    assert terms["f6"] > 0.0

    assert _milp_objective_at(
        problem, stable, unstable, reconnected
    ) == pytest.approx(terms["total"], rel=1e-6, abs=1e-6)


def test_objective_matches_with_nonzero_edge_energy(synthetic_edge_energy_model):
    # The triple-indexed families -- tau, q^sr, q^rr -- carry Phi^t, Psi^sr
    # and Psi^rr, every one of which is proportional to grad_W_ij. On the
    # linear-consensus fixtures W_ij == 0, so all 378 of those constants are
    # identically zero on the 10-node case and the other tests here never
    # exercise that arithmetic at all.
    #
    # This swaps in the synthetic quadratic edge energy from test_cost.py
    # (W_ij = 0.5 (x_i - x_j)^2, grad_W_ij = (x_i - x_j, -(x_i - x_j))) via
    # dataclasses.replace, so ONLY the model differs, and pins a partition
    # chosen to switch on every family:
    #
    #   tau off-diagonal   (5,8,9): edge (5,8) internal to slot 5, with 9
    #                               also in slot 5 and 9 in N_5
    #   tau diagonal       (5,8,8): the alias tau_{r,ijj} = wb_{r,ij}
    #   q^sr               (5,8,2): internal stable edge (5,8) at node 5,
    #                               which also reconnects to 2
    #   q^rr off-diagonal  (5,2,6): node 5 reconnects TWICE, to 2 and to 6
    #   q^rr diagonal      (5,2,2): the alias q^rr_{r,ijj} = q^r_{r,ij}
    #   q^r                (5,2)  : partner 2 ranks BELOW slot 5, the
    #                               per-family bound case
    topology = build_topology()
    problem = build_problem(
        NODE_AND_EDGE_FAULTS, topology, build_config(), seed=0
    )
    problem = replace(problem, model=synthetic_edge_energy_model(problem.model))

    stable = [frozenset({1, 2, 3}), frozenset({5, 8, 9}), frozenset({6, 7, 10})]
    unstable = [frozenset({4})]
    reconnected = {(2, 5), (5, 6)}

    # The families must actually be live, or this test proves nothing.
    phi = phi_certificates(
        topology, problem.model, problem.config, problem.samples_1
    )
    psi = psi_certificates(
        topology, problem.model, problem.config, problem.samples_2
    )
    assert abs(phi.Phi_t[(5, 8, 9)]).max() > 0.0     # tau, off-diagonal
    assert abs(phi.Phi_t[(5, 8, 8)]).max() > 0.0     # tau, aliased diagonal
    assert abs(psi.Psi_sr[(5, 8, 2)]).max() > 0.0    # q^sr
    assert abs(psi.Psi_rr[(5, 2, 6)]).max() > 0.0    # q^rr, off-diagonal
    assert abs(psi.Psi_rr[(5, 2, 2)]).max() > 0.0    # q^rr, aliased diagonal
    assert abs(psi.Psi_r[(5, 2)]).max() > 0.0        # q^r

    terms = _cost_terms(
        problem, _partition_result(stable, unstable, reconnected)
    )
    assert terms["f5"] != 0.0
    assert terms["f6"] != 0.0

    assert _milp_objective_at(
        problem, stable, unstable, reconnected
    ) == pytest.approx(terms["total"], rel=1e-9, abs=1e-9)


def test_triple_family_variables_are_switched_on(solved, synthetic_edge_energy_model):
    # Companion to the test above: it shows the CONSTANTS are nonzero;
    # this shows the VARIABLES multiplying them are actually 1, so the
    # products genuinely enter the sigma rows rather than being multiplied
    # by an off variable.
    topology = build_topology()
    problem = build_problem(
        NODE_AND_EDGE_FAULTS, topology, build_config(), seed=0
    )
    problem = replace(problem, model=synthetic_edge_energy_model(problem.model))

    stable = [frozenset({1, 2, 3}), frozenset({5, 8, 9}), frozenset({6, 7, 10})]
    unstable = [frozenset({4})]
    reconnected = {(2, 5), (5, 6)}

    built = _pinned_model(problem, stable, unstable, reconnected)

    assert built.tau[(5, (5, 8, 9))].X == pytest.approx(1.0, abs=1e-6)
    assert built.wb[(5, (5, 8))].X == pytest.approx(1.0, abs=1e-6)   # tau alias
    assert built.qsr[(5, (5, 8, 2))].X == pytest.approx(1.0, abs=1e-6)
    assert built.qrr[(5, (5, 2, 6))].X == pytest.approx(1.0, abs=1e-6)
    assert built.qr[(5, (5, 2))].X == pytest.approx(1.0, abs=1e-6)   # qrr alias
    assert built.qr[(5, (5, 6))].X == pytest.approx(1.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Phi^e's j-side partial: taken from the CANONICAL call, not a reversed one
# ---------------------------------------------------------------------------

# Deliberately non-order-equivariant mock. grad_W_ij returns a DIFFERENT
# pair depending on argument order, and in particular
#
#     grad_W_ij(1, 2, ...)[1] = 5.0   !=   3.0 = grad_W_ij(2, 1, ...)[0]
#
# No physical edge energy behaves this way -- W_ij(x_i,x_j) == W_ji(x_j,x_i)
# forces those two to agree, which is exactly why the synthetic quadratic
# model in conftest.py cannot tell the two code paths apart. The asymmetry
# here exists only to separate them: _sample_cache's grad_ji_of[(i,j)]
# (element 1 of the canonical call) versus grad_ij[(j,i)] (element 0 of a
# reversed call).
#
# Every callable ignores its state arguments and keys off the identifiers
# alone, so the expected Phi^e below is exact arithmetic on these constants
# with no floating-point trace to follow.
_MOCK_GRAD_SELF = {(1, 2): 2.0, (2, 1): 3.0}    # element 0 of grad_W_ij(i,j,..)
_MOCK_GRAD_OTHER = {(1, 2): 5.0, (2, 1): 7.0}   # element 1 of grad_W_ij(i,j,..)
_MOCK_F_SELF = {1: 1.0, 2: 10.0}                # f_i
_MOCK_GRAD_NODE = {1: 0.5, 2: 0.25}             # grad_W_i
_MOCK_F_PAIR = {(1, 2): 4.0, (2, 1): -6.0}      # f_ij, also order-dependent
_MOCK_W_PAIR = 0.75                             # W_ij
_MOCK_K_1 = 2.0


def _order_sensitive_mock_model():
    """SystemModel on nodes {1,2} whose grad_W_ij is not order-equivariant."""

    def f_i(i, x_i):
        return np.array([_MOCK_F_SELF[i]])

    def f_ij(i, j, x_i, x_j):
        return np.array([_MOCK_F_PAIR[(i, j)]])

    def W_i(i, x_i):
        return 0.0

    def W_ij(i, j, x_i, x_j):
        return _MOCK_W_PAIR

    def grad_W_i(i, x_i):
        return np.array([_MOCK_GRAD_NODE[i]])

    def grad_W_ij(i, j, x_i, x_j):
        return (
            np.array([_MOCK_GRAD_SELF[(i, j)]]),
            np.array([_MOCK_GRAD_OTHER[(i, j)]]),
        )

    def sampler(threshold, P, n, d):
        return np.zeros((P, n, d))

    def inflation(topology):
        return 0.0

    def counterexample(unit_nodes, unit_edges, node_order, d, threshold_radius,
                       stability_tolerance):
        return []

    return SystemModel(
        f_i=f_i, f_ij=f_ij, W_i=W_i, W_ij=W_ij,
        grad_W_i=grad_W_i, grad_W_ij=grad_W_ij,
        sampler=sampler, inflation=inflation, counterexample=counterexample,
    )


def test_phi_e_takes_the_j_side_partial_from_the_canonical_call():
    # Phi^e_ij = dW_ij/dx_i . f_i + dW_ij/dx_j . f_j
    #            + grad W_i . f_ij + grad W_j . f_ji + k_1 W_ij
    #
    # Hand-computed on the mock above, term by term (all d = 1):
    #   dW_12/dx_1 . f_1  = 2.0 * 1.0    = +2.0
    #   dW_12/dx_2 . f_2  = 5.0 * 10.0   = +50.0   <-- the term under test
    #   grad W_1 . f_12   = 0.5 * 4.0    = +2.0
    #   grad W_2 . f_21   = 0.25 * -6.0  = -1.5
    #   k_1 W_12          = 2.0 * 0.75   = +1.5
    #                                     -------
    #                                       54.0
    #
    # Reverting the j-side term to grad_ij[(2, 1)] -- element 0 of the
    # REVERSED call grad_W_ij(2, 1, ...) -- swaps 5.0 for 3.0, giving
    # 3.0 * 10.0 = 30.0 and a total of 34.0. The second assertion pins that
    # wrong value explicitly, so the regression is named, not merely
    # excluded by a number that happens to differ.
    topology = Topology(nodes=[1, 2], edges=[(1, 2)])
    model = _order_sensitive_mock_model()
    config = AlgorithmConfig(
        kappa_op=1.0, kappa_rec=0.7, k_1=_MOCK_K_1, k_2=0.5,
    )
    samples = np.zeros((1, 2, 1))

    # The mock must genuinely separate the two paths, or this test is vacuous.
    x = np.zeros(1)
    assert model.grad_W_ij(1, 2, x, x)[1] != model.grad_W_ij(2, 1, x, x)[0]

    phi = phi_certificates(topology, model, config, samples)

    assert phi.Phi_e[(1, 2)][0] == pytest.approx(54.0)
    assert phi.Phi_e[(1, 2)][0] != pytest.approx(34.0)  # the reversed-lookup value

    # Companion check on trap 1: grad_ij is element 0 of the canonical call,
    # so Phi^t_ijk = dW_ij/dx_i . f_ik uses the SELF-side partial.
    #   Phi^t_122 = 2.0 * 4.0  = 8.0     (element 0 of grad_W_ij(1, 2, ..))
    #   Phi^t_211 = 3.0 * -6.0 = -18.0   (element 0 of grad_W_ij(2, 1, ..))
    assert phi.Phi_t[(1, 2, 2)][0] == pytest.approx(8.0)
    assert phi.Phi_t[(2, 1, 1)][0] == pytest.approx(-18.0)


# ---------------------------------------------------------------------------
# Solver-attribute edge cases (no model needed -- these are pure helpers)
# ---------------------------------------------------------------------------

def test_status_names_cover_the_codes_gurobi_actually_returns():
    # The map used to stop at 13, so a MEM_LIMIT or WORK_LIMIT termination
    # serialized as the bare string "17" -- which reads like a name.
    assert _status_name(2) == "OPTIMAL"
    assert _status_name(9) == "TIME_LIMIT"
    assert _status_name(15) == "USER_OBJ_LIMIT"
    assert _status_name(16) == "WORK_LIMIT"
    assert _status_name(17) == "MEM_LIMIT"


def test_status_constants_are_actually_read_off_gurobi():
    # The DYNAMIC path, which is the whole point of the lookup and which
    # silently did nothing before: GRB.Status is a StatusConstClass whose
    # __dict__ is EMPTY, so the original vars()-based reverse lookup found
    # zero constants and every code fell through to the static table. Every
    # status the other tests exercise is in that table, so they all passed.
    constants = _gurobi_status_constants()

    # Non-empty is the assertion that would have caught the bug.
    assert constants, "no status constants read from gurobipy"
    assert constants["OPTIMAL"] == 2
    assert constants["TIME_LIMIT"] == 9
    # Gurobi exposes strictly more than the static fallback lists, which is
    # what the dynamic path is for.
    assert len(constants) >= len(_STATUS_NAMES)


def test_codes_outside_the_static_table_still_resolve():
    # 18 and 19 exist in current Gurobi. They resolve through whichever
    # path is live, so this stays true if the fallback is ever trimmed.
    assert _status_name(18) == "LOCALLY_OPTIMAL"
    assert _status_name(19) == "LOCALLY_INFEASIBLE"


def test_static_fallback_agrees_with_gurobi():
    # The fallback is only correct if it matches what gurobipy reports; a
    # drifted table would mislabel a status when the dynamic path is
    # unavailable (gurobipy absent).
    constants = _gurobi_status_constants()
    by_code = {code: name for name, code in constants.items()}

    for code, name in _STATUS_NAMES.items():
        assert by_code[code] == name, f"fallback disagrees on code {code}"


def test_an_unknown_status_code_is_labelled_as_unrecognised():
    name = _status_name(999)

    assert "999" in name
    assert name != "999"          # must not masquerade as a status name
    assert "UNRECOGNISED" in name.upper()


def test_gurobi_infinity_never_reaches_the_output_as_a_number():
    # MIPGap returns 1e100 when the incumbent objective is zero or no bound
    # exists. Serialized naively that is an enormous but ordinary-looking
    # float, indistinguishable from a real (catastrophic) gap.
    assert _finite_or_none(1e100) is None
    assert _finite_or_none(-1e100) is None
    assert _finite_or_none(float("inf")) is None
    assert _finite_or_none(float("nan")) is None
    assert _finite_or_none(None) is None
    assert _finite_or_none(0.4) == 0.4


def test_unavailable_relative_gap_is_reported_as_null_with_a_reason():
    block = _gap_block(objective=0.0, best_bound=0.0, raw_gap=1e100)

    assert block["attained_mip_gap"] is None
    assert block["attained_mip_gap_unavailable_reason"]
    # The absolute gap survives precisely where the relative one does not.
    assert block["absolute_gap"] == 0.0


def test_absolute_gap_is_recorded_alongside_a_usable_relative_gap():
    block = _gap_block(objective=10.0, best_bound=8.0, raw_gap=0.2)

    assert block["attained_mip_gap"] == 0.2
    assert block["attained_mip_gap_unavailable_reason"] is None
    assert block["absolute_gap"] == 2.0


def test_absolute_gap_is_null_when_there_is_no_bound():
    # An INFEASIBLE model has no bound to report, so neither gap exists.
    block = _gap_block(objective=None, best_bound=None, raw_gap=None)

    assert block["attained_mip_gap"] is None
    assert block["absolute_gap"] is None


def test_no_incumbent_gap_message_does_not_point_at_another_null():
    # With no incumbent the absolute gap is necessarily null too, so
    # telling the reader to "read absolute_gap instead" sends them nowhere.
    no_incumbent = _gap_block(objective=None, best_bound=-158.9, raw_gap=1e100)
    assert no_incumbent["absolute_gap"] is None
    assert "absolute_gap" not in no_incumbent["attained_mip_gap_unavailable_reason"]
    assert "best_bound" in no_incumbent["attained_mip_gap_unavailable_reason"]

    # With an incumbent the absolute gap IS available, so it is offered.
    with_incumbent = _gap_block(objective=0.0, best_bound=0.0, raw_gap=1e100)
    assert with_incumbent["absolute_gap"] == 0.0
    assert "absolute_gap" in with_incumbent["attained_mip_gap_unavailable_reason"]


def test_static_fallback_is_used_when_the_dynamic_lookup_is_unavailable(monkeypatch):
    # gurobipy absent, or a release that stops exposing GRB.Status the way
    # we read it. Every other status test runs with the dynamic path live,
    # so without this the fallback table is never actually exercised -- the
    # same blind spot that let the vars()-based lookup fail silently.
    import baseline.milp as milp

    monkeypatch.setattr(milp, "_gurobi_status_constants", dict)

    assert milp._status_name(2) == "OPTIMAL"
    assert milp._status_name(9) == "TIME_LIMIT"
    assert milp._status_name(17) == "MEM_LIMIT"
    assert milp._status_name(19) == "LOCALLY_INFEASIBLE"
    # Still labelled, not returned bare, when neither source knows it.
    assert "UNRECOGNISED" in milp._status_name(999).upper()


# ---------------------------------------------------------------------------
# The failure type survives the outer loop
# ---------------------------------------------------------------------------

def test_no_incumbent_error_is_an_inner_solve_failure():
    # The outer loop catches InnerSolveFailure without importing baseline
    # (islanding must not depend on it). If this subtyping is ever broken,
    # the failure escapes solve() unenriched and the run script records the
    # ORIGINAL problem's sample counts beside augmented diagnostics.
    assert issubclass(NoIncumbentError, InnerSolveFailure)


def test_no_incumbent_error_keeps_its_type_and_info_through_solve(two_node_problem):
    # The loop re-raises the SAME object, so `except NoIncumbentError` in
    # the run script still catches it after enrichment.
    raised = NoIncumbentError(
        "no feasible solution", info={"status": "TIME_LIMIT", "n_vars": 7}
    )

    def failing_solver(problem, **kwargs):
        raise raised

    with pytest.raises(NoIncumbentError) as caught:
        solve(two_node_problem, max_outer_iterations=3, inner_solver=failing_solver)

    assert caught.value is raised                     # same object
    assert isinstance(caught.value, NoIncumbentError)  # not widened
    assert caught.value.info["n_vars"] == 7            # solver diagnostics kept
    assert caught.value.outer_state is not None        # loop state attached
    assert caught.value.outer_state["failed_pass"] == 0
