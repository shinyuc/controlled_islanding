"""Tests for the JSON utils/save.py emits.

The saved result is the only artifact that outlives a run, so what it
records -- and what it claims -- is checked here rather than left to be
read off a terminal that has scrolled away. These assert on the emitted
payload, not on the printing in the run scripts.

The focus is the method block: whether a run was proved optimal, over
what scope that claim holds, and whether a run that produced no solution
at all leaves anything behind.
"""

import json

import numpy as np
import pytest

from islanding.topology import Topology
from islanding.cost import CostData, AlgorithmConfig, linear_consensus_model
from islanding.evaluate import Problem
from islanding.outer_loop import SolveResult
from utils.save import (
    LOCAL_SEARCH, MILP, save_result, save_failure_result,
)


def _problem():
    topology = Topology(nodes=[1, 2], edges=[(1, 2)])
    model = linear_consensus_model({1: -1.0, 2: 3.0}, {(1, 2): 3.0})
    config = AlgorithmConfig(
        kappa_op=1.0, kappa_rec=0.7,
        alpha_1=1.0, beta=1.0, k_1=0.2, P_1=1,
        alpha_2=1.0, gamma=1.0, k_2=0.2, P_2=1,
    )
    samples = np.array([[[1.0], [0.0]]])
    return Problem(
        topology=topology, model=model,
        cost_data=CostData(node_costs={1: 1, 2: 1}, edge_costs={(1, 2): 1}),
        config=config, samples_1=samples, samples_2=samples,
    )


def _result(info, reason="converged", final_K=(5, 7), objective=3.0):
    """A SolveResult with one merged stable island and the given info."""
    return SolveResult(
        objective=objective,
        inner_solve_count=2,
        augmentation_rounds=1,
        total_witnesses_added=2,
        total_island_witnesses_added=1,
        total_component_witnesses_added=1,
        final_K_phi=final_K[0],
        final_K_psi=final_K[1],
        reason=reason,
        stable_islands=[frozenset({1, 2})],
        unstable_islands=[],
        reconnected_edges=set(),
        info=info,
        pass_history=({"pass": 0, **info},),
    )


_COST_TERMS = {
    "f1": 0.0, "f3": 1.0, "f4": 0.0, "f5": 0.0, "f6": 0.0,
    "economic_cost": 1.0, "total_with_penalties": 1.0,
    "K_phi": 1, "K_psi": 1,
}


def _save(tmp_path, result, method, tree_init_seed="auto"):
    # Derived from the method by default: the MILP initialises no tree, so
    # its only consistent value is None. The helper used to hardcode 0 for
    # both, writing a MILP result stamped with a tree start that never
    # happened -- which save_result now rejects outright.
    if tree_init_seed == "auto":
        tree_init_seed = None if method == MILP else 0
    path = save_result(
        result, _problem(), example_name="fixture",
        method=method, sample_seed=0, tree_init_seed=tree_init_seed,
        max_outer_iterations=10,
        runtime_seconds=1.5,
        setup_name="fixture_setup", faulty_nodes=[1], faulty_edges=[(1, 2)],
        island_lambda_min=[1.0],       # one stable island
        component_lambda_min=[1.0],    # one reconnected component
        cost_terms=_COST_TERMS,
        final_island_witnesses=0, final_component_witnesses=0,
        results_dir=tmp_path,
    )
    with open(path) as f:
        return json.load(f)


_MILP_OPTIMAL = {
    "status": "OPTIMAL", "requested_mip_gap": 1e-6,
    "attained_mip_gap": 0.0, "attained_mip_gap_unavailable_reason": None,
    "absolute_gap": 0.0, "best_bound": 3.0, "time_limit": 300.0,
    "solve_time": 0.25, "n_vars": 100, "n_constrs": 200,
}
_MILP_TIMED_OUT = {
    **_MILP_OPTIMAL, "status": "TIME_LIMIT",
    "attained_mip_gap": 0.4, "absolute_gap": 2.0, "best_bound": 1.0,
}


# ---------------------------------------------------------------------------
# The MILP method block: what separates a proven run from a timed-out one
# ---------------------------------------------------------------------------

def test_milp_diagnostics_are_persisted_not_just_printed(tmp_path):
    # Before this block existed, a TIME_LIMIT result and a proven-optimal
    # one serialized identically -- same partition shape, same cost keys --
    # and everything distinguishing them lived only in the terminal.
    payload = _save(tmp_path, _result(_MILP_TIMED_OUT), MILP)
    method = payload["method"]

    assert method["name"] == MILP
    assert method["status"] == "TIME_LIMIT"
    assert method["requested_mip_gap"] == 1e-6
    assert method["attained_mip_gap"] == 0.4
    assert method["time_limit"] == 300.0
    assert method["final_pass_solve_time"] == 0.25
    assert method["n_vars"] == 100
    assert method["n_constrs"] == 200
    # Incumbent and bound together bracket the final inner optimum.
    assert method["best_bound"]["value"] == 1.0
    assert payload["objective"]["value"] == 3.0


def test_optimal_and_timed_out_runs_are_distinguishable(tmp_path):
    optimal = _save(tmp_path, _result(_MILP_OPTIMAL), MILP)["method"]
    timed_out = _save(tmp_path, _result(_MILP_TIMED_OUT), MILP)["method"]

    assert optimal["final_inner_solve"]["proved_optimal"] is True
    assert timed_out["final_inner_solve"]["proved_optimal"] is False
    assert optimal["best_bound"]["value"] != timed_out["best_bound"]["value"]


def test_optimality_claim_is_scoped_to_the_final_inner_solve(tmp_path):
    # proved_optimal must never read as an unqualified claim: it covers the
    # final sampled MILP at the recorded K's and nothing wider.
    payload = _save(tmp_path, _result(_MILP_OPTIMAL, final_K=(5, 7)), MILP)
    method = payload["method"]

    scope = method["final_inner_solve"]["optimality_scope"]
    assert "K_phi=5" in scope and "K_psi=7" in scope
    assert "FINAL inner solve" in scope
    # The same scope qualifies the bound, whose interval brackets that one
    # sampled objective and no other.
    assert method["best_bound"]["scope"] == scope


def test_proved_optimal_can_coexist_with_the_outer_cap(tmp_path):
    # The two describe different things, so this combination is valid and
    # must not be flattened into a single "did it work" verdict.
    payload = _save(
        tmp_path, _result(_MILP_OPTIMAL, reason="max_outer_iterations"), MILP
    )

    assert payload["reason"] == "max_outer_iterations"
    assert payload["method"]["final_inner_solve"]["proved_optimal"] is True


def test_unavailable_relative_gap_is_recorded_with_a_reason(tmp_path):
    # Gurobi returns its infinity sentinel for MIPGap when the incumbent is
    # zero; milp_solver converts that to None, and the reason has to travel
    # with it or the null looks like a missing field.
    info = {
        **_MILP_OPTIMAL,
        "attained_mip_gap": None,
        "attained_mip_gap_unavailable_reason": "no finite relative gap",
        "absolute_gap": 0.0,
    }
    method = _save(tmp_path, _result(info), MILP)["method"]

    assert method["attained_mip_gap"] is None
    assert method["attained_mip_gap_unavailable_reason"] == "no finite relative gap"
    assert method["absolute_gap"] == 0.0


# ---------------------------------------------------------------------------
# Local search: it CAN prove optimality
# ---------------------------------------------------------------------------

def test_zero_objective_is_recorded_as_proved_optimal(tmp_path):
    # F_hat >= 0 is guaranteed by CostData's nonnegative costs and
    # AlgorithmConfig's kappa_op > 0 (both validated at construction), so
    # attaining zero attains the minimum -- a proof, not a local optimum.
    #
    # The objective here is ACTUALLY zero. Passing a nonzero one would let
    # this test validate the reason->flag lookup while accepting a record
    # that contradicts itself.
    info = {"reason": "zero_objective", "iterations": 1}
    payload = _save(tmp_path, _result(info, objective=0.0), LOCAL_SEARCH)
    method = payload["method"]

    assert payload["objective"]["value"] == 0.0

    assert method["final_inner_solve"]["proved_optimal"] is True
    assert method["final_inner_solve"]["proof_basis"] == (
        "objective is nonnegative; zero attained"
    )


@pytest.mark.parametrize("reason", ["local_optimum", "max_iterations"])
def test_other_local_search_exits_prove_nothing(reason, tmp_path):
    method = _save(
        tmp_path, _result({"reason": reason, "iterations": 4}), LOCAL_SEARCH
    )["method"]

    assert method["final_reason"] == reason
    assert method["final_inner_solve"]["proved_optimal"] is False
    assert method["final_inner_solve"]["proof_basis"] is None


def test_local_search_optimality_is_scoped_too(tmp_path):
    # A zero objective proves optimality of the SAMPLED problem at those
    # K's -- not of the underlying one, whose F5/F6 are approximated by it.
    info = {"reason": "zero_objective", "iterations": 1}
    method = _save(
        tmp_path, _result(info, final_K=(9, 11), objective=0.0), LOCAL_SEARCH
    )["method"]

    scope = method["final_inner_solve"]["optimality_scope"]
    assert "K_phi=9" in scope and "K_psi=11" in scope
    assert "unsampled" in scope


# ---------------------------------------------------------------------------
# The no-incumbent path
# ---------------------------------------------------------------------------

def test_a_run_with_no_incumbent_still_writes_a_result(tmp_path):
    # Previously this wrote nothing at all, so a MILP that timed out before
    # its first incumbent was indistinguishable in results/ from a job that
    # was never launched.
    info = {
        "status": "TIME_LIMIT", "requested_mip_gap": 1e-6,
        "attained_mip_gap": None,
        "attained_mip_gap_unavailable_reason": "no finite relative gap",
        "absolute_gap": None, "best_bound": -158.9, "time_limit": 0.05,
        "solve_time": 0.05, "n_vars": 2491, "n_constrs": 6284,
        "incumbent": None,
    }
    path = save_failure_result(
        "fixture", method=MILP, sample_seed=0, tree_init_seed=None,
        max_outer_iterations=10,
        runtime_seconds=0.06, setup_name="fixture_setup",
        faulty_nodes=[1], faulty_edges=[(1, 2)],
        problem=_problem(), info=info,
        error=RuntimeError("MILP produced no feasible solution"),
        results_dir=tmp_path,
    )
    with open(path) as f:
        payload = json.load(f)

    # Discriminated from a solved run, not merely missing keys.
    assert payload["outcome"] == "no_incumbent"
    assert payload["incumbent"] is None
    assert payload["objective"] is None
    assert "no feasible solution" in payload["error"]

    # The termination is fully described: status, limits, size, bound.
    assert payload["method"]["status"] == "TIME_LIMIT"
    assert payload["method"]["time_limit"] == 0.05
    assert payload["method"]["best_bound"]["value"] == -158.9
    assert payload["method"]["n_vars"] == 2491
    assert payload["runtime_seconds"] == 0.06
    assert payload["sample_seed"] == 0
    # None, not 0: the MILP initialises no tree, and writing 0 would imply
    # a start that never happened.
    assert payload["tree_init_seed"] is None
    assert payload["config"]["k_1"] == 0.2


def test_solved_and_no_incumbent_payloads_are_discriminated(tmp_path):
    solved = _save(tmp_path, _result(_MILP_OPTIMAL), MILP)

    assert solved["outcome"] == "solved"
    # A reader keying on `outcome` never has to guess which schema it has.
    assert "stable_islands" in solved
    assert "cost" in solved


def test_run_parameters_are_recorded(tmp_path):
    # The seeds and max_outer_iterations are not in the AlgorithmConfig, so
    # without these a saved result cannot be reproduced. The two seeds are
    # recorded separately: they are independent knobs that merely happen to
    # both be 0 in the shipped configs, so a single field would hide the
    # tree-initialisation degree of freedom entirely.
    milp = _save(tmp_path, _result(_MILP_OPTIMAL), MILP)
    local = _save(tmp_path, _result({"reason": "local_optimum", "iterations": 3}),
                  LOCAL_SEARCH)

    assert milp["sample_seed"] == 0
    assert local["sample_seed"] == 0
    # The two seeds are independent, and the tree seed's value depends on
    # whether the method has a tree at all.
    assert milp["tree_init_seed"] is None
    assert local["tree_init_seed"] == 0
    assert "seed" not in milp             # the ambiguous single field is gone
    assert milp["max_outer_iterations"] == 10


def test_cost_terms_missing_a_key_is_rejected(tmp_path):
    result = _result(_MILP_OPTIMAL)
    incomplete = {key: value for key, value in _COST_TERMS.items() if key != "K_phi"}

    with pytest.raises(ValueError, match="K_phi"):
        save_result(
            result, _problem(), example_name="fixture",
            method=MILP, sample_seed=0, tree_init_seed=None,
            max_outer_iterations=10, runtime_seconds=1.5,
            setup_name="s", faulty_nodes=[], faulty_edges=[],
            island_lambda_min=[1.0], component_lambda_min=[1.0],
            cost_terms=incomplete,
            final_island_witnesses=0, final_component_witnesses=0,
            results_dir=tmp_path,
        )


@pytest.mark.parametrize(
    "objective",
    [
        3.0,            # plainly contradictory
        5e-13,          # inside the old > 1e-12 window, so it slipped through
        -5e-13,         # and so did its negation
        float("nan"),   # every comparison against NaN is False
        float("inf"),
    ],
    ids=["nonzero", "tiny_positive", "tiny_negative", "nan", "inf"],
)
def test_zero_objective_is_refused_unless_the_objective_is_exactly_zero(
    objective, tmp_path
):
    # reason and objective come from the same InnerResult and cannot
    # legitimately disagree. Recording "proved optimal" on anything but a
    # true zero would launder the inconsistency into an artifact that
    # outlives the run.
    #
    # The tiny values and NaN are the regressions: the old guard rejected
    # only abs(objective) > 1e-12, so 5e-13 passed the comparison and NaN
    # failed it in the wrong direction -- both persisted as proved optimal.
    info = {"reason": "zero_objective", "iterations": 1}

    with pytest.raises(ValueError, match="zero_objective"):
        _save(tmp_path, _result(info, objective=objective), LOCAL_SEARCH)


def test_an_exact_zero_is_still_accepted(tmp_path):
    # The tightened guard must reject nothing local_search legitimately
    # produces: it selects this exit with `objective == 0` exactly.
    info = {"reason": "zero_objective", "iterations": 1}

    payload = _save(tmp_path, _result(info, objective=0.0), LOCAL_SEARCH)

    assert payload["method"]["final_inner_solve"]["proved_optimal"] is True


@pytest.mark.parametrize("objective", [5e-13, float("nan")])
def test_other_exit_reasons_are_unaffected_by_the_guard(objective, tmp_path):
    # The guard is about a contradictory PROOF, not about policing every
    # objective: a local_optimum exit claims nothing and is left alone.
    info = {"reason": "local_optimum", "iterations": 4}

    method = _save(
        tmp_path, _result(info, objective=objective), LOCAL_SEARCH
    )["method"]

    assert method["final_inner_solve"]["proved_optimal"] is False


def test_milp_optimality_scope_does_not_claim_exact_optimality(tmp_path):
    # Gurobi's OPTIMAL is optimal to its tolerances, not exact arithmetic
    # optimality -- and the gaps that qualify it are recorded alongside.
    method = _save(tmp_path, _result(_MILP_OPTIMAL), MILP)["method"]

    scope = method["final_inner_solve"]["optimality_scope"]
    assert "tolerances" in scope
    assert method["requested_mip_gap"] == 1e-6


# ---------------------------------------------------------------------------
# The no-incumbent path after augmentation
# ---------------------------------------------------------------------------

_FAILURE_INFO = {
    "status": "TIME_LIMIT", "requested_mip_gap": 1e-6,
    "attained_mip_gap": None,
    "attained_mip_gap_unavailable_reason": "no finite relative gap",
    "absolute_gap": None, "best_bound": -158.9, "time_limit": 0.05,
    "solve_time": 0.05, "n_vars": 2491, "n_constrs": 6284, "incumbent": None,
}


def _save_failure(tmp_path, outer_state):
    path = save_failure_result(
        "fixture", method=MILP, sample_seed=0, tree_init_seed=None,
        max_outer_iterations=10,
        runtime_seconds=0.06, setup_name="fixture_setup",
        faulty_nodes=[1], faulty_edges=[(1, 2)],
        problem=_problem(), info=_FAILURE_INFO,
        error=RuntimeError("MILP produced no feasible solution"),
        outer_state=outer_state,
        results_dir=tmp_path,
    )
    with open(path) as f:
        return json.load(f)


def test_failure_after_augmentation_records_the_state_it_actually_failed_at(tmp_path):
    # The initial problem has P_1 = P_2 = 1, but the failing pass solved
    # against K = 4. Recording only the initial problem would put
    # config.P_1 = 1 next to a status and model size from the K = 4 solve.
    payload = _save_failure(tmp_path, {
        "failed_pass": 3, "K_phi": 4, "K_psi": 4,
        "inner_solve_count": 3, "augmentation_rounds": 3,
        "total_witnesses_added": 6,
        "total_island_witnesses_added": 3,
        "total_component_witnesses_added": 3,
        "pass_history": ({"pass": 0, "K_phi": 1}, {"pass": 1, "K_phi": 2}),
    })
    state = payload["solve_state_at_failure"]

    assert state["failed_pass"] == 3
    assert (state["K_phi"], state["K_psi"]) == (4, 4)
    assert state["completed_inner_solves"] == 3
    assert state["augmentation_rounds"] == 3
    assert state["total_witnesses_added"] == 6

    # The initial config is still recorded, but no longer mistakable for
    # the sample sets the failed pass saw.
    assert payload["config"]["P_1"] == 1
    assert state["K_phi"] != payload["config"]["P_1"]
    assert "INITIAL" in state["config_P_note"]

    # The history behind the failure survives it.
    assert [entry["pass"] for entry in payload["pass_history"]] == [0, 1]


def test_failure_bound_is_scoped_to_the_pass_that_produced_it(tmp_path):
    # A no-incumbent bound applies to that one sampled MILP -- not to the
    # initial problem, and not to the run.
    payload = _save_failure(tmp_path, {
        "failed_pass": 2, "K_phi": 3, "K_psi": 5,
        "inner_solve_count": 2, "augmentation_rounds": 2,
        "total_witnesses_added": 4,
        "total_island_witnesses_added": 2,
        "total_component_witnesses_added": 2,
        "pass_history": (),
    })
    bound = payload["method"]["best_bound"]

    assert bound["value"] == -158.9
    assert "pass 2" in bound["scope"]
    assert "K_phi=3" in bound["scope"] and "K_psi=5" in bound["scope"]


def test_failure_without_outer_state_falls_back_to_the_given_problem(tmp_path):
    # A solver called directly, outside the loop, has no pass to name.
    payload = _save_failure(tmp_path, None)
    state = payload["solve_state_at_failure"]

    assert state["failed_pass"] is None
    assert (state["K_phi"], state["K_psi"]) == (1, 1)   # the problem passed in
    assert payload["pass_history"] == []


# ---------------------------------------------------------------------------
# The run script's failure path actually reaches persistence
# ---------------------------------------------------------------------------

def test_run_script_persists_the_enriched_outer_state(monkeypatch, tmp_path):
    # The wiring between solve()'s enrichment and the saved JSON: the run
    # script must forward exc.outer_state, not just the original problem.
    # Everything below the failure is stubbed, so no solver runs -- the one
    # thing under test is which arguments reach save_failure_result.
    import examples.linear_10node.run_baseline_10node as script
    from baseline.milp import NoIncumbentError

    outer_state = {
        "failed_pass": 2, "K_phi": 22, "K_psi": 22,
        "inner_solve_count": 2, "augmentation_rounds": 2,
        "total_witnesses_added": 4,
        "total_island_witnesses_added": 2,
        "total_component_witnesses_added": 2,
        "pass_history": ({"pass": 0}, {"pass": 1}),
    }
    failure = NoIncumbentError("no feasible solution", info={"status": "TIME_LIMIT"})
    failure.attach_outer_state(**outer_state)

    def exploding_solve(problem, **kwargs):
        raise failure

    captured = {}

    def fake_save(example_name, **kwargs):
        captured["example_name"] = example_name
        captured.update(kwargs)
        return tmp_path / "written.json"

    monkeypatch.setattr(script, "solve", exploding_solve)
    monkeypatch.setattr(script, "save_failure_result", fake_save)

    script.main()

    # The enriched state was forwarded -- this is the regression.
    assert captured["outer_state"] == outer_state
    assert captured["outer_state"]["K_phi"] == 22
    assert captured["info"]["status"] == "TIME_LIMIT"
    assert captured["error"] is failure
    # And the run parameters that make the record reproducible.
    assert captured["example_name"] == "linear_10node_milp"
    assert captured["method"] == script.INNER_METHOD
    assert captured["sample_seed"] == script.SAMPLE_SEED
    assert captured["tree_init_seed"] == script.TREE_INIT_SEED
    assert captured["max_outer_iterations"] == script.MAX_OUTER_ITERATIONS


def test_run_script_failure_path_writes_a_readable_record(monkeypatch, tmp_path):
    # End to end through the REAL save_failure_result, so the wiring is
    # checked against the actual schema rather than a stub's signature.
    import examples.linear_10node.run_baseline_10node as script
    from baseline.milp import NoIncumbentError

    failure = NoIncumbentError(
        "no feasible solution",
        info={"status": "TIME_LIMIT", "best_bound": -12.5, "n_vars": 2511,
              "time_limit": 300.0, "requested_mip_gap": 1e-6},
    )
    failure.attach_outer_state(
        failed_pass=1, K_phi=21, K_psi=21,
        inner_solve_count=1, augmentation_rounds=1,
        total_witnesses_added=2, total_island_witnesses_added=1,
        total_component_witnesses_added=1,
        pass_history=({"pass": 0, "K_phi": 20},),
    )

    def exploding_solve(problem, **kwargs):
        raise failure

    monkeypatch.setattr(script, "solve", exploding_solve)
    monkeypatch.setattr(
        script, "save_failure_result",
        lambda *a, **kw: save_failure_result(*a, **kw, results_dir=tmp_path),
    )

    script.main()

    written = list(tmp_path.glob("*_no_incumbent.json"))
    assert len(written) == 1
    with open(written[0]) as f:
        payload = json.load(f)

    assert payload["outcome"] == "no_incumbent"
    assert payload["solve_state_at_failure"]["failed_pass"] == 1
    assert payload["solve_state_at_failure"]["K_phi"] == 21
    # The initial config is recorded too, and is NOT the failed pass's K.
    assert payload["config"]["P_1"] == 20
    assert "pass 1" in payload["method"]["best_bound"]["scope"]
    assert [e["pass"] for e in payload["pass_history"]] == [0]


@pytest.mark.parametrize(
    "method,tree_init_seed,message",
    [
        (MILP, 0, "initialises no spanning tree"),
        (MILP, 7, "initialises no spanning tree"),
        (LOCAL_SEARCH, None, "initialised from exactly one tree"),
    ],
    ids=["milp_with_zero", "milp_with_seed", "local_search_with_none"],
)
def test_contradictory_method_and_tree_seed_are_refused(
    method, tree_init_seed, message, tmp_path
):
    # The schema says a MILP result carries tree_init_seed=None and a
    # local-search result carries its actual start. Accepting the
    # contradictions would put a fabricated fact into an artifact that
    # outlives the run -- a MILP result naming a start that never happened,
    # or a local-search result hiding the single-start degree of freedom.
    info = _MILP_OPTIMAL if method == MILP else {"reason": "local_optimum",
                                                 "iterations": 3}

    with pytest.raises(ValueError, match=message):
        _save(tmp_path, _result(info), method, tree_init_seed=tree_init_seed)


def test_the_failure_writer_applies_the_same_guard(tmp_path):
    # save_failure_result has its own signature and its own payload, so the
    # guard has to be wired into both writers, not just the success path.
    with pytest.raises(ValueError, match="initialises no spanning tree"):
        save_failure_result(
            "fixture", method=MILP, sample_seed=0, tree_init_seed=0,
            max_outer_iterations=10,
            runtime_seconds=0.06, setup_name="fixture_setup",
            faulty_nodes=[1], faulty_edges=[(1, 2)],
            problem=_problem(), info=_FAILURE_INFO,
            error=RuntimeError("no feasible solution"),
            results_dir=tmp_path,
        )


@pytest.mark.parametrize(
    "method", ["some future solver", "MILP ", "milp", "Local Search"],
    ids=["new_solver", "trailing_space", "wrong_case", "wrong_case_local"],
)
def test_an_unknown_method_is_refused(method, tmp_path):
    # Exemption was worse than rejection: an unknown string does not merely
    # skip validation, it gets MISHANDLED. _method_block dispatches on
    # `method == MILP` and treats everything else as local search, so
    # "MILP " (trailing space) would bypass the seed guard and then be
    # written out with a local-search block -- final_reason and
    # final_iterations read off a MILP info dict that has neither, and no
    # status, bound or gap recorded at all.
    with pytest.raises(ValueError, match="unknown method"):
        _save(tmp_path, _result({"reason": "local_optimum", "iterations": 1}),
              method, tree_init_seed=None)


def test_the_failure_writer_also_refuses_an_unknown_method(tmp_path):
    with pytest.raises(ValueError, match="unknown method"):
        save_failure_result(
            "fixture", method="MILP ", sample_seed=0, tree_init_seed=None,
            max_outer_iterations=10,
            runtime_seconds=0.06, setup_name="fixture_setup",
            faulty_nodes=[1], faulty_edges=[(1, 2)],
            problem=_problem(), info=_FAILURE_INFO,
            error=RuntimeError("no feasible solution"),
            results_dir=tmp_path,
        )
