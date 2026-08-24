"""TEMPORARY(DISABLE_RECONNECTION) -- delete this file to revert.

Tests for the reconnection-disabling test flag. The flag shipped with none
of these, which is why three real problems survived review:

  - the run scripts re-decoded the REAL reconnected set for their report
    while evaluate() optimised against an empty one, so the search
    minimised one function and the report described another (on the
    30-node initial tree: objective 129.433 against a reported 126.105);
  - the skipped component check was recorded as a numeric zero, which
    reads as "checked, found none" rather than "not checked";
  - the MILP baseline picked the flag up through the shared config and
    would have run a hybrid.

Every test here FORCES the flag through `dataclasses.replace` rather than
reading the constant in examples/*/config.py. That constant gets flipped
while an experiment is running, and a test whose meaning depends on its
current value would be testing the experiment rather than the code.
"""

from dataclasses import replace

import pytest

from islanding.cost import (
    f4_reconnection, f5_stability_penalty, f6_reconnection_stability,
    f1_unstable_nodes, f3_operation_edges, F,
)
from islanding.counterexample import find_counterexamples_for_partition
from islanding.evaluate import evaluate, effective_reconnection
from islanding.augmented import augmented_graph
from islanding.search import (
    initial_spanning_tree, decode_islanding, decode_reconnection,
)
from examples.linear_30node.builder import (
    LINEAR_30, build_topology, build_problem,
)
from examples.linear_30node.config import build_config


def _problem(disable):
    """The 30-node problem with the flag forced, whatever the constant says."""
    base = build_problem(LINEAR_30, build_topology(), seed=0, config=build_config())
    return replace(base, config=replace(base.config, disable_reconnection=disable))


# ---------------------------------------------------------------------------
# The rule itself
# ---------------------------------------------------------------------------

def test_effective_reconnection_empties_the_set_only_under_the_flag():
    off = build_config()
    off = replace(off, disable_reconnection=False)
    on = replace(off, disable_reconnection=True)
    edges = {(1, 2), (3, 4)}

    assert effective_reconnection(off, edges) == edges
    assert effective_reconnection(on, edges) == set()


# ---------------------------------------------------------------------------
# The regression: report and objective must agree
# ---------------------------------------------------------------------------

def test_the_reported_total_equals_the_optimised_objective():
    # THE BUG THIS FILE EXISTS FOR. The run scripts compute their report
    # from the decoded partition; evaluate() computes the objective. Under
    # the flag those used different reconnection sets, and the totals
    # disagreed by the whole F4/F6 contribution of edges the objective had
    # ignored. Both now go through effective_reconnection().
    #
    # The initial spanning tree is used deliberately: it decodes to a
    # NON-EMPTY reconnected set, which is what makes the two paths
    # distinguishable. A converged tree may decode to an empty set, in
    # which case they agree by luck and the test proves nothing.
    problem = _problem(disable=True)
    topology = problem.topology
    tree = initial_spanning_tree(augmented_graph(topology))

    stable, unstable = decode_islanding(tree, topology)
    _, _, decoded = decode_reconnection(tree, topology)
    assert decoded, "fixture must decode to a non-empty E^rec to be meaningful"

    optimised = evaluate(tree, problem)

    config = problem.config
    reconnected = effective_reconnection(config, decoded)
    f1 = f1_unstable_nodes(unstable, problem.cost_data)
    f3 = f3_operation_edges(stable, unstable, topology, problem.cost_data, config)
    f4 = f4_reconnection(reconnected, problem.cost_data, config)
    f5 = f5_stability_penalty(
        stable, topology, problem.model, problem.samples_1, config
    )
    f6 = f6_reconnection_stability(
        stable, reconnected, topology, problem.model, problem.samples_2, config
    )
    reported = F(f1, 0.0, f3, f4, f5, f6)

    assert reported == pytest.approx(optimised)


def test_using_the_decoded_set_for_the_report_would_disagree():
    # Pins that the test above is not vacuous: with the decoded set, which
    # is what the run scripts used before the fix, the two DIVERGE.
    problem = _problem(disable=True)
    topology = problem.topology
    tree = initial_spanning_tree(augmented_graph(topology))

    stable, unstable = decode_islanding(tree, topology)
    _, _, decoded = decode_reconnection(tree, topology)
    config = problem.config

    f4_decoded = f4_reconnection(decoded, problem.cost_data, config)
    f6_decoded = f6_reconnection_stability(
        stable, decoded, topology, problem.model, problem.samples_2, config
    )

    assert f4_decoded > 0
    assert f6_decoded > 0
    assert evaluate(tree, problem) != pytest.approx(
        F(
            f1_unstable_nodes(unstable, problem.cost_data), 0.0,
            f3_operation_edges(
                stable, unstable, topology, problem.cost_data, config
            ),
            f4_decoded,
            f5_stability_penalty(
                stable, topology, problem.model, problem.samples_1, config
            ),
            f6_decoded,
        )
    )


# ---------------------------------------------------------------------------
# F4 and F6 under the flag
# ---------------------------------------------------------------------------

def test_f4_and_f6_are_zero_under_the_flag():
    # F6 reaching zero is its GENUINE value, not a mute: with N_i^rec empty
    # for every node, Psi_r loses Parts 1-3 and Part 4 = -(k_1-k_2) W_r is
    # non-positive, which the per-island hinge clips.
    topology = build_topology()
    cluster = frozenset({9, 10, 11, 12, 18, 27, 28})
    stable = [cluster, frozenset(set(topology.nodes()) - cluster - {5, 16})]
    decoded = {(11, 22)}

    on, off = _problem(True), _problem(False)

    assert f4_reconnection(
        effective_reconnection(off.config, decoded), off.cost_data, off.config
    ) > 0
    assert f6_reconnection_stability(
        stable, effective_reconnection(off.config, decoded), topology,
        off.model, off.samples_2, off.config
    ) > 0

    assert f4_reconnection(
        effective_reconnection(on.config, decoded), on.cost_data, on.config
    ) == 0
    assert f6_reconnection_stability(
        stable, effective_reconnection(on.config, decoded), topology,
        on.model, on.samples_2, on.config
    ) == 0


# ---------------------------------------------------------------------------
# The component side is skipped, not merely empty
# ---------------------------------------------------------------------------

def test_the_component_side_is_skipped_under_the_flag():
    # Two singleton islands joined by the FAULTY crossing (10,25), w=-20:
    # each is stable alone, the merged component is not. With the flag off
    # that produces a witness; with it on the check never runs.
    #
    # This is why the skip is necessary rather than cosmetic: a zeroed F6
    # cannot respond to such a witness, so the outer loop would re-find it
    # every pass and run to its cap instead of converging.
    stable = [frozenset({10}), frozenset({25})]
    decoded = {(10, 25)}

    _, without = find_counterexamples_for_partition(
        stable, decoded, _problem(disable=False)
    )
    _, with_flag = find_counterexamples_for_partition(
        stable, decoded, _problem(disable=True)
    )

    assert len(without) == 1
    assert with_flag == []


def test_the_island_side_still_runs_under_the_flag():
    # Only the COMPONENT side is skipped. If the island side were skipped
    # too the loop would converge immediately on anything.
    problem = _problem(disable=True)
    unstable_alone = [frozenset({5})]      # a faulty node, unstable in isolation

    island, component = find_counterexamples_for_partition(
        unstable_alone, set(), problem
    )

    assert len(island) > 0
    assert component == []


# ---------------------------------------------------------------------------
# What the saved result says
# ---------------------------------------------------------------------------

def _save(problem, tmp_path, decoded):
    from islanding.outer_loop import SolveResult
    from utils.save import save_result, LOCAL_SEARCH

    topology = problem.topology
    cluster = frozenset({9, 10, 11, 12, 18, 27, 28})
    rest = frozenset(set(topology.nodes()) - cluster)
    result = SolveResult(
        objective=1.0, inner_solve_count=1, augmentation_rounds=0,
        total_witnesses_added=0, total_island_witnesses_added=0,
        total_component_witnesses_added=0, final_K_phi=20, final_K_psi=20,
        reason="converged", stable_islands=[cluster, rest],
        unstable_islands=[], reconnected_edges=decoded,
        info={"reason": "local_optimum", "iterations": 3},
    )
    disabled = problem.config.disable_reconnection
    path = save_result(
        result, problem, example_name="fixture", method=LOCAL_SEARCH,
        sample_seed=0, tree_init_seed=0, max_outer_iterations=10,
        runtime_seconds=1.0, setup_name="s", faulty_nodes=[], faulty_edges=[],
        island_lambda_min=[1.0, 1.0],
        component_lambda_min=[1.0] * (1 if decoded and not disabled else 2),
        cost_terms={
            "f1": 0.0, "f3": 1.0, "f4": 0.0, "f5": 0.0, "f6": 0.0,
            "economic_cost": 1.0, "total_with_penalties": 1.0,
            "K_phi": 20, "K_psi": 20,
        },
        final_island_witnesses=0,
        final_component_witnesses=None if disabled else 0,
        results_dir=tmp_path,
    )
    import json
    with open(path) as handle:
        return json.load(handle)


def test_saved_json_marks_the_component_check_skipped(tmp_path):
    # null and an explicit "SKIPPED", never 0 -- a numeric zero reads as
    # "checked, found none".
    payload = _save(_problem(disable=True), tmp_path, {(11, 22)})

    assert payload["reconnection_disabled"] is True
    assert payload["final_stability_check"]["component_witnesses"] is None
    assert "SKIPPED" in payload["final_stability_check"]["component_check"]


def test_saved_json_records_what_the_objective_used(tmp_path):
    # The decoded set is kept, but under its own key and plainly labelled
    # as ignored -- `reconnected_edges` is what the objective saw.
    payload = _save(_problem(disable=True), tmp_path, {(11, 22)})

    assert payload["reconnected_edges"] == []
    assert payload["decoded_reconnected_edges_ignored_by_objective"] == [[11, 22]]


def test_saved_json_is_unchanged_when_the_flag_is_off(tmp_path):
    payload = _save(_problem(disable=False), tmp_path, {(11, 22)})

    assert payload["reconnection_disabled"] is False
    assert payload["reconnected_edges"] == [[11, 22]]
    assert payload["final_stability_check"]["component_witnesses"] == 0
    assert payload["final_stability_check"]["component_check"] == "performed"
    assert payload["decoded_reconnected_edges_ignored_by_objective"] == []


def test_saved_reason_scope_stops_claiming_the_oracle_passed(tmp_path):
    # "converged" means something weaker under the flag, and the permanent
    # scope note would overstate it.
    on = _save(_problem(disable=True), tmp_path, set())["method"]["reason_scope"]
    off = _save(_problem(disable=False), tmp_path, set())["method"]["reason_scope"]

    assert "ISLAND side only" in on
    assert "RECONNECTION DISABLED" in on
    assert "ISLAND side only" not in off


# ---------------------------------------------------------------------------
# The MILP baseline refuses
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("case", ["30", "40"])
def test_the_baseline_script_runs_under_the_flag(case, monkeypatch):
    # It used to REFUSE: the MILP did not honour the flag, so a run would
    # have produced a hybrid -- a reconnection-aware objective with the
    # component-side check switched off. Now that baseline/milp.py omits
    # the reconnection machinery, a flagged baseline run is valid and must
    # not be blocked.
    import importlib

    script = importlib.import_module(
        f"examples.linear_{case}node.run_baseline_{case}node"
    )
    problem = script.build_run_problem()
    flagged = replace(
        problem, config=replace(problem.config, disable_reconnection=True)
    )
    monkeypatch.setattr(script, "build_run_problem", lambda: flagged)

    class Reached(Exception):
        pass

    monkeypatch.setattr(
        script, "solve", lambda *a, **k: (_ for _ in ()).throw(Reached())
    )

    # Reaching solve() at all is the assertion: no SystemExit on the way.
    with pytest.raises(Reached):
        script.main()


# ---------------------------------------------------------------------------
# What main() ACTUALLY reports -- not a reconstruction of it
# ---------------------------------------------------------------------------

def test_main_reports_the_objectives_set_not_the_decoded_one(monkeypatch):
    # test_the_reported_total_equals_the_optimised_objective above rebuilds
    # the report's arithmetic inside the test, so it would pass even if a
    # run script went back to feeding the DECODED set to F4/F6. This drives
    # the real main() and captures what reaches save_result().
    #
    # The stubbed solve returns a partition with NON-EMPTY decoded
    # reconnections, which is what makes the two paths distinguishable.
    import examples.linear_30node.run_linear_30node as script
    from islanding.outer_loop import SolveResult

    real = script.build_run_problem()
    flagged = replace(
        real, config=replace(real.config, disable_reconnection=True)
    )
    monkeypatch.setattr(script, "build_run_problem", lambda: flagged)

    topology = flagged.topology
    tree = initial_spanning_tree(augmented_graph(topology))
    stable, unstable = decode_islanding(tree, topology)
    _, _, decoded = decode_reconnection(tree, topology)
    assert decoded, "fixture must decode to a non-empty E^rec to be meaningful"

    monkeypatch.setattr(script, "solve", lambda *a, **k: SolveResult(
        objective=evaluate(tree, flagged),
        inner_solve_count=1, augmentation_rounds=0, total_witnesses_added=0,
        total_island_witnesses_added=0, total_component_witnesses_added=0,
        final_K_phi=flagged.config.P_1, final_K_psi=flagged.config.P_2,
        reason="converged", stable_islands=stable, unstable_islands=unstable,
        reconnected_edges=decoded,
        info={"reason": "local_optimum", "iterations": 1},
    ))

    captured = {}
    monkeypatch.setattr(
        script, "save_result",
        lambda *args, **kwargs: (captured.update(kwargs), "(captured)")[1],
    )

    script.main()

    terms = captured["cost_terms"]
    # F4 and F6 must be zero in the REPORT, not merely inside evaluate().
    assert terms["f4"] == 0
    assert terms["f6"] == 0
    # And the economic figure must be F1 + F3, with nothing subtracted.
    assert terms["economic_cost"] == pytest.approx(terms["f1"] + terms["f3"])
    # The component check was skipped, so its count is absent, not zero.
    assert captured["final_component_witnesses"] is None


# ---------------------------------------------------------------------------
# The MILP honours the flag
# ---------------------------------------------------------------------------

def _built_model(disable):
    from baseline.config import MILPConfig
    from baseline.milp import phi_certificates, psi_certificates, build_model

    problem = _problem(disable)
    phi = phi_certificates(
        problem.topology, problem.model, problem.config, problem.samples_1
    )
    psi = psi_certificates(
        problem.topology, problem.model, problem.config, problem.samples_2
    )
    return build_model(
        problem, MILPConfig(time_limit=60.0, mip_gap=1e-6, verbose=False), phi, psi
    )


def test_the_milp_creates_no_reconnection_variables_under_the_flag():
    # OMISSION, not y == 0 constraints: the variables must not exist. A
    # model that merely pinned them would report the same partition while
    # carrying the whole machinery.
    with_rec = _built_model(disable=False)
    without = _built_model(disable=True)

    assert len(with_rec.y) > 0 and len(without.y) == 0
    for family in ("qr", "qsr", "qrr"):
        assert len(getattr(with_rec, family)) > 0
        assert len(getattr(without, family)) == 0, f"{family} survived the flag"

    # tau is NOT a reconnection family -- it is a product with a_kr and
    # carries Phi^t on the F5 side, so it must be untouched.
    assert len(without.tau) == len(with_rec.tau) > 0

    # And the model really is smaller, not just missing dict entries.
    assert without.model.NumVars < with_rec.model.NumVars
    assert without.model.NumConstrs < with_rec.model.NumConstrs


def test_the_psi_epigraph_is_zero_at_optimality_under_the_flag():
    # Psi_r keeps only -(k_1 - k_2) * W_r, which is nonpositive, so the
    # sigma >= 0 epigraph should clip it to zero -- matching the local
    # search's per-island hinge. Confirmed, not assumed.
    built = _built_model(disable=True)

    nodes = built.order
    cluster = frozenset({9, 10, 11, 12, 18, 27, 28})
    rest = frozenset(set(nodes) - cluster - {5, 16})
    stable = [cluster, rest]
    unstable = [frozenset({5}), frozenset({16})]

    slot_of = {i: min(isl) for isl in stable + unstable for i in isl}
    stable_slots = {min(isl) for isl in stable}
    for (i, r), var in built.a.items():
        value = 1.0 if slot_of[i] == r else 0.0
        var.setAttr("LB", value)
        var.setAttr("UB", value)
    for r, var in built.b.items():
        value = 1.0 if r in stable_slots else 0.0
        var.setAttr("LB", value)
        var.setAttr("UB", value)

    built.model.optimize()

    assert built.model.SolCount > 0
    assert max(var.X for var in built.sigma_psi.values()) == pytest.approx(0.0, abs=1e-9)


def test_the_milp_returns_an_empty_reconnected_set_under_the_flag():
    # The shared outer loop and the save path must see exactly what the
    # local search sees. Run on the 10-node case: the smallest with a MILP
    # baseline, so this stays a fast test.
    from baseline.config import MILPConfig
    from baseline.milp import milp_solver
    from examples.linear_10node.linear_10node import (
        NODE_AND_EDGE_FAULTS, build_topology as t10, build_problem as bp10,
    )
    from examples.linear_10node.config import build_config as c10

    base = bp10(NODE_AND_EDGE_FAULTS, t10(), seed=0, config=c10())
    flagged = replace(
        base, config=replace(base.config, disable_reconnection=True)
    )

    result = milp_solver(
        flagged, MILPConfig(time_limit=60.0, mip_gap=1e-6, verbose=False)
    )

    assert result.reconnected_edges == set()
    assert f4_reconnection(
        result.reconnected_edges, flagged.cost_data, flagged.config
    ) == 0


# ---------------------------------------------------------------------------
# The central equivalence: MILP objective == F on the same partition
# ---------------------------------------------------------------------------

def test_milp_objective_equals_the_cost_functions_under_the_flag(
    synthetic_edge_energy_model
):
    # THE PROPERTY THAT MATTERS. Everything else about the flagged MILP --
    # no y, no q families, tau retained, sigma_Psi zero, empty decoded set
    # -- is machinery. This is the claim those exist to support: on a fixed
    # partition the MILP minimises exactly F1 + F3 + F5 + F6 with an empty
    # E^rec, which is what the local search minimises under the flag. If
    # this fails the two methods are not comparable and every flagged
    # number is meaningless, however tidy the model looks.
    #
    # The SYNTHETIC quadratic edge energy is essential, not decorative. On
    # linear consensus W_ij == 0, so Psi^e is identically zero and the
    # flagged Psi row reduces to a trivially-satisfied 0 >= 0 -- the
    # epigraph clipping would be untested. With W_ij != 0 both Psi^n and
    # Psi^e are genuinely negative, and Phi^e/Phi^t are nonzero too, so the
    # whole certificate path is live.
    #
    # 10-node case, so the pinned solve stays fast.
    from baseline.config import MILPConfig
    from baseline.milp import phi_certificates, psi_certificates, build_model
    from examples.linear_10node.linear_10node import (
        NODE_AND_EDGE_FAULTS, build_topology as topo10,
        build_problem as problem10,
    )
    from examples.linear_10node.config import build_config as config10

    topology = topo10()
    problem = problem10(
        NODE_AND_EDGE_FAULTS, topology, seed=0, config=config10()
    )
    problem = replace(
        problem,
        model=synthetic_edge_energy_model(problem.model),
        config=replace(problem.config, disable_reconnection=True),
    )
    config = problem.config

    stable = [frozenset({1, 2, 3}), frozenset({5, 8, 9}), frozenset({6, 7, 10})]
    unstable = [frozenset({4})]

    phi = phi_certificates(topology, problem.model, config, problem.samples_1)
    psi = psi_certificates(topology, problem.model, config, problem.samples_2)
    built = build_model(
        problem, MILPConfig(time_limit=300.0, mip_gap=1e-9, verbose=False),
        phi, psi,
    )
    assert built.y == {}, "the flagged model must carry no y variables"

    slot_of = {i: min(island) for island in stable + unstable for i in island}
    stable_slots = {min(island) for island in stable}
    for (i, r), var in built.a.items():
        value = 1.0 if slot_of[i] == r else 0.0
        var.setAttr("LB", value)
        var.setAttr("UB", value)
    for r, var in built.b.items():
        value = 1.0 if r in stable_slots else 0.0
        var.setAttr("LB", value)
        var.setAttr("UB", value)
    built.model.optimize()
    assert built.model.SolCount > 0, "pinned partition is infeasible"

    f1 = f1_unstable_nodes(unstable, problem.cost_data)
    f3 = f3_operation_edges(
        stable, unstable, topology, problem.cost_data, config
    )
    f4 = f4_reconnection(set(), problem.cost_data, config)
    f5 = f5_stability_penalty(
        stable, topology, problem.model, problem.samples_1, config
    )
    f6 = f6_reconnection_stability(
        stable, set(), topology, problem.model, problem.samples_2, config
    )

    # Hand-verified: 83.4710688605915 from both sides, with F5 = 72.471
    # carrying the synthetic edge energy and F4 = F6 = 0.
    assert built.model.ObjVal == pytest.approx(
        F(f1, 0.0, f3, f4, f5, f6), rel=1e-9, abs=1e-9
    )
    assert built.model.ObjVal == pytest.approx(83.4710688605915, abs=1e-9)
    assert f4 == 0 and f6 == 0
    # ...and the F5 side is genuinely exercised, or the equality above
    # would hold for trivial reasons.
    assert f5 > 0


# ---------------------------------------------------------------------------
# The baseline scripts' report path
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("case", ["30", "40"])
def test_baseline_main_reports_the_skipped_check_and_no_reconnection(
    case, monkeypatch
):
    # The local-search equivalent of this test existed; this one did not,
    # which is why the baseline scripts kept printing "Final component
    # witnesses: 0" and passing 0 to save_result() -- producing JSON that
    # said component_witnesses 0 beside component_check "SKIPPED".
    import importlib

    from islanding.outer_loop import SolveResult

    script = importlib.import_module(
        f"examples.linear_{case}node.run_baseline_{case}node"
    )
    real = script.build_run_problem()
    flagged = replace(
        real, config=replace(real.config, disable_reconnection=True)
    )
    monkeypatch.setattr(script, "build_run_problem", lambda: flagged)

    topology = flagged.topology
    nodes = sorted(topology.nodes())
    stable = [frozenset(nodes[: len(nodes) // 2]), frozenset(nodes[len(nodes) // 2:])]
    # A NON-EMPTY decoded set, so a script using it instead of the
    # objective's would be caught.
    decoded = {tuple(sorted(edge)) for edge in list(topology.edges())[:1]}

    monkeypatch.setattr(script, "solve", lambda *a, **k: SolveResult(
        objective=1.0, inner_solve_count=1, augmentation_rounds=0,
        total_witnesses_added=0, total_island_witnesses_added=0,
        total_component_witnesses_added=0,
        final_K_phi=flagged.config.P_1, final_K_psi=flagged.config.P_2,
        reason="converged", stable_islands=stable, unstable_islands=[],
        reconnected_edges=decoded,
        info={"status": "OPTIMAL", "requested_mip_gap": 1e-6,
              "attained_mip_gap": 0.0, "best_bound": 1.0, "time_limit": 60.0,
              "solve_time": 0.1, "n_vars": 10, "n_constrs": 20},
    ))

    captured = {}
    monkeypatch.setattr(
        script, "save_result",
        lambda *args, **kwargs: (captured.update(kwargs), "(captured)")[1],
    )

    script.main()

    terms = captured["cost_terms"]
    assert terms["f4"] == 0
    assert terms["f6"] == 0
    assert terms["economic_cost"] == pytest.approx(terms["f1"] + terms["f3"])
    # The contradiction this test exists for: None, never 0.
    assert captured["final_component_witnesses"] is None
