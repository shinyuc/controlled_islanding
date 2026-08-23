"""Each case's two run scripts must pose the SAME problem to both solvers.

The local-search script and the MILP baseline script for a case exist to be
compared line for line. That comparison is only meaningful if the two hand
their solvers an identical `Problem` and an identical outer-loop budget --
if the sample sets differ by even one draw, the "optimality gap" between
the two reports is partly a difference of instance.

The baseline scripts were produced by copying their local-search
counterparts, and their docstrings claim the two are kept in step. Nothing
enforced that: once committed they are independent files, and a later edit
to one (a retuned alpha, a different seed, a changed iteration budget)
would silently decouple the pair while both scripts kept running fine and
kept producing comparable-looking reports. These tests are what makes the
claim checkable.

These tests go through the scripts' OWN construction path -- the
`build_run_problem()` each one exports and `main()` actually calls -- and
capture the real `solve()` invocation. An earlier version rebuilt the
Problem from SETUP/SAMPLE_SEED/build_config() inside the test, which
compared two reimplementations: it would have passed while main() hardcoded
a different seed, called a different config builder, mutated the Problem, or
passed a literal to solve() that disagreed with MAX_OUTER_ITERATIONS.

Importing a run script is safe: every one guards its work behind
`if __name__ == "__main__"`, so import evaluates only imports and module
constants.
"""

import inspect
from dataclasses import fields

import numpy as np
import pytest

import islanding.local_search as local_search_module
from baseline.milp import milp_solver
from examples.linear_10node import run_linear_10node, run_baseline_10node
from examples.linear_20node import run_linear_20node, run_baseline_20node
from examples.linear_30node import run_linear_30node, run_baseline_30node
from examples.linear_40node import run_linear_40node, run_baseline_40node
from examples.linear_50node import run_linear_50node


# (case name, local-search script, MILP baseline script)
PAIRS = [
    ("10node", run_linear_10node, run_baseline_10node),
    ("20node", run_linear_20node, run_baseline_20node),
    ("30node", run_linear_30node, run_baseline_30node),
    ("40node", run_linear_40node, run_baseline_40node),
]
PAIR_IDS = [pair[0] for pair in PAIRS]

# Every script individually, for the checks that are about one script's
# internal consistency rather than a pair's agreement.
# Every script, INCLUDING the two with no MILP counterpart: the
# execution-path checks below are about one script's internal consistency
# (does main() solve what its builder returns, with the constants it
# advertises), which applies just as much where there is nothing to pair
# with. Only PAIRS above is restricted to cases that have both methods.
SCRIPTS = [
    ("10node-local", run_linear_10node),
    ("10node-milp", run_baseline_10node),
    ("20node-local", run_linear_20node),
    ("20node-milp", run_baseline_20node),
    ("30node-local", run_linear_30node),
    ("30node-milp", run_baseline_30node),
    ("40node-local", run_linear_40node),
    ("40node-milp", run_baseline_40node),
    ("50node-local", run_linear_50node),
]
SCRIPT_IDS = [entry[0] for entry in SCRIPTS]
LOCAL_SCRIPTS = [entry for entry in SCRIPTS if entry[0].endswith("local")]
LOCAL_IDS = [entry[0] for entry in LOCAL_SCRIPTS]


def _problem_of(script):
    """The Problem that script solves, via its own build_run_problem().

    Not a reimplementation: this is the same callable main() invokes, so a
    drift inside it is a drift here.
    """
    return script.build_run_problem()



def _model_fingerprint(problem):
    """What a Problem's SystemModel computes on one fixed probe state.

    Stands in for equality, which a bundle of closures cannot support.
    The probe is deterministic and node-dependent, so two models built
    from DIFFERENT node/edge weights -- the realistic way a pair drifts,
    e.g. one script pointed at another builder -- produce different
    fingerprints, while two models built from the same weights agree
    exactly.
    """
    model, topology = problem.model, problem.topology
    nodes = sorted(topology.nodes())
    d = problem.samples_1.shape[2]
    x = {
        node: np.full(d, 1.0 + 0.5 * index)
        for index, node in enumerate(nodes)
    }

    out = [("inflation", float(model.inflation(topology)))]
    for i in nodes:
        out.append(("f_i", i, tuple(np.ravel(model.f_i(i, x[i])))))
        out.append(("W_i", i, float(model.W_i(i, x[i]))))
        out.append(("grad_W_i", i, tuple(np.ravel(model.grad_W_i(i, x[i])))))
    for u, v in sorted(tuple(sorted(edge)) for edge in topology.edges()):
        out.append(("f_ij", u, v, tuple(np.ravel(model.f_ij(u, v, x[u], x[v])))))
        out.append(("f_ji", u, v, tuple(np.ravel(model.f_ij(v, u, x[v], x[u])))))
        out.append(("W_ij", u, v, float(model.W_ij(u, v, x[u], x[v]))))
        grad_u, grad_v = model.grad_W_ij(u, v, x[u], x[v])
        out.append(("grad_W_ij", u, v,
                    tuple(np.ravel(grad_u)), tuple(np.ravel(grad_v))))
    return out


class _StopBeforeSolve(Exception):
    """Aborts main() at the solve call, once its arguments are captured."""


def _capture_solve_call(script, monkeypatch):
    """Run main() up to solve(), returning what it was actually handed.

    This is the only way to check the arguments main() really passes --
    the module constant and the value at the call site are different
    facts, and a literal at the call site is one of the drift modes these
    tests exist to catch. The solve itself never runs: the stub raises as
    soon as it has recorded its arguments, which also keeps these tests
    fast enough to sit in the normal suite.
    """
    captured = {}

    def fake_solve(problem, **kwargs):
        captured["problem"] = problem
        captured["kwargs"] = kwargs
        raise _StopBeforeSolve()

    monkeypatch.setattr(script, "solve", fake_solve)
    with pytest.raises(_StopBeforeSolve):
        script.main()
    return captured


@pytest.mark.parametrize("case,local,milp", PAIRS, ids=PAIR_IDS)
def test_both_scripts_use_the_same_setup(case, local, milp):
    # Comparing two methods on different fault scenarios is meaningless,
    # and nothing in either script would notice.
    assert local.SETUP is milp.SETUP
    assert local.SETUP.name == milp.SETUP.name


@pytest.mark.parametrize("case,local,milp", PAIRS, ids=PAIR_IDS)
def test_both_scripts_build_the_same_topology(case, local, milp):
    a, b = local.build_topology(), milp.build_topology()

    assert sorted(a.nodes()) == sorted(b.nodes())
    assert sorted(map(sorted, a.edges())) == sorted(map(sorted, b.edges()))


@pytest.mark.parametrize("case,local,milp", PAIRS, ids=PAIR_IDS)
def test_both_scripts_use_the_same_cost_data(case, local, milp):
    a, b = _problem_of(local), _problem_of(milp)

    assert a.cost_data.node_costs == b.cost_data.node_costs
    assert a.cost_data.edge_costs == b.cost_data.edge_costs


@pytest.mark.parametrize("case,local,milp", PAIRS, ids=PAIR_IDS)
def test_both_scripts_use_an_identical_algorithm_config(case, local, milp):
    # Field by field rather than == on the dataclass, so a mismatch names
    # the parameter that drifted instead of dumping two long repr()s. A
    # retuned alpha in one config and not the other is the likely way this
    # breaks, and it would change what each solver is minimising.
    a, b = local.build_config(), milp.build_config()

    for field in fields(a):
        assert getattr(a, field.name) == getattr(b, field.name), (
            f"{case}: AlgorithmConfig.{field.name} differs between the "
            f"local-search and MILP scripts"
        )


@pytest.mark.parametrize("case,local,milp", PAIRS, ids=PAIR_IDS)
def test_both_scripts_share_the_outer_iteration_budget(case, local, milp):
    # The witness-growing schedule is part of what is being compared, so a
    # different cap would give one method more chances to improve.
    assert local.MAX_OUTER_ITERATIONS == milp.MAX_OUTER_ITERATIONS


@pytest.mark.parametrize("case,local,milp", PAIRS, ids=PAIR_IDS)
def test_both_scripts_draw_bit_identical_sample_sets(case, local, milp):
    # The strictest of these, and the one most likely to catch a real
    # divergence: CONTENTS, not just shapes. Equal P_1/P_2 with different
    # draws would leave every report looking normal while the two methods
    # optimised different sampled objectives -- exactly the failure the
    # side-by-side comparison cannot survive.
    a, b = _problem_of(local), _problem_of(milp)

    assert local.SAMPLE_SEED == milp.SAMPLE_SEED
    np.testing.assert_array_equal(a.samples_1, b.samples_1)
    np.testing.assert_array_equal(a.samples_2, b.samples_2)
    # P must track the arrays, or f5/f6 average by the wrong denominator.
    assert a.config.P_1 == len(a.samples_1) == len(b.samples_1)
    assert a.config.P_2 == len(a.samples_2) == len(b.samples_2)


@pytest.mark.parametrize("case,local,milp", PAIRS, ids=PAIR_IDS)
def test_the_scripts_disagree_only_where_the_methods_genuinely_differ(
    case, local, milp
):
    # The inverse guard: the pair must NOT be identical in the places that
    # identify the method, or one of them is not doing what it claims.
    assert local.INNER_METHOD != milp.INNER_METHOD

    # Only the local search initialises a tree, so only it has a start to
    # record. None here is the honest value, not a missing field.
    assert local.TREE_INIT_SEED is not None
    assert milp.TREE_INIT_SEED is None


# ---------------------------------------------------------------------------
# What main() ACTUALLY does -- not what the module constants imply
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name,script", SCRIPTS, ids=SCRIPT_IDS)
def test_main_solves_exactly_what_build_run_problem_returned(
    name, script, monkeypatch
):
    # IDENTITY, not field-by-field equality. Comparing fields only covers
    # the fields someone thought to list -- the previous version checked
    # samples, config, cost data and topology NODES, so a substituted
    # topology with the same nodes but different EDGES, or a different
    # SystemModel, would have gone straight through.
    #
    # Pinning `is` proves main() hands the solver the builder's result
    # unaltered, whatever a Problem happens to contain now or later. The
    # sentinel is a real Problem so main() can run up to the solve.
    sentinel = script.build_run_problem()
    monkeypatch.setattr(script, "build_run_problem", lambda: sentinel)

    # Snapshot the arrays too: `is` cannot see an IN-PLACE edit of a
    # mutable field, which is the one substitution-free way to change what
    # gets solved.
    samples_1 = sentinel.samples_1.copy()
    samples_2 = sentinel.samples_2.copy()

    captured = _capture_solve_call(script, monkeypatch)

    assert captured["problem"] is sentinel, (
        f"{name}: main() solved a Problem other than the one "
        "build_run_problem() returned"
    )
    np.testing.assert_array_equal(captured["problem"].samples_1, samples_1)
    np.testing.assert_array_equal(captured["problem"].samples_2, samples_2)


@pytest.mark.parametrize("name,script", SCRIPTS, ids=SCRIPT_IDS)
def test_main_passes_the_module_constant_as_the_outer_cap(name, script, monkeypatch):
    # The constant and the value at the call site are different facts. A
    # literal passed to solve() would leave MAX_OUTER_ITERATIONS looking
    # authoritative while the run used something else -- and the pairwise
    # budget test above would still pass.
    captured = _capture_solve_call(script, monkeypatch)

    assert captured["kwargs"]["max_outer_iterations"] == script.MAX_OUTER_ITERATIONS


@pytest.mark.parametrize("case,local,milp", PAIRS, ids=PAIR_IDS)
def test_both_scripts_solve_bit_identical_problems(case, local, milp):
    # The pairwise claim, made against what the scripts' own builders
    # produce rather than against a test-side reconstruction.
    a = _problem_of(local)
    b = _problem_of(milp)

    np.testing.assert_array_equal(a.samples_1, b.samples_1)
    np.testing.assert_array_equal(a.samples_2, b.samples_2)

    # EDGES as well as nodes: two topologies over the same node set but
    # different edges are different instances, and every cost term and
    # every certificate constant depends on the edge set.
    assert sorted(map(sorted, a.topology.edges())) == sorted(
        map(sorted, b.topology.edges())
    )

    # The dynamics too. A different SystemModel changes F5/F6 and every
    # Phi/Psi constant while leaving the topology, costs and samples
    # identical -- invisible to everything above.
    #
    # Compared BEHAVIOURALLY: a SystemModel is a bundle of closures, and
    # two calls to linear_consensus_model() with the same weights produce
    # equal models but unequal function objects, so `==` on the fields
    # fails on a correct pair. Evaluating both on one deterministic probe
    # state is what actually answers "do these describe the same system".
    assert _model_fingerprint(a) == _model_fingerprint(b), (
        f"{case}: the two scripts' SystemModels disagree on the probe "
        "state -- different node or edge weights"
    )


def _tree_seed_local_search_actually_uses(problem, monkeypatch):
    """The seed initial_spanning_tree really receives inside local_search.

    Behavioural rather than a comparison of two constants. The recorder's
    own default is read off the LIBRARY function's signature, so "called
    with no seed argument" resolves to whatever the library default is at
    the time -- not to a value duplicated in this test. A local_search that
    started passing an explicit seed, or a changed library default, both
    show up here.

    Raises out of local_search as soon as the seed is known: the search
    itself is irrelevant to the question and would be slow.
    """
    real_default = inspect.signature(
        local_search_module.initial_spanning_tree
    ).parameters["seed"].default
    captured = {}

    def recorder(gbar, seed=real_default):
        captured["seed"] = seed
        raise _StopBeforeSolve()

    monkeypatch.setattr(local_search_module, "initial_spanning_tree", recorder)
    with pytest.raises(_StopBeforeSolve):
        local_search_module.local_search(problem)
    return captured["seed"]


@pytest.mark.parametrize("name,script", LOCAL_SCRIPTS, ids=LOCAL_IDS)
def test_recorded_tree_seed_is_the_one_local_search_will_use(
    name, script, two_node_problem, monkeypatch
):
    # The saved result labels every local-search figure with
    # TREE_INIT_SEED, but nothing PASSES that constant anywhere:
    # local_search initialises its own tree. So the constant is a claim
    # about the library's behaviour, and this checks it against the
    # behaviour rather than against another constant.
    #
    # A 2-node problem is enough -- which seed local_search uses is a
    # property of local_search, not of the instance.
    actual = _tree_seed_local_search_actually_uses(two_node_problem, monkeypatch)

    assert script.TREE_INIT_SEED == actual, (
        f"{name}: results are labelled tree_init_seed="
        f"{script.TREE_INIT_SEED}, but local_search initialises from "
        f"seed {actual}"
    )


# ---------------------------------------------------------------------------
# Which inner solver main() actually hands to solve()
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("case,local,milp", PAIRS, ids=PAIR_IDS)
def test_the_baseline_script_really_passes_the_milp(case, local, milp, monkeypatch):
    # The gap every other test here leaves open. A baseline script that
    # regressed from
    #     solve(problem, inner_solver=partial(milp_solver, ...))
    # to
    #     solve(problem)
    # would run the LOCAL SEARCH while INNER_METHOD still said "MILP",
    # every parity check still passed, and the saved result recorded a
    # method block for a solver that never ran.
    captured = _capture_solve_call(milp, monkeypatch)
    inner = captured["kwargs"].get("inner_solver")

    assert inner is not None, f"{case}: baseline script passed no inner_solver"
    assert inner.func is milp_solver, (
        f"{case}: inner_solver wraps {inner.func!r}, not milp_solver"
    )


@pytest.mark.parametrize("case,local,milp", PAIRS, ids=PAIR_IDS)
def test_the_milp_is_bound_with_this_cases_solver_config(case, local, milp, monkeypatch):
    # The config must be the one this case declares, compared field by
    # field rather than by identity: build_milp_config() returns a fresh
    # instance per call, so `is` would fail on a correct script, and `==`
    # on the dataclass would not name which field drifted.
    captured = _capture_solve_call(milp, monkeypatch)
    bound = captured["kwargs"]["inner_solver"].keywords["milp_config"]
    expected = milp.build_milp_config()

    for field in fields(expected):
        assert getattr(bound, field.name) == getattr(expected, field.name), (
            f"{case}: solve() got MILPConfig.{field.name}="
            f"{getattr(bound, field.name)!r}, but build_milp_config() "
            f"declares {getattr(expected, field.name)!r}"
        )


@pytest.mark.parametrize("name,script", LOCAL_SCRIPTS, ids=LOCAL_IDS)
def test_local_search_scripts_pass_no_inner_solver_override(
    name, script, monkeypatch
):
    # The mirror of the above: these scripts must take solve()'s DEFAULT.
    # Passing an explicit inner_solver would not be wrong in itself, but it
    # would mean the script no longer runs what INNER_METHOD claims unless
    # someone checked what it passed -- so the absence is the thing to pin.
    captured = _capture_solve_call(script, monkeypatch)

    assert "inner_solver" not in captured["kwargs"], (
        f"{name}: a local-search script passed an explicit inner_solver"
    )
