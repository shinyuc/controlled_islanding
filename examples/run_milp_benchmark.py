"""Paired proposed-vs-MILP benchmark for N = 10, 20, 30, 40.

Experiment 3, "Comparison with the MILP Benchmark". For each network
size, the proposed inner solver and the MILP inner solver are run on the
SAME numerical instance, sequentially, on this machine.

WHAT IS HELD IDENTICAL between the two runs of one size: the frozen
topology and its node coordinates, the nominal and faulted weights, the
fault configuration, the whole AlgorithmConfig (alphas, betas, k's, P's,
epsilon, tolerances), the sample seed, the counterexample oracle, the
eigenvalue stability test, the stopping criterion and the outer cap. The
only intended difference is which callable solve() uses as its inner
solver -- and TREE_INIT_SEED, which is not a free parameter: the MILP
initialises no spanning tree, so there is nothing for a tree seed to
control and recording 0 for it would describe a start that never
happened.

FRESH MUTABLE STATE PER RUN. Each run builds its own Topology and its own
Problem, hence its own SystemModel. That is not tidiness: Topology owns
an induced-edge cache that every island evaluated during a search
populates (islanding/topology.py:19, read at islanding/cost.py:243), so a
shared object would let the second run of a pair start warm and make the
measured runtimes depend on execution order. An earlier sweep was caught
by exactly this. The model-keyed caches -- island_quadratic_form's
closure, and the two verification registries, which are
WeakKeyDictionaries keyed by the model -- start cold automatically
because build_problem constructs a new model each time.

The one process-global cache is cost._zero_arrays, keyed by (shape,
dtype) rather than by model. It holds two small read-only arrays and is
filled by a single np.zeros_like, so only the first run of the process
would pay for it. It is PRE-WARMED below rather than reasoned about, so
neither method carries that cost.

NOT RUN CONCURRENTLY. The eight runs are sequential. Wall clock is the
reported quantity and concurrent execution would contaminate it. Those
timings are of the proposed method as implemented and of Gurobi at its
default automatic thread setting; neither side is thread-pinned, so they
are not a like-for-like comparison of computational resources.

WHAT reason="local_optimum" MEANS in a per-run file. The local search
takes a swap only when objective > (1 + epsilon) * best_candidate, a
RELATIVE test at epsilon = 0.01 -- so it stops when no swap improves by
more than about 1%, which is weaker than "no improving swap exists". It
is not a certificate of exact local optimality. The OUTER reason,
"converged", is a different statement again: it means the returned
partition survived the counterexample oracle.

NOTATION, and WHICH SAMPLE BASIS EACH QUANTITY USES. The paper's F1 is
the code's economic cost, code_F1 + code_F3 - code_F4, recorded here as
`paper_F1`. The code's F5 and F6 are the paper's F2 and F3, the sampled
stability penalties.

    paper_F1                  no sample set enters it. Comparable
                              unconditionally.
    code_F5, code_F6,         evaluated by evaluate_partition on
    total_with_penalties      problem.samples_1 / samples_2 -- the COMMON
                              INITIAL arrays, K_Phi = K_Psi = 20, which
                              are identical within a pair because both
                              runs are built from one setup at one sample
                              seed. Comparable WITHIN a pair, as
                              diagnostic scores on a shared basis.
    objective                 each method's OWN FINAL augmented sample
                              sets, whose K differs between the two runs
                              of a pair. NOT guaranteed comparable, which
                              is why no table reports it.

F5 and F6 are kept out of the paper table not because they cannot be
compared -- they can, within a pair -- but because that table reports the
sample-INDEPENDENT outcome, and a penalty scored against one draw of
20+20 samples is a diagnostic about the returned partition rather than a
property of it. They are in every result file.

How to run
----------
    python -m examples.run_milp_benchmark

Roughly forty minutes for all eight. Progress is printed as each run
starts and finishes. Writes one full result per (N, method) plus one
combined summary, all timestamped; nothing is overwritten.
"""

import importlib
import json
import platform
import sys
import time
from datetime import datetime
from functools import partial

import numpy as np

import baseline.milp as milp_module
from baseline.milp import milp_solver
from examples.common import faulty_nodes, faulty_edges
from islanding.counterexample import find_counterexamples_for_partition
from islanding.cost import (
    f1_unstable_nodes, f3_operation_edges, f4_reconnection,
    f5_stability_penalty, f6_reconnection_stability, F, _shared_zero,
)
from islanding.local_search import local_search
from islanding.outer_loop import solve
from islanding.search import build_reconnected_components, DEFAULT_TREE_SEED
from utils.save import save_result, REPO_ROOT

# (N, package, Setup attribute). Every case's constants are read from its
# own modules rather than restated here, so this file cannot drift from
# what the individual run scripts do.
CASES = [
    # The GENERATED 10-node case, not examples/linear_10node/. That one
    # is hand-specified, so putting it beside three random-geometric
    # instances in a scaling table would imply a homogeneity that does not
    # hold. linear_10node_rgg is drawn by the same rule as the other three.
    (10, "linear_10node_rgg", "LINEAR_10_RGG"),
    (20, "linear_20node", "LINEAR_20"),
    (30, "linear_30node", "LINEAR_30"),
    (40, "linear_40node", "LINEAR_40"),
]

SAMPLE_SEED = 0

# UNIFORM ACROSS N, at the explicit request of the experiment design. It
# is a real change from what the individual run scripts use for N =
# 20/30/40, which set 10; the 10-node RGG scripts now set 20 to match.
# Within a pair the cap was already identical, which is what the
# comparison requires; making it identical ACROSS sizes as well means a
# runtime or solve-count difference between two values of N cannot be a
# difference of budget.
#
# Raising a cap can only ever admit more outer passes, so it cannot change
# a run that already converged strictly below the old value; it can only
# rescue one that was truncated by it. Every run is checked for
# convergence regardless -- see the validity gate -- so a cap that still
# binds would be reported, not absorbed.
OUTER_CAP = 20


def _case_modules(package):
    """The builder, config and topology modules for one case."""
    try:
        builder = importlib.import_module(f"examples.{package}.builder")
    except ModuleNotFoundError:
        builder = importlib.import_module(f"examples.{package}.{package}")
    config = importlib.import_module(f"examples.{package}.config")
    return builder, config


def _topology_metadata(package, builder):
    """Plane side length and topology seed, where the case has them.

    Every case in CASES is generated, so every one has both. The branch
    below survives for a case that might not -- examples/linear_10node/ is
    hand-specified and would take it -- and reporting "n/a" is the honest
    answer there rather than inventing a plane and a seed.
    """
    import pathlib
    import re

    try:
        generator = importlib.import_module(f"examples.{package}.generate_{package}")
    except ModuleNotFoundError:
        return {"plane_size": None, "topology_seed": None,
                "topology_source": "hand-specified, not generated"}

    plane = getattr(generator, "PLANE_SIZE", None)
    artefact = importlib.import_module(f"examples.{package}.{package}_topology")
    header = pathlib.Path(artefact.__file__).read_text()[:400]
    match = re.search(r"seed=(\d+)", header)
    return {
        "plane_size": plane,
        # The seed RECORDED IN THE ARTEFACT, which is the seed that
        # actually produced it -- not the generator's DEFAULT_SEED, which
        # the connectivity-retry loop may have advanced past.
        "topology_seed": int(match.group(1)) if match else None,
        "edge_radius": getattr(generator, "EDGE_RADIUS", None),
        "topology_source": f"generate_{package}.py, frozen artefact",
    }


def timed(record, solver):
    """`solver`, wrapped so each call's wall clock lands in `record`.

    Brackets the inner solve alone: solve() runs the counterexample
    search, the augmentation and the bookkeeping outside this call, so
    they are excluded by construction rather than by subtraction.
    """
    def wrapped(problem, **kwargs):
        start = time.perf_counter()
        try:
            return solver(problem, **kwargs)
        finally:
            record.append(time.perf_counter() - start)
    return wrapped


def capture_model_sizes(record):
    """Patch build_model to record each MILP's size as it is built.

    n_vars and n_constrs reach pass_history through the solver's info
    dict, but NumBinVars does not, and the binary count is the size figure
    that tracks branch-and-bound difficulty most closely. Rather than
    change baseline/milp.py's return contract for a diagnostic, the
    builder is wrapped here for the duration of one run and restored
    afterwards. Returns a callable that undoes the patch.
    """
    original = milp_module.build_model

    def wrapped(problem, milp_config, phi_cert, psi_cert):
        built = original(problem, milp_config, phi_cert, psi_cert)
        model = built.model
        model.update()
        record.append({
            "K_phi": len(problem.samples_1),
            "K_psi": len(problem.samples_2),
            "n_vars": int(model.NumVars),
            "n_binary_vars": int(model.NumBinVars),
            "n_constrs": int(model.NumConstrs),
        })
        return built

    milp_module.build_model = wrapped
    return lambda: setattr(milp_module, "build_model", original)


def evaluate_partition(result, problem, topology, setup, config):
    """Cost terms and eigenvalue checks for a returned partition."""
    from examples.common import build_M

    stable = result.stable_islands
    unstable = result.unstable_islands
    reconnected = result.reconnected_edges
    components = build_reconnected_components(stable, reconnected, topology)

    def induced(nodes):
        return [(u, v) for u, v in topology.edges() if u in nodes and v in nodes]

    def lam(nodes, edges):
        return float(np.linalg.eigvalsh(
            build_M(sorted(nodes), edges, setup.node_weights, setup.edge_weights)
        ).min())

    island_lambda = [lam(sorted(i), induced(i)) for i in stable]
    component_lambda = [lam(sorted(c.nodes()), list(c.edges())) for c in components]

    f1 = f1_unstable_nodes(unstable, problem.cost_data)
    f3 = f3_operation_edges(stable, unstable, topology, problem.cost_data, config)
    f4 = f4_reconnection(reconnected, problem.cost_data, config)
    f5 = f5_stability_penalty(
        stable, topology, problem.model, problem.samples_1, config)
    f6 = f6_reconnection_stability(
        stable, reconnected, topology, problem.model, problem.samples_2, config)

    island_witnesses, component_witnesses = find_counterexamples_for_partition(
        stable, reconnected, problem)

    # The eigenvalue test the oracle applies: an eigenvalue at or below
    # stability_tolerance yields a counterexample, so passing means
    # strictly above it. Checked against the tolerance rather than against
    # zero, so the criterion here is the same one the solver used.
    tol = config.stability_tolerance
    return {
        "stable_islands": [sorted(i) for i in stable],
        "unstable_islands": [sorted(i) for i in unstable],
        "n_stable_islands": len(stable),
        "n_unstable_islands": len(unstable),
        "reconnected_edges": sorted(list(e) for e in reconnected),
        "reconnected_components": [sorted(c.nodes()) for c in components],
        "island_lambda_min": island_lambda,
        "component_lambda_min": component_lambda,
        "stability_tolerance": tol,
        "all_stable_islands_pass": all(v > tol for v in island_lambda),
        "all_components_pass": all(v > tol for v in component_lambda),
        "code_F1": f1, "code_F3": f3, "code_F4": f4,
        "code_F5": f5, "code_F6": f6,
        # THE PAPER'S F1. Named unambiguously because the code also has a
        # variable called F1 that is only one of its three terms.
        "paper_F1": F(f1, 0.0, f3, f4),
        "paper_F1_definition": "economic_cost = code_F1 + code_F3 - code_F4",
        "total_with_penalties": F(f1, 0.0, f3, f4, f5, f6),
        "initial_K_phi": len(problem.samples_1),
        "initial_K_psi": len(problem.samples_2),
        "final_island_witnesses": len(island_witnesses),
        "final_component_witnesses": len(component_witnesses),
        "_island_lambda": island_lambda,
        "_component_lambda": component_lambda,
    }


def run_case(size, package, setup_name, method, timestamp):
    """One (N, method) run, on freshly constructed mutable state."""
    builder, config_module = _case_modules(package)
    setup = getattr(builder, setup_name)
    config = config_module.build_config()
    cap = OUTER_CAP

    # FRESH Topology, built here and used by nothing else -- see the
    # module docstring. builder.build_topology() constructs a new object
    # on every call.
    topology = builder.build_topology()
    assert not topology._induced_edge_cache, "a fresh Topology must start cold"

    problem = builder.build_problem(setup, topology, config, seed=SAMPLE_SEED)

    solve_times = []
    model_sizes = []
    restore = None
    if method == "MILP":
        milp_config = config_module.build_milp_config()
        inner = timed(solve_times, partial(milp_solver, milp_config=milp_config))
        restore = capture_model_sizes(model_sizes)
        tree_seed = None
    else:
        milp_config = None
        inner = timed(solve_times, local_search)
        tree_seed = DEFAULT_TREE_SEED

    print(f"  [{method:>13}] N={size}: starting "
          f"(cap {cap}, K_phi=K_psi={config.P_1})", flush=True)
    start = time.perf_counter()
    try:
        result = solve(problem, max_outer_iterations=cap, inner_solver=inner)
    finally:
        if restore is not None:
            restore()
    elapsed = time.perf_counter() - start

    assert len(solve_times) == result.inner_solve_count, (
        "inner-solve timing did not bracket the solver one-to-one")

    metrics = evaluate_partition(result, problem, topology, setup, config)
    island_lambda = metrics.pop("_island_lambda")
    component_lambda = metrics.pop("_component_lambda")

    row = {
        "N": size, "M": len(list(topology.edges())), "method": method,
        "case_package": package,
        "sample_seed": SAMPLE_SEED, "tree_init_seed": tree_seed,
        "max_outer_iterations": cap,
        "faulty_nodes": faulty_nodes(setup),
        "faulty_edges": [list(e) for e in faulty_edges(setup)],
        "n_faulty_nodes": len(faulty_nodes(setup)),
        "n_faulty_edges": len(faulty_edges(setup)),
        "reason": result.reason,
        "converged": result.reason == "converged",
        "total_runtime": elapsed,
        "inner_solve_count": result.inner_solve_count,
        "augmentation_rounds": result.augmentation_rounds,
        "inner_solve_times": solve_times,
        "final_K_phi": result.final_K_phi,
        "final_K_psi": result.final_K_psi,
        "objective": result.objective,
        **metrics,
    }
    row.update(_topology_metadata(package, builder))

    if method == "MILP":
        # PER INNER SOLVE, joined to the K it ran at. The model grows with
        # the sample sets, so one size figure for the whole run would
        # describe only the first pass.
        passes = []
        for index, entry in enumerate(result.pass_history):
            size_entry = model_sizes[index] if index < len(model_sizes) else {}
            passes.append({
                "pass": entry.get("pass", index),
                "K_phi": entry.get("K_phi"), "K_psi": entry.get("K_psi"),
                "status": entry.get("status"),
                "attained_mip_gap": entry.get("attained_mip_gap"),
                "requested_mip_gap": entry.get("requested_mip_gap"),
                "best_bound": entry.get("best_bound"),
                "solve_time": entry.get("solve_time"),
                "time_limit_reached": entry.get("status") == "TIME_LIMIT",
                "n_vars": size_entry.get("n_vars"),
                "n_binary_vars": size_entry.get("n_binary_vars"),
                "n_constrs": size_entry.get("n_constrs"),
            })
        row["milp_passes"] = passes
        row["milp_config"] = {
            "mip_gap": milp_config.mip_gap,
            "time_limit": milp_config.time_limit,
            "verbose": milp_config.verbose,
        }
        row["any_time_limit_reached"] = any(p["time_limit_reached"] for p in passes)
        row["all_solves_optimal"] = all(p["status"] == "OPTIMAL" for p in passes)
        row["milp_statuses"] = [p["status"] for p in passes]

    path = save_result(
        result, problem,
        example_name=f"benchmark_{package}_{'milp' if method == 'MILP' else 'proposed'}",
        method="MILP" if method == "MILP" else "local search",
        sample_seed=SAMPLE_SEED, tree_init_seed=tree_seed,
        max_outer_iterations=cap, runtime_seconds=elapsed,
        setup_name=setup.name,
        faulty_nodes=faulty_nodes(setup), faulty_edges=faulty_edges(setup),
        island_lambda_min=island_lambda, component_lambda_min=component_lambda,
        cost_terms={
            "f1": metrics["code_F1"], "f3": metrics["code_F3"],
            "f4": metrics["code_F4"], "f5": metrics["code_F5"],
            "f6": metrics["code_F6"],
            "economic_cost": metrics["paper_F1"],
            "total_with_penalties": metrics["total_with_penalties"],
            "K_phi": metrics["initial_K_phi"], "K_psi": metrics["initial_K_psi"],
        },
        final_island_witnesses=metrics["final_island_witnesses"],
        final_component_witnesses=metrics["final_component_witnesses"],
    )
    row["result_path"] = str(path)

    print(f"  [{method:>13}] N={size}: {result.reason}  "
          f"{elapsed:.1f} s  {result.inner_solve_count} solves  "
          f"paper F1 = {metrics['paper_F1']:.4f}  "
          f"islands {metrics['n_stable_islands']}S/"
          f"{metrics['n_unstable_islands']}U", flush=True)
    print(f"                  saved {path}", flush=True)
    return row


def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Pre-warm the one process-global cache so the first run does not pay
    # a cost the others avoid. See the module docstring.
    for shape in ((2,), (1,)):
        _shared_zero(np.zeros(shape))

    import gurobipy as gp
    environment = {
        "gurobi_version": ".".join(str(v) for v in gp.gurobi.version()),
        # Threads is left at Gurobi's default of 0, which means the solver
        # CHOOSES a thread count automatically -- not that it necessarily
        # uses every core. baseline/milp.py sets only OutputFlag,
        # TimeLimit and MIPGap, so nothing here pins it either way.
        "gurobi_threads_param": "0 (Gurobi default: automatic selection)",
        "cpu_count_reported_by_os": __import__("os").cpu_count(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "resource_note": (
            "What these timings are: wall-clock time for the complete "
            "end-to-end procedure, for the proposed method AS IMPLEMENTED "
            "and for the MILP with Gurobi at its default automatic thread "
            "setting, run sequentially on one machine. What they are NOT: "
            "a like-for-like comparison of computational resources. "
            "Neither side is thread-pinned -- Gurobi selects its own "
            "parallelism, and the proposed method's NumPy operations may "
            "use whatever threading the platform BLAS (Apple Accelerate "
            "here) applies, so it cannot be called single-threaded "
            "either. Reading the runtime ratio as a per-core or "
            "per-FLOP comparison would not be supported by this setup."
        ),
        "note": "all eight runs executed sequentially on this machine",
    }
    print("=== Environment ===", flush=True)
    for key, value in environment.items():
        print(f"  {key}: {value}", flush=True)
    print(flush=True)

    rows = []
    started = time.perf_counter()
    for size, package, setup_name in CASES:
        print(f"=== N = {size} ({package}) ===", flush=True)
        for method in ("proposed", "MILP"):
            rows.append(run_case(size, package, setup_name, method, timestamp))
        spent = time.perf_counter() - started
        print(f"  elapsed {spent / 60:.1f} min", flush=True)
        print(flush=True)

    summary = {
        "example_name": "milp_benchmark",
        "timestamp": timestamp,
        "purpose": (
            "Experiment 3: proposed inner solver vs MILP inner solver on "
            "identical instances, N = 10/20/30/40, all run locally and "
            "sequentially on one machine."
        ),
        "excluded": (
            "N = 50 is deliberately absent: the only 50-node result was "
            "produced on different hardware, and wall-clock runtimes from "
            "two machines cannot share a table. The N = 10 entry is "
            "linear_10node_rgg, drawn by the same random geometric rule as "
            "the other three sizes -- NOT examples/linear_10node/, whose "
            "hand-specified topology would not belong in a scaling table "
            "beside them."
        ),
        "paper_F1_definition": "economic_cost = code_F1 + code_F3 - code_F4",
        # WHAT THE INNER STOPPING RULE ESTABLISHES. Recorded here because
        # the per-run files report the local search's reason as
        # "local_optimum", which reads stronger than it is.
        "local_search_stopping_rule": (
            "The local search accepts a swap only when "
            "objective > (1 + epsilon) * best_candidate, a RELATIVE test "
            "with epsilon = 0.01. So reason='local_optimum' means no "
            "single swap improved the objective by more than about 1% in "
            "relative terms -- NOT that no improving swap exists. It is "
            "not an exact one-swap local-optimality certificate, and a "
            "swap improving by less than the threshold would have been "
            "declined. Reducing epsilon would tighten the claim at the "
            "cost of more inner iterations."
        ),
        "notation": (
            "code F5/F6 are the paper's F2/F3 and are recorded but kept "
            "out of the paper table: the two methods may end on different "
            "sample sets, so those terms are not comparable within a pair."
        ),
        "environment": environment,
        "sample_seed": SAMPLE_SEED,
        "outer_cap": OUTER_CAP,
        "outer_cap_uniform_across_N": True,
        "outer_cap_note": (
            "20 for every N, which differs from the per-case run "
            "scripts (20 at N=10, 10 at N=20/30/40). Set uniform so "
            "a difference between sizes cannot be a difference of "
            "outer budget."
        ),
        "runs": rows,
        "total_wall_clock_seconds": time.perf_counter() - started,
    }
    path = REPO_ROOT / "results" / f"milp_benchmark_summary_{timestamp}.json"
    path.write_text(json.dumps(summary, indent=2))
    print(f"Saved summary to {path}", flush=True)
    return path


if __name__ == "__main__":
    main()
