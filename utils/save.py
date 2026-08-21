"""Result-saving helper: writes a JSON summary of a solved Problem.

utils depends on islanding (one-way) -- islanding must never import
utils, to keep the core algorithm free of I/O/reporting concerns.

This module stays free of example-specific knowledge: anything that needs
a case's weights (lambda_min per island, the faulty node/edge lists, the
six cost terms) is computed by the run script and passed in as plain data.
save_result only serializes.
"""

import json
import os
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from islanding.cost import canon
from islanding.search import build_reconnected_components

# utils/save.py -> utils/ -> repo root.
REPO_ROOT = Path(__file__).resolve().parent.parent

COST_KEYS = ("f1", "f2", "f3", "f4", "f5", "f6", "total", "f2_muted")


def save_result(
    result,
    problem,
    example_name,
    *,
    runtime_seconds,
    setup_name,
    faulty_nodes,
    faulty_edges,
    island_lambda_min,
    component_lambda_min,
    cost_terms,
    final_island_witnesses,
    final_component_witnesses,
    results_dir=None,
):
    """Write a JSON summary to <results_dir>/<example_name>_<timestamp>.json.
    Creates results_dir if missing. Returns the path written.

    Everything the terminal report prints is recorded, so a saved result
    can be interpreted months later without re-running anything:

    - `runtime_seconds`: wall clock for the solve, measured by the caller
      with time.perf_counter(). It cannot be measured here -- save_result
      runs after the solve -- so it must be passed in.
    - `setup_name`, `faulty_nodes`, `faulty_edges`: which instance this is.
    - `island_lambda_min` / `component_lambda_min`: sequences of floats
      aligned by INDEX with result.stable_islands and with the components
      rebuilt below, in that order. Computed by the caller via build_M,
      since only the caller knows the case's weights.
    - `cost_terms`: dict with keys f1..f6, total, f2_muted.
    - the loop's two work counts (`inner_solve_count`,
      `augmentation_rounds`) and the sample-set sizes the objective was
      computed on (`final_K_phi`, `final_K_psi`), all read off the
      SolveResult -- see islanding/outer_loop.py for why one iteration
      count could not stand for both.
    - the full AlgorithmConfig, without which the numbers above cannot be
      interpreted (alpha and P in particular change what F5/F6 mean).

    results_dir defaults to <repo_root>/results, anchored to this file's
    location rather than the current working directory, so results always
    land inside the project regardless of where the script is launched
    from. Pass an explicit results_dir to override.
    """
    missing = [key for key in COST_KEYS if key not in cost_terms]
    if missing:
        raise ValueError(f"cost_terms is missing {missing}; expected {list(COST_KEYS)}")

    if results_dir is None:
        results_dir = REPO_ROOT / "results"
    os.makedirs(results_dir, exist_ok=True)

    topology = problem.topology

    # Read the partition off the result rather than decoding a tree: the
    # inner solver may not have produced one (a MILP does not).
    stable = result.stable_islands
    unstable = result.unstable_islands
    reconnected = result.reconnected_edges
    components = build_reconnected_components(stable, reconnected, topology)

    # Guard the index alignment: a mis-wired call site would otherwise
    # attach the wrong lambda_min to an island, silently.
    if len(island_lambda_min) != len(stable):
        raise ValueError(
            f"island_lambda_min has {len(island_lambda_min)} entries but there "
            f"are {len(stable)} stable islands"
        )
    if len(component_lambda_min) != len(components):
        raise ValueError(
            f"component_lambda_min has {len(component_lambda_min)} entries but "
            f"there are {len(components)} reconnected components"
        )

    # Original edges not inside any (stable or unstable) island -- the cut
    # set. Do NOT serialize the raw tree: it's over augmented nodes
    # (supernodes/relays), not the original graph.
    within_island_edges = set()
    for island in list(stable) + list(unstable):
        within_island_edges |= {
            canon((u, v)) for u, v in topology.edges()
            if u in island and v in island
        }
    all_edges = {canon((u, v)) for u, v in topology.edges()}
    cut_edges = sorted(all_edges - within_island_edges)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{example_name}_{timestamp}.json"
    path = os.path.join(results_dir, filename)

    payload = {
        "example_name": example_name,
        "timestamp": timestamp,
        "runtime_seconds": float(runtime_seconds),
        "setup": {
            "name": setup_name,
            "faulty_nodes": [int(node) for node in faulty_nodes],
            "faulty_edges": [list(edge) for edge in faulty_edges],
        },
        "problem_size": {
            "n_nodes": len(list(topology.nodes())),
            "n_edges": len(list(topology.edges())),
        },
        "config": {key: _plain(value) for key, value in asdict(problem.config).items()},
        "reason": result.reason,
        "objective": float(result.objective),
        "inner_solve_count": result.inner_solve_count,
        "augmentation_rounds": result.augmentation_rounds,
        "total_witnesses_added": result.total_witnesses_added,
        "total_island_witnesses_added": result.total_island_witnesses_added,
        "total_component_witnesses_added": result.total_component_witnesses_added,
        # Sample-set sizes the recorded `objective` was computed on. Do not
        # reconstruct these as initial P + total_witnesses_added: on the cap
        # path the last pass's witnesses were added after the final solve.
        "final_K_phi": result.final_K_phi,
        "final_K_psi": result.final_K_psi,
        "cost": {key: _plain(cost_terms[key]) for key in COST_KEYS},
        "stable_islands": [
            {"nodes": sorted(island), "lambda_min": float(lam)}
            for island, lam in zip(stable, island_lambda_min)
        ],
        "unstable_islands": [sorted(island) for island in unstable],
        "cut_edges": [list(edge) for edge in cut_edges],
        "reconnected_edges": [list(edge) for edge in sorted(reconnected)],
        "reconnected_components": [
            {"nodes": sorted(component.nodes()), "lambda_min": float(lam)}
            for component, lam in zip(components, component_lambda_min)
        ],
        "final_stability_check": {
            "island_witnesses": int(final_island_witnesses),
            "component_witnesses": int(final_component_witnesses),
        },
    }

    with open(path, "w") as f:
        json.dump(payload, f, indent=2)

    return path


def _plain(value):
    """Coerce numpy scalars to built-ins so json.dump accepts them.

    np.float64 subclasses float and would serialize anyway, but np.int64
    and np.bool_ do not subclass their Python counterparts and raise. Every
    numpy scalar exposes .item(); no built-in scalar does.
    """
    return value.item() if hasattr(value, "item") else value
