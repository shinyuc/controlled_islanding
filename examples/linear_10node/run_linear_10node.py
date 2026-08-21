"""Diagnostic run script for the 10-node linear consensus example.

Not a test -- makes no assertions, because the answer isn't known in
advance. Compare the printed report against the prediction in
docs/example_linear_10node_spec.md section 8. See
docs/run_linear_10_node_spec.md for the report format this follows.

Writes a JSON summary via utils.save.save_result, matching the 20-, 30-
and 50-node scripts, so every case produces a comparable result file.

How to run
----------
From the repo root, with the venv active:

    python -m examples.linear_10node.run_linear_10node

Use the `-m` form, not
`python examples/linear_10node/run_linear_10node.py` -- that puts the repo
root on sys.path so `examples.linear_10node.linear_10node` resolves.

This script currently runs the EDGE_FAULTS setup (see SETUP below).
examples/linear_10node/linear_10node.py also defines NODE_AND_EDGE_FAULTS;
to run that scenario instead, change the setup passed to build_problem() in
main() below (and update SETUP, which _lambda_min() also relies on).
"""

import time

import numpy as np

from examples.linear_10node.linear_10node import (
    EDGE_FAULTS, NODE_AND_EDGE_FAULTS, build_topology, build_problem, build_M,
    faulty_nodes, faulty_edges,
)
from examples.linear_10node.config import build_config
from utils.save import save_result
from islanding.outer_loop import solve
from islanding.search import build_reconnected_components
from islanding.counterexample import find_counterexamples_for_partition
from islanding.cost import (
    canon, f1_unstable_nodes, f2_unstable_edges, f3_operation_edges,
    f4_reconnection, f5_stability_penalty, f6_reconnection_stability, F,
)

# SETUP = EDGE_FAULTS
SETUP = NODE_AND_EDGE_FAULTS


def _lambda_min(nodes, edges):
    M = build_M(nodes, edges, SETUP.node_weights, SETUP.edge_weights)
    return float(np.linalg.eigvalsh(M).min())


def _induced_edges(topology, nodes):
    return [(u, v) for u, v in topology.edges() if u in nodes and v in nodes]


def main():
    problem = build_problem(
        SETUP, build_topology(), seed=0, config=build_config()
    )
    topology = problem.topology
    config = problem.config

    start = time.perf_counter()
    result = solve(problem, max_outer_iterations=20)
    elapsed = time.perf_counter() - start

    # The partition comes straight off the result -- the outer loop no
    # longer hands back a tree to decode (an inner MILP would have none).
    stable = result.stable_islands
    unstable = result.unstable_islands
    reconnected = result.reconnected_edges
    components = build_reconnected_components(stable, reconnected, topology)

    print("=== Solve ===")
    print(f"reason:                     {result.reason}")
    # Two counts, not one: they differ by one when the loop converges (the
    # final solve found nothing to add) and are equal when it hits the cap.
    print(f"inner solves:               {result.inner_solve_count}")
    print(f"augmentation rounds:        {result.augmentation_rounds}")
    print(f"total witnesses added:      {result.total_witnesses_added}")
    print(f"  island witnesses (F5):    {result.total_island_witnesses_added}")
    print(f"  component witnesses (F6): {result.total_component_witnesses_added}")
    # The sample sets the objective below was actually computed on. On the
    # cap path these are SMALLER than initial P + total witnesses added:
    # the last pass's witnesses were appended after the final solve.
    print(f"final sample counts:        K_phi = {result.final_K_phi}, "
          f"K_psi = {result.final_K_psi}")
    if result.reason == "max_outer_iterations":
        print(
            "  NOTE: hit the outer cap. The reported objective belongs to "
            "the PRE-augmentation sample sets shown above -- the final "
            "pass's witnesses were added but never solved against."
        )
    print(f"runtime (s):                {elapsed:.3f}")
    print()

    print("=== Post-islanding ===")
    print(f"Stable islands ({len(stable)}):")
    island_lambda_min = []
    for island in stable:
        nodes = sorted(island)
        lam = _lambda_min(nodes, _induced_edges(topology, island))
        island_lambda_min.append(lam)
        print(f"  {nodes}  lambda_min = {lam:.4f}")

    print(f"Unstable islands ({len(unstable)}):")
    for island in unstable:
        print(f"  {sorted(island)}")

    within_island_edges = set()
    for island in list(stable) + list(unstable):
        within_island_edges |= {canon(e) for e in _induced_edges(topology, island)}
    all_edges = {canon((u, v)) for u, v in topology.edges()}
    cut_edges = sorted(all_edges - within_island_edges)
    print(f"Cut edges: {cut_edges}")
    print()

    print("=== Reconnection ===")
    print(f"Reconnected edges: {sorted(reconnected)}")
    print(f"Reconnected components ({len(components)}):")
    component_lambda_min = []
    for component in components:
        nodes = sorted(component.nodes())
        lam = _lambda_min(nodes, list(component.edges()))
        component_lambda_min.append(lam)
        print(f"  {nodes}  lambda_min = {lam:.4f}")
    print(f"Unstable islands ({len(unstable)}):")
    for island in unstable:
        print(f"  {sorted(island)}")
    print()

    print("=== Cost ===")
    f1 = f1_unstable_nodes(unstable, problem.cost_data)
    f2 = f2_unstable_edges(unstable, topology, problem.cost_data)
    f3 = f3_operation_edges(stable, unstable, topology, problem.cost_data, config)
    f4 = f4_reconnection(reconnected, problem.cost_data, config)
    f5 = f5_stability_penalty(stable, topology, problem.model, problem.samples_1, config)
    f6 = f6_reconnection_stability(
        stable, reconnected, topology, problem.model, problem.samples_2, config
    )
    print(f"F1 (unstable nodes):      {f1}")
    print(f"F2 (unstable edges):      {f2}  -- MUTED to 0.0 inside evaluate() "
          f"(see islanding/evaluate.py); shown here for reference only")
    print(f"F3 (operation/cut edges): {f3}")
    print(f"F4 (reconnection reward): {f4}")
    print(f"F5 (island stability):    {f5}")
    print(f"F6 (reconnection stab.):  {f6}")
    total = F(f1, 0.0, f3, f4, f5, f6)
    print(f"Total F (F2 muted):       {total}")
    print()

    print("=== Stability check ===")
    island_witnesses, component_witnesses = find_counterexamples_for_partition(
        stable, reconnected, problem
    )
    print(f"Final island witnesses:    {len(island_witnesses)}")
    print(f"Final component witnesses: {len(component_witnesses)}")

    k1_half = config.k_1 / 2
    for island in stable:
        nodes = sorted(island)
        lam = _lambda_min(nodes, _induced_edges(topology, island))
        if lam < k1_half:
            print(
                f"  FLAG: island {nodes} lambda_min={lam:.4f} < k_1/2={k1_half:.4f} "
                "-- passes the counterexample check but still contributes to F5"
            )

    k2_half = config.k_2 / 2
    for component in components:
        nodes = sorted(component.nodes())
        lam = _lambda_min(nodes, list(component.edges()))
        if lam < k2_half:
            print(
                f"  FLAG: component {nodes} lambda_min={lam:.4f} < k_2/2={k2_half:.4f} "
                "-- passes the counterexample check (which is still "
                "component-based; revised F6 is per-island)"
            )
    print()

    path = save_result(
        result, problem, example_name="linear_10node",
        runtime_seconds=elapsed,
        setup_name=SETUP.name,
        faulty_nodes=faulty_nodes(SETUP),
        faulty_edges=faulty_edges(SETUP),
        island_lambda_min=island_lambda_min,
        component_lambda_min=component_lambda_min,
        cost_terms={
            "f1": f1, "f2": f2, "f3": f3, "f4": f4, "f5": f5, "f6": f6,
            "total": total,
            # evaluate() currently zeroes F2, and `total` above was
            # computed as F(f1, 0.0, f3, f4, f5, f6) to match; the f2
            # recorded here is the unmuted value, for reference only.
            "f2_muted": True,
        },
        final_island_witnesses=len(island_witnesses),
        final_component_witnesses=len(component_witnesses),
    )
    print(f"Saved result to {path}")


if __name__ == "__main__":
    main()
