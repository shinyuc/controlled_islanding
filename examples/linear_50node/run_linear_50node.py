"""Diagnostic run script for the 50-node linear consensus example.

Not a test -- makes no assertions, because the answer isn't known in
advance. Compare the printed report against the prediction in
docs/example_linear_50node_spec.md section 8. Mirrors
examples/linear_10node/run_linear_10node.py's report format; see
docs/run_linear_10_node_spec.md for that format's rationale.

How to run
----------
From the repo root, with the venv active:

    python -m examples.linear_50node.run_linear_50node

Use the `-m` form, not `python examples/linear_50node/run_linear_50node.py`
-- that puts the repo root on sys.path so `examples.linear_50node.builder`
resolves.
"""

import time

import numpy as np

from examples.linear_50node.builder import (
    LINEAR_50, build_topology, build_problem, build_M, faulty_nodes, faulty_edges,
)
from utils.save import save_result
from islanding.outer_loop import solve
from islanding.search import decode_islanding, decode_reconnection
from islanding.counterexample import find_counterexamples
from islanding.cost import (
    canon, f1_unstable_nodes, f2_unstable_edges, f3_operation_edges,
    f4_reconnection, f5_stability_penalty, f6_reconnection_stability, F,
)

SETUP = LINEAR_50


def _lambda_min(nodes, edges):
    M = build_M(nodes, edges, SETUP.node_weights, SETUP.edge_weights)
    return float(np.linalg.eigvalsh(M).min())


def _induced_edges(topology, nodes):
    return [(u, v) for u, v in topology.edges() if u in nodes and v in nodes]


def main():
    topology = build_topology()
    problem = build_problem(SETUP, topology, seed=0)

    print(f"Setup: {SETUP.name}")
    print(f"Faulty nodes: {faulty_nodes(SETUP)}")
    print(f"Faulty edges: {faulty_edges(SETUP)}")
    print()

    config = problem.config

    start = time.perf_counter()
    result = solve(problem, max_outer_iterations=20)
    elapsed = time.perf_counter() - start

    tree = result.tree
    stable, unstable = decode_islanding(tree, topology)
    components, _, reconnected = decode_reconnection(tree, topology)

    print("=== Solve ===")
    print(f"reason:           {result.reason}")
    print(f"outer_iterations: {result.outer_iterations}")
    print(f"witnesses_added:  {result.witnesses_added}")
    print(f"runtime (s):      {elapsed:.3f}")
    print()

    print("=== Post-islanding ===")
    print(f"Stable islands ({len(stable)}):")
    for island in stable:
        nodes = sorted(island)
        lam = _lambda_min(nodes, _induced_edges(topology, island))
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
    for component in components:
        nodes = sorted(component.nodes())
        lam = _lambda_min(nodes, list(component.edges()))
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
    island_witnesses, component_witnesses = find_counterexamples(tree, problem)
    print(f"Island witnesses:    {len(island_witnesses)}")
    print(f"Component witnesses: {len(component_witnesses)}")

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

    path = save_result(result, problem, example_name="linear_50node")
    print(f"Saved result to {path}")


if __name__ == "__main__":
    main()
