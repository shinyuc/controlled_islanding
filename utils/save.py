"""Result-saving helper: writes a JSON summary of a solved Problem.

utils depends on islanding (one-way) -- islanding must never import
utils, to keep the core algorithm free of I/O/reporting concerns.
"""

import json
import os
from datetime import datetime

from islanding.cost import canon
from islanding.search import decode_islanding, decode_reconnection


def save_result(result, problem, example_name, results_dir="results"):
    """Decode the solved tree and write a JSON summary to
    results/<example_name>_<timestamp>.json. Creates results_dir if missing.

    Returns the path written.
    """
    os.makedirs(results_dir, exist_ok=True)

    topology = problem.topology
    tree = result.tree

    stable, unstable = decode_islanding(tree, topology)
    components, _, reconnected = decode_reconnection(tree, topology)

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
        "reason": result.reason,
        "objective": float(result.objective),
        "outer_iterations": result.outer_iterations,
        "witnesses_added": result.witnesses_added,
        "stable_islands": [sorted(island) for island in stable],
        "unstable_islands": [sorted(island) for island in unstable],
        "cut_edges": [list(edge) for edge in cut_edges],
        "reconnected_edges": [list(edge) for edge in sorted(reconnected)],
        "reconnected_components": [
            sorted(component.nodes()) for component in components
        ],
    }

    with open(path, "w") as f:
        json.dump(payload, f, indent=2)

    return path
