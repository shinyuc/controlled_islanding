"""Bridges search.py's decode functions and cost.py's cost terms.

cost.py stays a leaf module (no dependency on search.py); this module is
the only place that imports both, decoding a spanning tree once and
computing the full six-term cost F = F1 + F2 + F3 - F4 + F5 + F6.
"""

from dataclasses import dataclass

from islanding.search import decode_islanding, decode_reconnection
from islanding.cost import (
    f1_unstable_nodes, f2_unstable_edges, f3_operation_edges,
    f4_reconnection, f5_stability_penalty, f6_reconnection_stability, F,
)


@dataclass(frozen=True)
class Problem:
    topology: object
    model: object
    cost_data: object
    config: object
    samples_1: object  # for F5, drawn from {W <= beta} with P_1 points
    samples_2: object  # for F6, drawn from {W <= gamma} with P_2 points


def evaluate(A, problem):
    """Decode spanning tree A once and return the combined cost F(A)."""
    stable, unstable = decode_islanding(A, problem.topology)
    components, _, reconnected = decode_reconnection(A, problem.topology)

    f1 = f1_unstable_nodes(unstable, problem.cost_data)
    f2 = f2_unstable_edges(unstable, problem.topology, problem.cost_data)
    f3 = f3_operation_edges(
        stable, unstable, problem.topology, problem.cost_data, problem.config
    )
    f4 = f4_reconnection(reconnected, problem.cost_data, problem.config)
    f5 = f5_stability_penalty(
        stable, problem.topology, problem.model, problem.samples_1, problem.config
    )
    f6 = f6_reconnection_stability(
        components, problem.topology, problem.model, problem.samples_2, problem.config
    )

    return F(f1, f2, f3, f4, f5, f6)
