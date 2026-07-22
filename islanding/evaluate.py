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
    samples_1: object  # for F5, drawn from {W <= (1+inflation)*beta} with P_1 points
    samples_2: object  # for F6, drawn from {W <= (1+inflation)*gamma} with P_2 points


def draw_samples(topology, model, config, n, d):
    """Draw both fixed sample sets, applying the model's inflation factor.

    Omega     = {x : W(x) <= (1 + inflation) * beta}    -- for F5
    Omega_hat = {x : W(x) <= (1 + inflation) * gamma}   -- for F6

    Same factor for both; only the base radius (beta vs gamma) differs.
    model.sampler itself stays inflation-agnostic -- it only ever receives
    a plain, already-inflated threshold.
    """
    factor = 1.0 + model.inflation(topology)
    samples_1 = model.sampler(config.beta * factor, config.P_1, n, d)
    samples_2 = model.sampler(config.gamma * factor, config.P_2, n, d)
    return samples_1, samples_2


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
