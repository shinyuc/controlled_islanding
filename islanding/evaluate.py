"""Bridges search.py's decode functions and cost.py's cost terms.

cost.py stays a leaf module (no dependency on search.py); this module is
the only place that imports both, decoding a spanning tree once and
computing the cost F = F1 + F3 - F4 + F5 + F6. F2 was dropped from the
formulation, so F's f2 argument is passed 0.0 here.
"""

from dataclasses import dataclass

from islanding.search import decode_islanding, decode_reconnection
from islanding.cost import (
    f1_unstable_nodes, f3_operation_edges,
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

    def __post_init__(self):
        assert self.samples_1.shape[1:] == self.samples_2.shape[1:], (
            "samples_1 and samples_2 must agree on (n, d); "
            f"got {self.samples_1.shape[1:]} vs {self.samples_2.shape[1:]}"
        )


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



# TEMPORARY(DISABLE_RECONNECTION) -- test flag, revert before committing.
def effective_reconnection(config, reconnected):
    """The E^rec the objective actually uses: empty when the flag is set.

    THE SINGLE DEFINITION OF THE RULE. It exists because the run scripts
    got this wrong: evaluate() applied the flag while the scripts re-decoded
    the REAL reconnected set for their report, so the search minimised one
    function and the report described another. On the 30-node initial tree
    that was an objective of 129.433 against a reported total of 126.105,
    with F4 = 3.5 and F6 = 0.171 for reconnections the objective had
    ignored. Every consumer -- evaluate(), the run scripts, utils.save --
    now goes through here, so they cannot disagree.

    With an empty E^rec, F4 is 0 by construction and Psi_r loses Parts 1-3
    (N_i^rec is empty for every node), leaving Part 4 = -(k_1 - k_2) W_r
    <= 0, which the per-island hinge clips to zero. F6 is therefore zero as
    its GENUINE value under no reconnection -- not muted, and a sign error
    in Part 4 would still surface.
    """
    return set() if config.disable_reconnection else reconnected


def evaluate(A, problem):
    """Decode spanning tree A once and return the combined cost F(A)."""
    stable, unstable = decode_islanding(A, problem.topology)
    # The components are no longer needed: revised F6 sums over stable
    # islands and E^rec, not over reconnected components.
    _, _, reconnected = decode_reconnection(A, problem.topology)

    f1 = f1_unstable_nodes(unstable, problem.cost_data)
    # No F2 term: the unstable-edge cost was dropped from the formulation,
    # not disabled pending a decision, so there is nothing to switch back
    # on. F still takes an f2 argument for its published signature; 0.0 is
    # passed below. f2_unstable_edges remains in cost.py, unused here.
    f3 = f3_operation_edges(
        stable, unstable, problem.topology, problem.cost_data, problem.config
    )
    # TEMPORARY(DISABLE_RECONNECTION) -- test flag, revert before committing.
    reconnected = effective_reconnection(problem.config, reconnected)

    f4 = f4_reconnection(reconnected, problem.cost_data, problem.config)
    f5 = f5_stability_penalty(
        stable, problem.topology, problem.model, problem.samples_1, problem.config
    )
    f6 = f6_reconnection_stability(
        stable, reconnected, problem.topology, problem.model,
        problem.samples_2, problem.config,
    )

    return F(f1, 0.0, f3, f4, f5, f6)
