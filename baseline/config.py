"""Solver mechanics for the MILP baseline (spec section 2).

Deliberately separate from islanding.cost.AlgorithmConfig, which carries
the PROBLEM parameters (kappa_op, kappa_rec, k_1, k_2, alpha_1, alpha_2,
P_1, P_2) and is shared with the local search so the two methods cannot
drift apart. Nothing here affects what is being minimised -- only how the
solver goes about it.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class MILPConfig:
    time_limit: float   # seconds; mandatory, see spec section 8
    mip_gap: float      # relative gap to accept
    verbose: bool = False

    def __post_init__(self):
        assert self.time_limit > 0
        assert self.mip_gap >= 0
