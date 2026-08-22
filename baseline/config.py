"""Solver mechanics for the MILP baseline (spec section 2).

Deliberately separate from islanding.cost.AlgorithmConfig, which carries
the PROBLEM parameters (kappa_op, kappa_rec, k_1, k_2, alpha_1, alpha_2,
P_1, P_2) and is shared with the local search so the two methods cannot
drift apart. Nothing here affects what is being minimised -- only how the
solver goes about it.
"""

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class MILPConfig:
    time_limit: float   # seconds; mandatory, see spec section 8
    mip_gap: float      # relative gap to accept
    verbose: bool = False

    def __post_init__(self):
        # ValueError, not assert: `mip_gap` is what the persisted
        # "optimal within the configured Gurobi tolerances" claim refers
        # to, so the check behind it must not vanish under `python -O`.
        # A NaN gap in particular would pass any comparison-based test and
        # then be recorded as the tolerance a proof was established at.
        for name in ("time_limit", "mip_gap"):
            value = getattr(self, name)
            try:
                number = float(value)
            except (TypeError, ValueError):
                raise ValueError(
                    f"{name} must be a real number; got {value!r}"
                ) from None
            if not math.isfinite(number):
                raise ValueError(f"{name} must be finite; got {number!r}")
        if not self.time_limit > 0:
            raise ValueError(
                f"time_limit must be positive; got {self.time_limit!r}"
            )
        if not self.mip_gap >= 0:
            raise ValueError(
                f"mip_gap must be nonnegative; got {self.mip_gap!r}"
            )
