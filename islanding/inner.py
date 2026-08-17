"""Solver-agnostic result type for the inner problem.

The outer loop (islanding/outer_loop.py) treats the inner solver as a
parameter, so every inner solver -- local search over spanning trees today,
a MILP baseline later -- reports its answer in this one shape.

Lives in its own module rather than beside SolveResult so that
local_search can import it without importing outer_loop, which imports
local_search back.

The four required fields are the partition itself: everything the outer
loop needs to check stability and grow the sample sets. Notably absent is
the spanning tree -- that is an artifact of the matroid encoding local
search happens to use, not part of the inner problem, and a MILP will not
produce one. Solvers that do have a tree (or any other private
diagnostic) put it in `info`.
"""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class InnerResult:
    stable_islands: list       # list[frozenset] -- the set S
    unstable_islands: list     # list[frozenset] -- the set U
    reconnected_edges: set     # E^rec, canonical (min, max) tuples
    objective: float           # F_hat at the returned partition
    info: dict = field(default_factory=dict)  # solver-specific diagnostics
