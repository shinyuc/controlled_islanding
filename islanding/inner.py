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


class InnerSolveFailure(RuntimeError):
    """An inner solver terminated without a usable partition.

    Solver-agnostic, and lives here beside InnerResult for the same
    reason: the outer loop must be able to catch and enrich a failure
    without importing any particular solver. baseline.milp.NoIncumbentError
    subclasses it, so the dependency still runs one way (baseline imports
    islanding, never the reverse).

    `info` is the solver's own diagnostics -- status, model size, limits,
    whatever it can report without a solution. `outer_state` is filled in
    by islanding.outer_loop.solve, which is the only layer that knows
    which pass failed and what sample sets that pass was solving against.
    It stays None if the solver is called outside the outer loop.
    """

    def __init__(self, message, info=None):
        super().__init__(message)
        self.info = dict(info or {})
        self.outer_state = None

    def attach_outer_state(self, **state):
        """Record the loop state around the failed pass, and return self.

        Returns self so a caller can `raise exc.attach_outer_state(...)`
        and keep the original traceback.
        """
        self.outer_state = dict(state)
        return self
