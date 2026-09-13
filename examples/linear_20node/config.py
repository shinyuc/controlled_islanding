"""AlgorithmConfig for the 20-node case.

Every parameter is spelled out here rather than inherited from a shared
baseline: parameter choices differ substantially per case, so each case owns
its full config. examples/common.py holds only generic builder logic.

Importing this module defines the builder only; it runs no computation.

PLACEHOLDER VALUES -- alpha has NOT been tuned for this case. These are
copied from the 10-node case (examples/linear_10node/config.py), where a
sweep put the useful band at roughly 3-5. That band is unlikely to carry
over: alpha weighs the stability penalties against F1 + F3 - F4 (there
is no F2 -- it was dropped from the formulation), and this case's
cost magnitudes are larger -- 20 nodes at c_i = 2 and 42 edges at
c_ij = 1, versus 10 nodes and 17 edges. The same alpha therefore buys
proportionally less influence over the search here, so expect to need a
larger value. Run a sweep before trusting any result from this case.

Starts from the same values as every other case, so they stay
comparable until
either is tuned.
"""

from islanding.cost import AlgorithmConfig
from baseline.config import MILPConfig

ALPHA = 5.0  # UNTUNED -- inherited from the 10-node sweep, see docstring


def build_config():
    return AlgorithmConfig(
        kappa_op=1.0, kappa_rec=0.7,
        alpha_1=ALPHA, beta=1.0, k_1=0.3, P_1=20,
        alpha_2=ALPHA, gamma=1.0, k_2=0.1, P_2=20,
        epsilon=0.01, max_iterations=100, swap_supernode_edges=True,
    )


def build_milp_config():
    """Solver mechanics for the MILP baseline on THIS case.

    Lives beside build_config() for the same reason the AlgorithmConfig
    does: the values a case actually ran with stay auditable in one place.
    It is a separate builder because it carries nothing about what is being
    minimised -- only how the solver goes about it -- so both methods still
    share build_config() and cannot drift apart.

    time_limit is a PER-CASE choice, and 600 s rather than the 10-node
    case's 300 s. Measured on this case's model at P_1 = P_2 = 20:

        10 nodes, 17 edges:   2,491 vars,  82 binaries,  6,284 constraints
        20 nodes, 42 edges:  10,871 vars, 272 binaries, 29,583 constraints

    so 4.4x the variables (the tau and q families scale as N * sum_i d_i^2)
    and 4.7x the rows. But raw size is not what makes it hard -- the
    BINARIES are (docs/milp_baseline_spec.md section 8). Here that is 210
    assignment binaries a_ir, plus 20 slot flags and 42 reconnection
    binaries: 272 in all. (The often-quoted O(N^2) = 400 for 20 nodes is
    the count BEFORE the representative convention restricts a_ir to
    r <= i; branch-and-bound difficulty still grows faster than the
    variable count suggests.)

    600 s is a starting point, not a measured requirement. A pass that
    times out is a reportable outcome, not a failure -- the run script
    records the incumbent, the dual bound and the status, so the result
    stays interpretable. But a limit hit on EVERY pass wastes the run: the
    outer loop would then be growing witnesses against a sequence of
    unproven incumbents, and the comparison against the local search would
    measure the time limit rather than the method. If the report shows
    TIME_LIMIT on most passes, raise this before reading anything into the
    numbers.

    Note the limit applies PER OUTER PASS, not to the whole run: solve()
    calls the inner MILP once per pass, so the worst-case wall clock is
    roughly time_limit * MAX_OUTER_ITERATIONS.

    mip_gap = 1e-6 is tight on purpose, and matches the 10-node case. The
    outer loop reads the returned partition as if it were the inner
    optimum, so a loose gap would let solver slack, rather than the sampled
    stability constraints, decide which partition each pass hands back.
    """
    return MILPConfig(time_limit=600.0, mip_gap=1e-6, verbose=False)
