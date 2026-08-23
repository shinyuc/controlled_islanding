"""AlgorithmConfig for the 30-node case.

Every parameter is spelled out here rather than inherited from a shared
baseline: parameter choices differ substantially per case, so each case owns
its full config. examples/common.py holds only generic builder logic.

Importing this module defines the builder only; it runs no computation.

PLACEHOLDER VALUES -- alpha has NOT been tuned for this case. These are
copied from the 10-node case (examples/linear_10node/config.py), where a
sweep put the useful band at roughly 3-5. That band is unlikely to carry
over: alpha weighs the stability penalties against F1 + F3 - F4 (there
is no F2 -- it was dropped from the formulation), and this case's
cost magnitudes are larger -- 30 nodes at c_i = 2 and 82 edges at
c_ij = 1, versus 10 nodes and 17 edges. The same alpha therefore buys
proportionally less influence over the search here, so expect to need a
larger value. Run a sweep before trusting any result from this case.

Starts from the same values as the 50-node case
(examples/linear_50node/config.py), so the two stay comparable until
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

    time_limit is a PER-CASE choice. Measured model sizes, at
    P_1 = P_2 = 20 (the INITIAL model; it grows with the sample sets, one
    sigma row per slot per witness):

        nodes  edges   vars    binaries  constraints
           10     17    2,491        82        6,284
           20     42   10,871       272       29,583
           30     82   38,728       577      109,849

    So 3.6x the 20-node variables and 3.7x its rows, but only 2.1x its
    binaries -- and the binaries are what governs branch-and-bound
    difficulty (docs/milp_baseline_spec.md section 8). Here that is 465
    assignment binaries a_ir (r <= i over 30 nodes), 30 slot flags and 82
    reconnection binaries: 577.

    900 s, up from the 20-node case's 600 s. That is HEADROOM, not a
    measured requirement, and the measurement says so: the first pass of
    this case solves to OPTIMAL in 0.7 s, and no 20-node pass ever came
    within three orders of magnitude of its limit (0.3-0.5 s against
    600 s). Nothing observed here needs 900 s.

    The limit exists for the pass that does not behave -- branch-and-bound
    time can grow far faster than the binary count, and the passes that
    bite are the later ones, where accumulated witnesses make the penalty
    terms active and the LP relaxation weaker. 900 s bounds one such pass
    without abandoning it early.

    Note the limit applies PER OUTER PASS, not to the whole run: solve()
    calls the inner MILP once per pass, so the worst case is
    time_limit * MAX_OUTER_ITERATIONS = 900 * 10 = 9,000 s (2.5 hours) if
    every pass stalls. Raise MAX_OUTER_ITERATIONS and that ceiling moves
    with it.

    A pass that times out is a reportable outcome, not a failure -- the run
    script records the incumbent, the dual bound and the per-pass status,
    so the result stays interpretable. But TIME_LIMIT on most passes means
    the comparison is measuring the limit rather than the method; raise
    this before reading anything into those numbers.

    mip_gap = 1e-6 is tight on purpose, matching the 10- and 20-node cases.
    The outer loop reads the returned partition as if it were the inner
    optimum, so a loose gap would let solver slack, rather than the sampled
    stability constraints, decide which partition each pass hands back.
    """
    return MILPConfig(time_limit=900.0, mip_gap=1e-6, verbose=False)
