"""AlgorithmConfig for the 40-node case.

Every parameter is spelled out here rather than inherited from a shared
baseline: parameter choices differ substantially per case, so each case owns
its full config. examples/common.py holds only generic builder logic.

Importing this module defines the builder only; it runs no computation.

PLACEHOLDER VALUES -- alpha has NOT been tuned for this case. These are
copied from the 10-node case (examples/linear_10node/config.py), where a
sweep put the useful band at roughly 3-5. That band is unlikely to carry
over: alpha weighs the stability penalties against F1 + F3 - F4 (there
is no F2 -- it was dropped from the formulation), and this case's
cost magnitudes are larger -- 40 nodes at c_i = 2 and 109 edges at
c_ij = 1, versus 10 nodes and 17 edges. The same alpha therefore buys
proportionally less influence over the search here, so expect to need a
larger value. Run a sweep before trusting any result from this case.

Starts from the same values as the 20-, 30- and 50-node cases, so all
four stay comparable until any of them is tuned.
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

        nodes  edges   vars     binaries  constraints
           10     17    2,491         82        6,284
           20     42   10,871        272       29,583
           30     82   38,728        577      109,849
           40    109   64,814        969      184,690

    So 1.7x the 30-node case on every axis: 820 assignment binaries a_ir
    (r <= i over 40 nodes), 40 slot flags and 109 reconnection binaries,
    969 in all.

    The binary count is the size figure that tracks branch-and-bound
    difficulty most closely (docs/milp_baseline_spec.md section 8), but it
    does not GOVERN it. Continuous size, row count, LP relaxation
    tightness and the formulation's own structure all matter -- which is
    exactly why the measured scaling below outruns the binary ratio by so
    much. Treat the binaries as the best available proxy, not as the
    explanation.

    1800 s, double the 30-node case's 900 s. The reasoning is the observed
    SCALING, not the size ratio -- solve time has grown far faster than the
    binary count:

        20 nodes, 272 binaries: slowest pass 0.5 s
        30 nodes, 577 binaries: slowest pass 21.9 s

    That is roughly 40x the solve time for 2.1x the binaries. Applying
    anything like that factor to this case's 1.7x jump puts a hard pass in
    the hundreds of seconds, which a 900 s limit could plausibly bind on --
    and a limit that binds turns the comparison into a measurement of the
    limit rather than of the method. 1800 s leaves room for a pass that is
    merely slow without abandoning it.

    Note the limit applies PER OUTER PASS, not to the whole run: solve()
    calls the inner MILP once per pass, so the worst case is
    time_limit * MAX_OUTER_ITERATIONS = 1800 * 10 = 18,000 s (5 hours) if
    every pass stalls. That ceiling is deliberately generous; if the report
    shows TIME_LIMIT on most passes, the answer is to lower the cap or
    accept the case as out of the MILP's reach, not to raise this further.

    A pass that times out is a reportable outcome, not a failure -- the run
    script records the incumbent, the dual bound and the per-pass status,
    so the result stays interpretable.

    mip_gap = 1e-6 is tight on purpose, matching the other three cases. The
    outer loop reads the returned partition as if it were the inner
    optimum, so a loose gap would let solver slack, rather than the sampled
    stability constraints, decide which partition each pass hands back.
    """
    return MILPConfig(time_limit=1800.0, mip_gap=1e-6, verbose=False)
