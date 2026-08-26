"""AlgorithmConfig for the 50-node case.

Every parameter is spelled out here rather than inherited from a shared
baseline: parameter choices differ substantially per case, so each case owns
its full config. examples/common.py holds only generic builder logic.

Importing this module defines the builder only; it runs no computation.

PLACEHOLDER VALUES -- alpha has NOT been tuned for this case. These are
copied from the 10-node case (examples/linear_10node/config.py), where a
sweep put the useful band at roughly 3-5. That band is unlikely to carry
over: alpha weighs the stability penalties against F1 + F3 - F4 (there
is no F2 -- it was dropped from the formulation), and this case's
cost magnitudes are far larger -- 50 nodes at c_i = 2 and 136 edges at
c_ij = 1, versus 10 nodes and 17 edges. The same alpha therefore buys
proportionally less influence over the search here, so expect to need a
larger value. Run a sweep before trusting any result from this case.
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

    time_limit is a PER-CASE choice. Measured model sizes, all at
    K_phi = K_psi = P_1 = P_2 = 20 -- the INITIAL model, first pass, before
    any augmentation:

        nodes  edges   vars      binaries  constraints
           10     17     2,491         82        6,284
           20     42    10,871        272       29,583
           30     82    38,728        577      109,849
           40    109    64,814        969      184,690
           50    136   112,430      1,461      321,825

    Those K values are not incidental: the model GROWS with the sample
    sets, one sigma variable and one sigma row per slot per witness. For
    this case, with 50 slots:

        n_vars    = 110,430 + 50 * (K_phi + K_psi)
        n_constrs = 319,825 + 50 * (K_phi + K_psi)

    which is exact -- verified against real builds at (20,20), (25,20),
    (20,23) and (31,29). So any later pass can be reconstructed from the
    K values the run reports, and a figure that does not satisfy these
    rules is not this model at any sample count.

    The 112,430 decomposes as 1,461 binaries + 108,969 continuous
    non-sigma + 1,000 sigma_phi + 1,000 sigma_psi. The binary count does
    NOT depend on K -- augmentation adds only sigma -- so it stays 1,461
    for the whole run.

    So 1.7x the 40-node case in variables and constraints but only 1.5x in
    binaries: 1,275 assignment binaries a_ir (r <= i over 50 nodes), 50
    slot flags and 136 reconnection binaries, 1,461 in all.

    Why the binary count rises more slowly than the row count ACROSS THIS
    BENCHMARK SEQUENCE: the assignment family is N(N+1)/2, quadratic in N,
    while the slot flags are N and the reconnection binaries are |E|. The
    edge counts here (17, 42, 82, 109, 136) happen to have grown roughly
    linearly in N, which is a property of these particular instances --
    geometric graphs with a fixed connection radius and a spread-out
    layout -- and not of the formulation. |E| is quadratic in a dense
    graph, and asymptotically so in a fixed-area random geometric graph,
    where the reconnection family would grow at the same rate as the
    assignment family rather than more slowly. Do not carry this
    observation to an instance drawn differently.

    The binary count is the size figure that tracks branch-and-bound
    difficulty most closely (docs/milp_baseline_spec.md section 8), but it
    does not GOVERN it. Continuous size, row count, LP relaxation
    tightness and the formulation's own structure all matter -- which is
    exactly why the measured scaling below outruns the binary ratio by so
    much. Treat the binaries as the best available proxy, not as the
    explanation.

    3600 s, double the 40-node case's 1800 s. The reasoning is the observed
    SCALING, and the honest answer is that the data admits a band wide
    enough that this limit MAY BIND. Slowest single pass, from the saved
    results of each case's latest run:

        20 nodes, 272 binaries:   0.4 s
        30 nodes, 577 binaries:   2.5 s
        40 nodes, 969 binaries:  98.6 s   (5 passes, 244 s total, all OPTIMAL)

    Read at face value that is a 40x jump in solve time for 1.7x the
    binaries. But the 30-node figure moved when that case's faults were
    relocated to the cluster bottleneck: the runs of the same vintage as
    the 40-node one took 21.9 s on their slowest pass, which makes the same
    step 4.5x instead. Both readings are of real runs; they are simply of
    different 30-node instances.

    Carrying each factor across this case's 1.5x binary jump gives

        98.6 s * 4.5  =    444 s
        98.6 s * 40   =  3,944 s

    so the plausible slowest pass is roughly 400 to 4,000 s. 3600 s sits
    INSIDE that band, near its top -- it is a generous cutoff, NOT a limit
    that can be relied on to leave the solver unconstrained. If the upper
    reading of the scaling is the right one, passes will hit it.

    That is a reportable outcome rather than a failure. A pass that stops
    at the limit returns its best incumbent together with the dual bound,
    so the inner optimum is still bracketed in [best_bound, objective], and
    the run script prints the per-pass status and flags earlier passes that
    stalled. What is lost is the claim that each pass's partition is the
    inner optimum -- read the bound, not the incumbent, when that happens.

    Raising the limit further was rejected: the band's top is an
    extrapolation from a single step (40 nodes was the first case where any
    pass reached three figures at all), so a limit chosen to dominate it
    would be guessing in the same direction, and the worst-case ceiling
    below already runs to a day. If most passes report TIME_LIMIT, the
    answer is to lower MAX_OUTER_ITERATIONS or accept the case as out of
    the MILP's reach.

    One further reason this case needs more room than the size ratio alone
    suggests: MAX_OUTER_ITERATIONS is 20 here, against 10 at 40 nodes, so
    the model passes through twice as many augmentation rounds and the late
    passes are larger than anything the 40-node run ever built.

    THE CEILING, stated precisely. The limit applies PER OUTER PASS, not to
    the whole run: solve() calls the inner MILP once per pass, so
    time_limit * MAX_OUTER_ITERATIONS = 3600 * 20 = 72,000 s (20 hours) is
    the maximum total time spent INSIDE GUROBI if every pass stalls. It is
    not a wall-clock ceiling. Model construction, phi/psi certificate
    generation, the counterexample search after each pass, and the report
    itself all run outside Gurobi's clock and add to it. On the 40-node run
    the five passes spent 233.1 s inside Gurobi out of 244.3 s wall clock,
    so the overhead was about 11 s -- around 5%, small there, but it scales
    with the node count and with the number of passes, and this case has
    twice as many passes and a larger counterexample search.

    OPERATIONAL CAVEAT at that duration: persistence happens only after
    solve() RETURNS. An ordinary Gurobi time-limit exit is handled --
    milp_solver reports it, the outer loop carries on, and the run is saved
    at the end -- and a run with no incumbent at all is saved through
    save_failure_result. But there is no incremental checkpoint: an
    external kill, a machine failure or a lost session during a long later
    pass loses the whole run, including every completed pass. At a 20-hour
    worst case that is a real exposure, and the mitigation available today
    is to run it detached from any interactive session rather than to rely
    on the script.

    mip_gap = 1e-6 is tight on purpose, matching the other four cases. The
    outer loop reads the returned partition as if it were the inner
    optimum, so a loose gap would let solver slack, rather than the sampled
    stability constraints, decide which partition each pass hands back.
    """
    return MILPConfig(time_limit=3600.0, mip_gap=1e-6, verbose=False)
