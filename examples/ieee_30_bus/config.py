"""AlgorithmConfig for the IEEE 30-bus case.

Every parameter is spelled out here rather than inherited from a shared
baseline: parameter choices differ substantially per case, so each case owns
its full config. examples/common.py holds only generic builder logic.

Importing this module defines the builder only; it runs no computation.

Only the TOPOLOGY of this case comes from the IEEE 30-bus test case. The
dynamics are grounded linear consensus and the weights are drawn from one
seed by the repository's own rules, so none of the parameters below is an
electrical quantity and nothing here is a power-flow study. See
examples/ieee_30_bus/builder.py.

PLACEHOLDER VALUES -- alpha has NOT been tuned for this case. These are
copied from the 10-node case (examples/linear_10node/config.py), where a
sweep put the useful band at roughly 3-5. That band is unlikely to carry
over: alpha weighs the stability penalties against F1 + F3 - F4 (there
is no F2 -- it was dropped from the formulation), and this case's
cost magnitudes are larger -- 30 buses at c_i = 2 and 41 branches at
c_ij = 1, versus 10 nodes and 17 edges. Run a sweep before trusting any
result from this case.

Identical to examples/linear_30node/config.py's build_config(), and
deliberately so: this case exists to be compared against the synthetic
30-node one, which has the same node count but 82 edges to this one's 41.
Holding the AlgorithmConfig fixed removes one source of difference; it
does not reduce the comparison to topology alone. The generator seeds
differ, so every realised weight differs, and the fault locations and
counts differ too -- see generate_ieee_30_bus.py. Only build_milp_config()
differs beyond that, and only in its time limit, which is solver mechanics
rather than problem definition.
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

    MEASURED MODEL SIZE, at K_phi = K_psi = P_1 = P_2 = 20 -- the INITIAL
    model, first pass, before any augmentation:

        case                nodes  edges   vars     binaries  constraints
        linear_10node          10     17    2,491         82        6,284
        linear_20node          20     42   10,871        272       29,583
        ieee_30_bus            30     41   13,042        536       34,111
        linear_30node          30     82   38,728        577      109,849
        linear_40node          40    109   64,814        969      184,690
        linear_50node          50    136  112,430      1,461      321,825

    The model GROWS with the sample sets, one sigma variable and one sigma
    row per slot per witness. For this case, with 30 slots:

        n_vars    = 11,842 + 30 * (K_phi + K_psi)
        n_constrs = 32,911 + 30 * (K_phi + K_psi)

    exact, verified against real builds at (20,20), (25,20) and (23,27).
    The binary count does not depend on K and stays 536 throughout.

    THE COMPARISON THIS CASE IS FOR. Set against linear_30node -- same
    node count, same AlgorithmConfig, half the edges -- the size figures
    separate cleanly by which quantity drives them:

      * binaries, 536 against 577. The gap is exactly 41, the difference
        in edge count, because the reconnection family is one binary per
        edge. The assignment family (465 = 30*31/2) and the slot flags
        (30) are identical, being functions of N alone.
      * variables and rows, roughly a THIRD of the synthetic case's. Those
        are dominated by the per-edge certificate families, which is why
        halving the edge count does so much more to them than to the
        binaries.

    So for MODEL SIZE this pair isolates edge count from node count, which
    no other pair in the repository does -- the families are counted per
    node and per edge, so with N held equal the size gap is the edge gap.
    That does NOT extend to solve behaviour or to solution quality: graph
    shape, the realised weights and the fault pattern differ between the
    two cases as well, so a runtime or objective difference has several
    candidate causes and edge count is only one of them.

    900 s, matching linear_30node. That case's limit was already headroom
    rather than a measured requirement -- its slowest recorded pass was
    2.5 s -- and this model is smaller than it on every axis, so the same
    limit is at least as generous here. Matching rather than reducing it
    keeps one fewer difference between the two cases being compared: if
    this case's MILP behaves differently, the limit is not a candidate
    explanation.

    Note the limit applies PER OUTER PASS, not to the whole run: solve()
    calls the inner MILP once per pass, so time_limit *
    MAX_OUTER_ITERATIONS = 900 * 10 = 9,000 s (2.5 hours) is the maximum
    total time spent INSIDE GUROBI if every pass stalls. It is not a
    wall-clock ceiling -- model construction, certificate generation, the
    counterexample search after each pass and the report all run outside
    Gurobi's clock and add to it.

    A pass that times out is a reportable outcome, not a failure -- the run
    script records the incumbent, the dual bound and the per-pass status,
    so the result stays interpretable. But TIME_LIMIT on most passes means
    the comparison is measuring the limit rather than the method; raise
    this before reading anything into those numbers.

    mip_gap = 1e-6 is tight on purpose, matching every other case. The
    outer loop reads the returned partition as if it were the inner
    optimum, so a loose gap would let solver slack, rather than the sampled
    stability constraints, decide which partition each pass hands back.
    """
    return MILPConfig(time_limit=900.0, mip_gap=1e-6, verbose=False)
