"""AlgorithmConfig for the 10-node case.

Every parameter is spelled out here rather than inherited from a shared
baseline: parameter choices differ substantially per case, so each case owns
its full config and the values it actually ran with stay auditable in one
place. examples/common.py holds only generic builder logic.

Importing this module defines the builder only; it runs no computation.

Alpha sweep on this case (NODE_AND_EDGE_FAULTS, seed=0), total F at
convergence:

    alpha = 1     never converges -- with the 1/P_1 averaging the sampled
                  violations are diluted below the combinatorial savings,
                  so the search prefers cutting to satisfying stability
    alpha = 3     F = 6.9   sheds only node 10 (the one genuinely
                            unshedable node), three reconnected edges;
                            6 outer passes
    alpha = 5     F = 6.9   same partition and cost; 3 outer passes
    alpha = 8     F = 14.0  sheds four nodes, no reconnection at all
    alpha = 100   F = 14.0  same over-shedding as alpha = 8

The useful band is therefore roughly 3-5: below it the penalty cannot steer
the search, above it the search over-sheds. alpha = 5 is picked over
alpha = 3 because it reaches the same optimum in half the outer passes.
"""

from islanding.cost import AlgorithmConfig
from baseline.config import MILPConfig

ALPHA = 5.0  # see the sweep in this module's docstring


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
    It is kept as a separate builder because it carries nothing about what
    is being minimised -- only how the solver goes about it -- so the two
    methods still share build_config() and cannot drift apart.

    time_limit is a PER-CASE choice: how long the MILP needs scales with
    instance size (node and edge count, and through the certificate
    families with P_1/P_2 as well), so it cannot be set once globally.
    300 s is generous for 10 nodes -- the solve closes the gap well inside
    it -- and larger cases should raise it rather than inherit this number.
    Note the limit applies PER OUTER PASS, not to the whole run: solve()
    calls the inner MILP once per pass.

    mip_gap = 1e-6 is tight on purpose. The outer loop reads the returned
    partition as if it were the inner optimum, so a loose gap would let
    solver slack, rather than the sampled stability constraints, decide
    which partition each pass hands back.
    """
    return MILPConfig(time_limit=300.0, mip_gap=1e-6, verbose=False)
