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

ALPHA = 5.0  # see the sweep in this module's docstring


def build_config():
    return AlgorithmConfig(
        kappa_op=1.0, kappa_rec=0.7,
        alpha_1=ALPHA, beta=1.0, k_1=0.3, P_1=20,
        alpha_2=ALPHA, gamma=1.0, k_2=0.1, P_2=20,
        epsilon=0.01, max_iterations=100, swap_supernode_edges=True,
    )
