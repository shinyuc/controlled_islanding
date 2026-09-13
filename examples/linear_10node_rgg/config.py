"""AlgorithmConfig for the 10-node random-geometric-graph case.

Every parameter is spelled out here rather than inherited from a shared
baseline: parameter choices differ substantially per case, so each case owns
its full config. examples/common.py holds only generic builder logic.

Importing this module defines the builder only; it runs no computation.

PLACEHOLDER VALUES -- alpha has NOT been tuned for this case. These are
copied from the 10-node case (examples/linear_10node/config.py), where a
sweep put the useful band at roughly 3-5.

IDENTICAL to every other case's build_config(), and deliberately so: this
case exists to sit in a scaling table beside the 20-, 30- and 40-node
ones, and a scaling curve is only readable if the algorithm parameters
are held fixed along it. What differs between the cases is the instance,
not the algorithm.
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

        case                nodes  edges    vars   binaries  constraints
        linear_10node_rgg      10     19    2,445         84        6,206
        linear_10node          10     17    2,491         82        6,284
        linear_20node          20     42   10,871        272       29,583
        linear_30node          30     82   38,728        577      109,849
        linear_40node          40    109   64,814        969      184,690

    The model GROWS with the sample sets, one sigma variable and one sigma
    row per slot per witness. For this case, with 10 slots:

        n_vars    = 2,045 + 10 * (K_phi + K_psi)
        n_constrs = 5,806 + 10 * (K_phi + K_psi)

    exact, verified against real builds at (20,20), (25,20) and (23,27).
    The binary count does not depend on K and stays 84 throughout: 55
    assignment binaries a_ir (r <= i over 10 nodes), 10 slot flags and 19
    reconnection binaries.

    NOTE THE NON-MONOTONICITY, which is easy to get wrong: this case has
    MORE edges than the hand-specified 10-node one (19 against 17) and yet
    FEWER variables and rows (2,445 against 2,491; 6,206 against 6,284).
    Only the binary count moves the way edge count alone would predict,
    84 against 82. The continuous families are generated per node over
    admissible slot ranges (baseline/milp.py's _slots_for), and those
    ranges depend on WHICH nodes are adjacent, not merely on how many
    edges there are -- so model size is not a function of (N, M). An
    earlier draft of this docstring asserted the difference was "entirely
    the two extra edges"; the measurement says otherwise.

    300 s, matching examples/linear_10node/config.py. That limit was
    already far more than the case needs -- the hand-specified 10-node
    model solves in well under a second per pass -- and this model is the
    same size to within 10%, so the same limit is at least as generous.
    Matching rather than re-deriving keeps one fewer difference between
    two cases that exist to be compared.

    Note the limit applies PER OUTER PASS, not to the whole run.

    mip_gap = 1e-6 is tight on purpose, matching every other case. The
    outer loop reads the returned partition as if it were the inner
    optimum, so a loose gap would let solver slack, rather than the sampled
    stability constraints, decide which partition each pass hands back.
    """
    return MILPConfig(time_limit=300.0, mip_gap=1e-6, verbose=False)
