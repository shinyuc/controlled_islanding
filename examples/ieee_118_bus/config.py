"""AlgorithmConfig for the IEEE 118-bus case.

Every parameter is spelled out here rather than inherited from a shared
baseline: parameter choices differ substantially per case, so each case owns
its full config. examples/common.py holds only generic builder logic.

Importing this module defines the builder only; it runs no computation.

Only the TOPOLOGY of this case comes from the IEEE 118-bus test case. The
dynamics are grounded linear consensus and the weights are drawn from one
seed by the repository's own rules, so none of the parameters below is an
electrical quantity and nothing here is a power-flow study. See
examples/ieee_118_bus/builder.py.

PLACEHOLDER VALUES -- alpha has NOT been tuned for this case. These are
copied from the 10-node case (examples/linear_10node/config.py), where a
sweep put the useful band at roughly 3-5. That band is unlikely to carry
over: alpha weighs the stability penalties against F1 + F3 - F4 (there
is no F2 -- it was dropped from the formulation), and this case's
cost magnitudes are far larger -- 118 buses at c_i = 2 and 179 branches
at c_ij = 1, versus 10 nodes and 17 edges. Run a sweep before trusting any
result from this case.

Identical to every other case's build_config(), and deliberately so: this
case exists to extend the SCALING comparison to a size no synthetic case
reaches, and a scaling curve is only readable if the algorithm parameters
are held fixed along it. What is NOT held fixed is everything about the
instance -- topology, weights, fault locations and counts all differ from
every other case, so this is a further point on a curve, not a controlled
variant of anything.

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

        case                nodes  edges    vars     binaries  constraints
        linear_10node          10     17     2,491         82        6,284
        linear_20node          20     42    10,871        272       29,583
        ieee_30_bus            30     41    13,042        536       34,111
        linear_30node          30     82    38,728        577      109,849
        linear_40node          40    109    64,814        969      184,690
        linear_50node          50    136   112,430      1,461      321,825
        ieee_118_bus          118    179   269,995      7,318      754,046

    The model GROWS with the sample sets, one sigma variable and one sigma
    row per slot per witness. For this case, with 118 slots:

        n_vars    = 265,275 + 118 * (K_phi + K_psi)
        n_constrs = 749,326 + 118 * (K_phi + K_psi)

    exact, verified against real builds at (20,20), (25,20) and (23,27).
    The binary count does not depend on K and stays 7,318 throughout:
    7,021 assignment binaries a_ir (r <= i over 118 buses), 118 slot flags
    and 179 reconnection binaries.

    THE STEP UP IS IN THE BINARIES. Against linear_50node, the largest
    previous case, this is 2.4x the variables and 2.3x the rows but 5.0x
    the binaries -- because the assignment family is N(N+1)/2, quadratic
    in bus count, while the row count here is held down by the sparsity
    (179 edges at 118 buses is average degree 3.03, against 5.4 for the
    synthetic cases). Branch-and-bound difficulty tracks the binaries more
    closely than the row count, so this case is expected to be harder than
    its variable count alone suggests. How much harder is exactly what the
    cost probe is for.

    3600 s, matching linear_50node -- but read the ceiling differently
    here. MAX_OUTER_ITERATIONS is 1 in both run scripts (a deliberate cost
    probe; see the run scripts), so time_limit * MAX_OUTER_ITERATIONS is
    3600 s, and the limit IS the whole Gurobi budget rather than a
    per-pass slice of a larger one. It is still not a wall-clock ceiling:
    model construction, certificate generation, the counterexample search
    and the report all run outside Gurobi's clock, and at 118 buses the
    certificate families and the counterexample search are themselves
    substantial.

    A pass that hits the limit is a REPORTABLE RESULT here, more so than
    anywhere else in the study: the probe's question is how long one pass
    costs, and "more than an hour, with the incumbent and dual bound as
    reported" answers it. Do not raise this limit to force an OPTIMAL
    before the probe has been read.

    mip_gap = 1e-6 is tight on purpose, matching every other case. The
    outer loop reads the returned partition as if it were the inner
    optimum, so a loose gap would let solver slack, rather than the sampled
    stability constraints, decide which partition each pass hands back.
    """
    return MILPConfig(time_limit=3600.0, mip_gap=1e-6, verbose=False)
