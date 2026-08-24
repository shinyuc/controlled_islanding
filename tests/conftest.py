"""Shared pytest fixtures for the islanding test suite.

Provides the toy network cases used across test modules. Each case is
split into a topology fixture (undirected graph: nodes and edges only)
and a dynamics fixture (a model built on that topology, e.g. linear
consensus with signed edge weights).

Fixtures shared by more than one test module live here rather than in
whichever module happened to define them first: conftest.py is the only
file pytest guarantees is importable from every test module, under any
invocation (`pytest`, `pytest tests/test_x.py`, `python -m pytest`).
Cross-importing between test modules is not -- tests/ is not a package
and is not in package discovery, so `from tests.test_cost import ...`
resolves only when the repo root happens to be on sys.path.
"""

from dataclasses import replace

import networkx as nx
import numpy as np
import pytest

from islanding.topology import Topology
from islanding.dynamics import LinearConsensus
from islanding.cost import CostData, AlgorithmConfig, linear_consensus_model
from islanding.evaluate import Problem


@pytest.fixture
def toy10_topology():
    """10-node connected topology (Petersen graph: 10 nodes, 15 edges).

    Pure connectivity only -- no weights, signs, or dynamics. The
    partition into islands is an output of the algorithm, not encoded here.
    """
    g = nx.petersen_graph()
    return Topology(nodes=g.nodes(), edges=g.edges())


@pytest.fixture
def toy10_dynamics(toy10_topology):
    """Signed linear-consensus weights on toy10_topology.

    Some edges are negative so the full system is not PSD (unstable
    before islanding), matching the paper's setup. Whether any island
    is stable is computed by the certificate test, not asserted here.
    """
    edges = list(toy10_topology.edges())
    weights = {e: 1.0 for e in edges}
    # make a few edges negative -> full Laplacian not PSD
    for e in edges[:4]:
        weights[e] = -2.0
    return LinearConsensus(topology=toy10_topology, weights=weights)


@pytest.fixture
def synthetic_edge_energy_model():
    """Factory for spec section 16b's synthetic model -- TEST FIXTURE ONLY.

    Not a physical system model, so it lives here rather than beside
    linear_consensus_model in islanding/cost.py. Its only purpose is to
    make W_ij nonzero, which is what brings Psi_r's Parts 2 and 3 to life;
    on linear consensus both are identically zero and untestable.

    The returned callable derives a model from `base_model` by swapping
    exactly two callables, so the dynamics (f_i, f_ij), the node energy
    (W_i, grad_W_i), the sampler and the inflation rule are the
    linear-consensus ones, closed over the same node/edge weights. Any
    deviation from section 16's result is therefore attributable purely to
    the edge-energy terms.

        W_ij(i, j, x_i, x_j)      = 0.5 * (x_i - x_j)^2       (c = 1)
        grad_W_ij(i, j, x_i, x_j) = (x_i - x_j, -(x_i - x_j))

    Note this model is order-EQUIVARIANT: grad_W_ij(j, i, ...)[0] equals
    grad_W_ij(i, j, ...)[1], as any physical edge energy would be. It
    therefore cannot distinguish a certificate that takes the j-side
    partial from the canonical call from one that re-calls with the
    arguments reversed. tests/test_baseline.py has a separate,
    deliberately non-equivariant mock for that.
    """

    def build(base_model):
        def W_ij(i, j, x_i, x_j):
            delta = x_i - x_j
            return 0.5 * float(np.dot(delta, delta))

        def grad_W_ij(i, j, x_i, x_j):
            delta = x_i - x_j
            return (delta, -delta)

        # island_quadratic_form MUST be cleared. It is inherited by
        # dataclasses.replace from the linear-consensus base, but its
        # closed form assumes W_ij == 0 -- which is exactly what this
        # model breaks. Leaving it set makes F5's fast path compute a
        # different (wrong) value while every signature still matches:
        # on the 10-node fixture it gave 82.58 where the true value is
        # 30.16, and only tests/test_baseline.py's MILP-vs-cost equality
        # noticed.
        return replace(
            base_model, W_ij=W_ij, grad_W_ij=grad_W_ij,
            island_quadratic_form=None,
        )

    return build


def build_two_node_problem():
    """The 2-node outer-loop fixture (docs/outer_loop_spec.md section 6).

    w_1 = -1, w_2 = 3, w_12 = 3, beta = gamma = 1. Island {1} alone has
    M = [-1], so the counterexample search yields the node-1 witness and
    the augment path can be exercised deterministically.

    Lives here because more than one test module needs it, and conftest is
    the only file pytest guarantees is importable from all of them --
    `from tests.test_outer_loop import ...` resolves only when the repo
    root happens to be on sys.path. Same reasoning as
    [synthetic_edge_energy_model] above.
    """
    topology = Topology(nodes=[1, 2], edges=[(1, 2)])
    model = linear_consensus_model({1: -1.0, 2: 3.0}, {(1, 2): 3.0})
    cost_data = CostData(node_costs={1: 1, 2: 1}, edge_costs={(1, 2): 1})
    config = AlgorithmConfig(
        kappa_op=1.0, kappa_rec=0.7,
        alpha_1=1.0, beta=1.0, k_1=0.2, P_1=1,
        alpha_2=1.0, gamma=1.0, k_2=0.2, P_2=1,
    )
    # Single sample point x = (1, 0) -- NOT the node-1 witness (sqrt(2), 0):
    # same direction, different magnitude, so it is not literally present
    # in the initial sample sets.
    samples = np.array([[[1.0], [0.0]]])
    return Problem(
        topology=topology, model=model, cost_data=cost_data, config=config,
        samples_1=samples, samples_2=samples,
    )


@pytest.fixture
def two_node_problem():
    """A fresh build_two_node_problem() per test."""
    return build_two_node_problem()
