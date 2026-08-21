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

        return replace(base_model, W_ij=W_ij, grad_W_ij=grad_W_ij)

    return build
