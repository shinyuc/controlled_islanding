"""Shared pytest fixtures for the islanding test suite.

Provides the toy network cases used across test modules. Each case is
split into a topology fixture (undirected graph: nodes and edges only)
and a dynamics fixture (a model built on that topology, e.g. linear
consensus with signed edge weights).
"""

import networkx as nx
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