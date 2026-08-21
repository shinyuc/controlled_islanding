"""Structural tests for the 30-node linear consensus example, mirroring
tests/test_examples/test_linear_50node.py one for one. With random
weights, specific islands can't be hand-verified -- these check
structural invariants, not outcomes. None of these call solve().
"""

import networkx as nx
import numpy as np

from examples.linear_30node.linear_30node_topology import (
    NODES, EDGES, NODE_WEIGHTS, EDGE_WEIGHTS, POSITIONS,
)
from examples.linear_30node.builder import (
    LINEAR_30, build_topology, build_M, build_problem, faulty_nodes,
    faulty_edges,
)
from examples.linear_30node.config import build_config


def _canon(edge):
    i, j = edge
    return (i, j) if i < j else (j, i)


def test_topology_is_connected():
    topology = build_topology()
    graph = nx.Graph()
    graph.add_nodes_from(topology.nodes())
    graph.add_edges_from(topology.edges())
    assert nx.is_connected(graph)


def test_counts():
    topology = build_topology()
    assert len(list(topology.nodes())) == 30
    assert len(list(topology.edges())) == len(EDGES)


def test_geometry_every_edge_within_distance_2():
    for u, v in EDGES:
        xu, yu = POSITIONS[u]
        xv, yv = POSITIONS[v]
        distance = ((xu - xv) ** 2 + (yu - yv) ** 2) ** 0.5
        assert distance <= 2.0 + 1e-9


def test_full_system_is_genuinely_unstable():
    M = build_M(NODES, EDGES, NODE_WEIGHTS, EDGE_WEIGHTS)
    lambda_min = np.linalg.eigvalsh(M).min()
    assert lambda_min <= 0


def test_faults_are_placed_where_intended():
    # Unlike the 20-/50-node cases, faults here are placed explicitly
    # (generate_linear_30node.py's FAULTY_NODES/FAULTY_EDGES), so the exact
    # identities are part of the fixture, not an artifact of "first n nodes".
    assert faulty_nodes(LINEAR_30) == [5, 16]
    assert faulty_edges(LINEAR_30) == [(10, 25), (11, 25), (12, 24), (18, 24)]


def test_faults_are_real():
    neighbors = {node: set() for node in NODES}
    for u, v in EDGES:
        neighbors[u].add(v)
        neighbors[v].add(u)

    # Each faulty node is unstable in EVERY island: its diagonal is
    # non-positive even at full original-graph coupling.
    for node in faulty_nodes(LINEAR_30):
        diagonal = NODE_WEIGHTS[node] + sum(
            EDGE_WEIGHTS[_canon((node, j))] for j in neighbors[node]
        )
        assert diagonal < 0 + 1e-9

    # Each faulty edge is strongly negative, so any island holding both
    # endpoints is unstable and the edge has to be cut.
    for edge in faulty_edges(LINEAR_30):
        assert EDGE_WEIGHTS[edge] == -20.0


def test_reconnection_candidate_survives_the_cut():
    # The placement's whole purpose: the four faulty edges are four of the
    # five crossing edges between the upper cluster {9,10,11,12,18,27,28}
    # and the rest, leaving (11,22) as the only healthy link across. That
    # is what can give the search two stable islands with an eligible cut
    # edge between them -- the precondition for a non-empty E^rec.
    upper = {9, 10, 11, 12, 18, 27, 28}
    crossing = sorted(
        _canon((u, v)) for u, v in EDGES if (u in upper) != (v in upper)
    )
    assert crossing == [(10, 25), (11, 22), (11, 25), (12, 24), (18, 24)]

    healthy = [edge for edge in crossing if EDGE_WEIGHTS[edge] > 0]
    assert healthy == [(11, 22)]


def test_build_problem_draws_matching_sample_shapes():
    problem = build_problem(LINEAR_30, build_topology(), build_config(), seed=0)

    assert problem.samples_1.shape == (problem.config.P_1, 30, 2)
    assert problem.samples_2.shape == (problem.config.P_2, 30, 2)
    assert problem.config.P_1 == problem.config.P_2
