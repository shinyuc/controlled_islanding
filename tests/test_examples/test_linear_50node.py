"""Structural tests for the 50-node linear consensus example (spec
section 7). At 50 nodes with random weights, specific islands can't be
hand-verified -- these check structural invariants, not outcomes. None
of these call solve().
"""

import networkx as nx
import numpy as np

from examples.linear_50node.linear_50node_topology import (
    NODES, EDGES, NODE_WEIGHTS, EDGE_WEIGHTS, POSITIONS,
)
from examples.linear_50node.builder import (
    LINEAR_50, build_topology, build_M, build_problem, faulty_nodes,
)
from examples.linear_50node.config import build_config


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
    assert len(list(topology.nodes())) == 50
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


def test_faults_are_real():
    nodes = faulty_nodes(LINEAR_50)
    assert len(nodes) == 7

    neighbors = {node: set() for node in NODES}
    for u, v in EDGES:
        neighbors[u].add(v)
        neighbors[v].add(u)

    for node in nodes:
        diagonal = NODE_WEIGHTS[node] + sum(
            EDGE_WEIGHTS[_canon((node, j))] for j in neighbors[node]
        )
        assert diagonal < 0 + 1e-9


def test_build_problem_draws_matching_sample_shapes():
    problem = build_problem(LINEAR_50, build_topology(), build_config(), seed=0)

    assert problem.samples_1.shape == (problem.config.P_1, 50, 2)
    assert problem.samples_2.shape == (problem.config.P_2, 50, 2)
    assert problem.config.P_1 == problem.config.P_2
