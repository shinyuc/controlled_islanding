"""Structural tests for the 50-node linear consensus example (spec
section 7). At 50 nodes with random weights, specific islands can't be
hand-verified -- these check structural invariants, not outcomes. None
of these call solve().
"""

import pathlib

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


def test_the_generator_default_reproduces_the_committed_artefact():
    # This case's default seed used to be 0 while the artefact records 1.
    # That reproduced anyway -- seed 0's draw is disconnected, so the retry
    # loop advanced to 1 -- but only as an incidental property of this
    # draw. DEFAULT_SEED now names 1 outright; see the generator.
    from examples.linear_50node import generate_linear_50node as generator

    result = generator.generate(generator.DEFAULT_SEED, generator.DEFAULT_N_FAULTS)

    assert generator.DEFAULT_SEED == 1
    assert result["seed"] == 1
    assert result["nodes"] == NODES
    assert result["edges"] == EDGES
    assert result["node_weights"] == NODE_WEIGHTS
    assert result["edge_weights"] == EDGE_WEIGHTS
    assert result["positions"] == POSITIONS


def test_no_argument_main_writes_the_committed_bytes(tmp_path, monkeypatch):
    # As for the 30- and 40-node cases: the check above compares
    # generate()'s return value, so a regression in write_topology() or in
    # main()'s argument handling would produce a different FILE while it
    # still passed. This drives the real no-argument entry point and
    # compares bytes.
    import sys

    from examples.linear_50node import generate_linear_50node as generator

    committed = (
        pathlib.Path(generator.__file__).parent / "linear_50node_topology.py"
    )
    written = tmp_path / "linear_50node_topology.py"
    monkeypatch.setattr(generator, "OUTPUT_PATH", written)
    monkeypatch.setattr(sys, "argv", ["generate_linear_50node.py"])

    # THE ASSERTION THAT MATTERS HERE. The byte comparison below would
    # still pass with DEFAULT_SEED = 0: seed 0's draw fails connectivity,
    # the retry loop advances to 1, and identical bytes result. That
    # incidental behaviour is precisely what naming the seed was meant to
    # eliminate, so it has to be checked directly rather than through its
    # output.
    assert generator.DEFAULT_SEED == 1

    generator.main()

    assert written.read_bytes() == committed.read_bytes()
