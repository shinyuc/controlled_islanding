"""Tests for islanding/counterexample.py, reusing the 2-node local-search
fixture from docs/counterexample_spec.md section 7 (w_1=-1, w_2=3, w_12=3).
"""

import networkx as nx
import numpy as np
import pytest

from islanding.topology import Topology
from islanding.augmented import SOURCE, SINK
from islanding.cost import CostData, AlgorithmConfig, linear_consensus_model
from islanding.evaluate import Problem
from islanding.counterexample import find_counterexamples


def _fixture():
    topology = Topology(nodes=[1, 2], edges=[(1, 2)])

    node_weights = {1: -1.0, 2: 3.0}
    edge_weights = {(1, 2): 3.0}
    model = linear_consensus_model(node_weights, edge_weights)

    cost_data = CostData(node_costs={1: 1, 2: 1}, edge_costs={(1, 2): 1})

    config = AlgorithmConfig(
        kappa_op=1.0, kappa_rec=0.7,
        alpha_1=1.0, beta=1.0, k_1=0.2, P_1=1,
        alpha_2=1.0, gamma=1.0, k_2=0.2, P_2=1,
    )

    samples = np.zeros((1, 2, 1))
    problem = Problem(
        topology=topology, model=model, cost_data=cost_data, config=config,
        samples_1=samples, samples_2=samples,
    )
    return problem


def _n(i, j, endpoint):
    return ("n", i, j, endpoint)


def _tree_a0():
    # Islands {1}, {2} separate: edge (1,2) is "cut near i".
    n1, n2 = _n(1, 2, 1), _n(1, 2, 2)
    tree = nx.Graph()
    tree.add_edges_from([
        (SOURCE, SINK), (SOURCE, 1), (SOURCE, 2), (n1, n2), (n2, 2),
    ])
    return tree


def _tree_a1():
    # Merged island {1,2}: edge (1,2) is "through".
    n1, n2 = _n(1, 2, 1), _n(1, 2, 2)
    tree = nx.Graph()
    tree.add_edges_from([
        (SOURCE, SINK), (SOURCE, 1), (1, n1), (n1, n2), (n2, 2),
    ])
    return tree


def test_a0_yields_one_island_witness_for_node_1():
    # Island {1} alone: M = [-1], eigenvalue -1 <= 0 -> one witness. Island
    # {2}: M = [3], eigenvalue 3 > 0 -> no witness. Witness is scaled to
    # ||x|| = sqrt(2 * factor * beta) = sqrt(2) (factor = 1, no inflation),
    # zero-padded at node 2's position. Assert on magnitude, not sign.
    problem = _fixture()
    island_witnesses, _ = find_counterexamples(_tree_a0(), problem)

    assert len(island_witnesses) == 1
    witness = island_witnesses[0]
    assert witness.shape == (2, 1)

    expected_radius = (2.0 * 1.0 * problem.config.beta) ** 0.5
    assert np.linalg.norm(witness) == pytest.approx(expected_radius)
    assert abs(witness[0, 0]) == pytest.approx(expected_radius)
    assert witness[1, 0] == pytest.approx(0.0)


def test_a0_component_witnesses_mirror_island_witnesses():
    # A0 has no reconnection (edge (1,2) is "cut near i", not middle-cut),
    # so components == stable islands unmerged: component {1} alone
    # reproduces the same M = [-1] as island {1} (beta == gamma == 1 in
    # this fixture, so the radius matches too).
    problem = _fixture()
    island_witnesses, component_witnesses = find_counterexamples(_tree_a0(), problem)

    assert len(component_witnesses) == 1
    assert component_witnesses[0].shape == (2, 1)
    assert np.linalg.norm(component_witnesses[0]) == pytest.approx(
        np.linalg.norm(island_witnesses[0])
    )
    assert abs(component_witnesses[0][0, 0]) == pytest.approx(
        abs(island_witnesses[0][0, 0])
    )
    assert component_witnesses[0][1, 0] == pytest.approx(0.0)


def test_a1_yields_no_witnesses():
    # Merged island {1,2}: M = [[2,-3],[-3,6]], eigenvalues ~0.394 and
    # ~7.6, both > 0 -- genuinely stable, no witness for either the island
    # (F5 side) or its (identical, unreconnected) component (F6 side).
    problem = _fixture()
    island_witnesses, component_witnesses = find_counterexamples(_tree_a1(), problem)

    assert island_witnesses == []
    assert component_witnesses == []
