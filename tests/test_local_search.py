"""Tests for islanding/local_search.py, reproducing the worked example in
docs/local_search_spec.md section 7 (2-node fixture: node 1 faulty in
isolation but rescued by coupling to node 2).
"""

from dataclasses import replace

import networkx as nx
import numpy as np
import pytest

from islanding.topology import Topology
from islanding.augmented import SOURCE, SINK
from islanding.cost import CostData, AlgorithmConfig, linear_consensus_model
from islanding.evaluate import Problem
from islanding.local_search import local_search


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
        epsilon=0.01, max_iterations=100, swap_supernode_edges=True,
    )

    # Single sample point x = (1, 0): x_1 = 1, x_2 = 0. Node order is
    # sorted(topology.nodes()) = [1, 2].
    samples = np.array([[[1.0], [0.0]]])

    problem = Problem(
        topology=topology, model=model, cost_data=cost_data, config=config,
        samples_1=samples, samples_2=samples,
    )
    return problem


def _n(i, j, endpoint):
    return ("n", i, j, endpoint)


def _tree_a0():
    # Nodes as separate islands: (s0,d0),(s0,1),(s0,2),(n1,n2),(n2,2).
    # Edge (1,2) is "cut near i": (1,n1) absent.
    n1, n2 = _n(1, 2, 1), _n(1, 2, 2)
    tree = nx.Graph()
    tree.add_edges_from([
        (SOURCE, SINK), (SOURCE, 1), (SOURCE, 2), (n1, n2), (n2, 2),
    ])
    return tree


def _tree_a1():
    # One swap from A0: remove (s0,2), add (1,n1) -- edge (1,2) is now
    # "through", so {1,2} is a single stable island (already optimal).
    n1, n2 = _n(1, 2, 1), _n(1, 2, 2)
    tree = nx.Graph()
    tree.add_edges_from([
        (SOURCE, SINK), (SOURCE, 1), (1, n1), (n1, n2), (n2, 2),
    ])
    return tree


def _tree_a2():
    # Stable island {1}, unstable island {2}: edge (1,2) is "cut near j"
    # (the (n2,2) stub absent). The only fixture tree with an unstable
    # island, so the only one where F1 is nonzero.
    n1, n2 = _n(1, 2, 1), _n(1, 2, 2)
    tree = nx.Graph()
    tree.add_edges_from([
        (SOURCE, SINK), (SOURCE, 1), (SINK, 2), (1, n1), (n1, n2),
    ])
    return tree


def test_local_search_from_a0_reaches_zero_in_one_swap():
    problem = _fixture()
    result = local_search(problem, initial_tree=_tree_a0())

    assert result.objective == pytest.approx(0.0, abs=1e-9)
    assert result.info["reason"] == "zero_objective"
    assert result.info["iterations"] == 1


def test_local_search_from_a1_already_optimal():
    problem = _fixture()
    result = local_search(problem, initial_tree=_tree_a1())

    assert result.objective == pytest.approx(0.0, abs=1e-9)
    assert result.info["reason"] == "zero_objective"
    assert result.info["iterations"] == 0


def test_local_search_without_supernode_swaps_cannot_merge_islands():
    problem = _fixture()
    problem = replace(problem, config=replace(problem.config, swap_supernode_edges=False))

    result = local_search(problem, initial_tree=_tree_a0())

    # Every supernode edge present in A0 is still present in the result --
    # the restricted search can never remove or add one, so it cannot
    # reach {1,2} as a single island.
    assert result.info["tree"].has_edge(SOURCE, 1)
    assert result.info["tree"].has_edge(SOURCE, 2)
    assert result.info["tree"].has_edge(SOURCE, SINK)
    assert not result.info["tree"].has_edge(SINK, 1)
    assert not result.info["tree"].has_edge(SINK, 2)

    assert result.info["reason"] == "local_optimum"


def test_local_search_max_iterations_zero_stops_immediately():
    problem = _fixture()
    problem = replace(problem, config=replace(problem.config, max_iterations=0))

    result = local_search(problem, initial_tree=_tree_a0())

    assert result.info["reason"] == "max_iterations"
    assert result.info["iterations"] == 0


def test_local_search_from_a2_reaches_zero_in_one_swap():
    # A2 has stable island {1} and unstable island {2} (F_hat=4.2). One
    # swap -- remove (SINK,2), add (n2,2) -- yields A1 (F_hat=0). Assert
    # on the objective and reason, not on a specific tree.
    problem = _fixture()
    result = local_search(problem, initial_tree=_tree_a2())

    assert result.objective == pytest.approx(0.0, abs=1e-9)
    assert result.info["reason"] == "zero_objective"
    assert result.info["iterations"] == 1
