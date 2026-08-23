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
from islanding.augmented import augmented_graph
from islanding.search import initial_spanning_tree
from islanding.local_search import local_search, _feasible_swaps


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
    # A2 has stable island {1} and unstable island {2} (F_hat=3.1). One
    # swap -- remove (SINK,2), add (n2,2) -- yields A1 (F_hat=0). Assert
    # on the objective and reason, not on a specific tree.
    problem = _fixture()
    result = local_search(problem, initial_tree=_tree_a2())

    assert result.objective == pytest.approx(0.0, abs=1e-9)
    assert result.info["reason"] == "zero_objective"
    assert result.info["iterations"] == 1


# ---------------------------------------------------------------------------
# The enumerated neighbourhood contains no no-op swaps
# ---------------------------------------------------------------------------

def _tree_key(tree):
    return frozenset(frozenset(edge) for edge in tree.edges())


_SWAP_TOPOLOGIES = [
    Topology(nodes=[1, 2], edges=[(1, 2)]),
    Topology(nodes=[1, 2, 3], edges=[(1, 2), (2, 3)]),
    Topology(nodes=[1, 2, 3], edges=[(1, 2), (2, 3), (1, 3)]),
    Topology(nodes=[1, 2, 3, 4], edges=[(1, 2), (1, 3), (1, 4)]),
]
_SWAP_IDS = ["two_node_edge", "three_node_path", "triangle", "star"]


@pytest.mark.parametrize("topology", _SWAP_TOPOLOGIES, ids=_SWAP_IDS)
def test_no_enumerated_swap_rebuilds_the_same_tree(topology):
    # "remove (SOURCE, i), add (SOURCE, i)" used to pass the feasibility
    # test -- removing the edge really does separate its endpoints -- and
    # returned the original tree. Epsilon > 0 meant it could never be
    # accepted, so no result changed, but each one cost a full objective
    # evaluation and the neighbourhood was not the one the spec describes.
    tree = initial_spanning_tree(augmented_graph(topology))
    original = _tree_key(tree)

    for removal, addition in _feasible_swaps(tree, topology, True):
        swapped = tree.copy()
        swapped.remove_edge(*removal)
        swapped.add_edge(*addition)

        assert _tree_key(swapped) != original, (
            f"swap remove={removal} add={addition} rebuilds the same tree"
        )


@pytest.mark.parametrize("topology", _SWAP_TOPOLOGIES, ids=_SWAP_IDS)
def test_no_swap_adds_an_edge_the_tree_already_has(topology):
    # The direct form of the same rule, and the one the candidate
    # generators enforce.
    tree = initial_spanning_tree(augmented_graph(topology))

    for removal, addition in _feasible_swaps(tree, topology, True):
        assert not tree.has_edge(*addition), (
            f"addition {addition} is already in the tree"
        )
        assert set(removal) != set(addition)


@pytest.mark.parametrize("topology", _SWAP_TOPOLOGIES, ids=_SWAP_IDS)
def test_filtering_removes_only_no_ops_and_no_genuine_swap(topology):
    # The guarantee that matters: filtering must not shrink the set of
    # trees reachable in one step. Recomputed here from the UNFILTERED
    # candidate set, so this compares the two neighbourhoods directly
    # rather than trusting that only no-ops were dropped.
    tree = initial_spanning_tree(augmented_graph(topology))
    original = _tree_key(tree)

    filtered = _feasible_swaps(tree, topology, True)
    reachable = {
        _tree_key(_apply(tree, removal, addition))
        for removal, addition in filtered
    }

    unfiltered_reachable = set()
    for removal, addition in _unfiltered_swaps(tree, topology):
        result = _tree_key(_apply(tree, removal, addition))
        if result != original:          # the no-ops, which we mean to drop
            unfiltered_reachable.add(result)

    assert reachable == unfiltered_reachable
    # ...and every enumerated pair now leads somewhere distinct.
    assert len(reachable) == len(filtered)


def _apply(tree, removal, addition):
    swapped = tree.copy()
    swapped.remove_edge(*removal)
    swapped.add_edge(*addition)
    return swapped


def _unfiltered_swaps(tree, topology):
    """The pre-fix enumeration: every supernode edge offered, filtered or not."""
    import networkx as nx

    from islanding.local_search import (
        _removal_candidates, _relay_addition_candidates,
    )

    additions = list(_relay_addition_candidates(tree, topology))
    for node in topology.nodes():
        additions.append((SOURCE, node))
        additions.append((SINK, node))

    swaps = []
    for removal in _removal_candidates(tree, True):
        remainder = tree.copy()
        remainder.remove_edge(*removal)
        component_of = {}
        for index, component in enumerate(nx.connected_components(remainder)):
            for node in component:
                component_of[node] = index
        for addition in additions:
            u, v = addition
            if component_of.get(u) != component_of.get(v):
                swaps.append((removal, addition))
    return swaps
