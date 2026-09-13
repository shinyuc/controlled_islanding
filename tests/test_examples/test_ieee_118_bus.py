"""Structural tests for the IEEE 118-bus case.

Mirrors tests/test_examples/test_ieee_30_bus.py. Same two departures from
the synthetic cases -- no geometry test, because the CDF is a bus/branch
list with no coordinates; and extra placement tests, because the fault
placement was chosen against this topology's structure.

Two things are specific to this case.

THE PARALLEL-LINE COLLAPSE. The CDF branch section holds 186 rows but the
graph has 179 edges: seven bus pairs are joined by more than one circuit.
The gap is checked here so it can never be mistaken for a parsing error
or quietly grow.

THE JOINT FAULT CHECK. The placement's premise is that shedding all four
faulty buses leaves ONE component. That does not follow from each bus
individually not being an articulation point -- the IEEE 30-bus case is
the worked counterexample, where buses 6 and 10 are individually fine but
jointly strand {9, 11}. So the four are tested together, and a fifth is
tested to show where the property stops holding.

With random weights, specific islands can't be hand-verified -- these
check structural invariants, not outcomes. None of these call solve().
"""

import pathlib

import networkx as nx
import numpy as np
import pytest

from examples.ieee_118_bus.ieee_118_bus_topology import NODES, EDGES
from examples.ieee_118_bus.ieee_118_bus_weights import NODE_WEIGHTS, EDGE_WEIGHTS
from examples.ieee_118_bus.builder import (
    IEEE_118, build_topology, build_M, build_problem, faulty_nodes,
    faulty_edges,
)
from examples.ieee_118_bus.config import build_config


ARTICULATION_POINTS = [8, 9, 12, 68, 71, 85, 86, 100, 110]
FAULTY = [17, 37, 59, 80]
# The fifth high-degree bus that breaks the single-component property.
FIFTH = 77
BRANCH_ROWS = 186


def _canon(edge):
    i, j = edge
    return (i, j) if i < j else (j, i)


def _graph():
    graph = nx.Graph()
    graph.add_nodes_from(NODES)
    graph.add_edges_from(EDGES)
    return graph


def _lambda_min(nodes, edges):
    return float(np.linalg.eigvalsh(
        build_M(sorted(nodes), edges, NODE_WEIGHTS, EDGE_WEIGHTS)
    ).min())


def test_topology_is_connected():
    topology = build_topology()
    graph = nx.Graph()
    graph.add_nodes_from(topology.nodes())
    graph.add_edges_from(topology.edges())
    assert nx.is_connected(graph)


def test_counts():
    topology = build_topology()
    assert len(list(topology.nodes())) == 118
    assert len(list(topology.edges())) == len(EDGES) == 179
    assert sorted(NODES) == list(range(1, 119))


def test_the_parallel_lines_collapsed_and_the_count_is_recorded():
    # 186 branch rows, 179 edges. Checked three ways so the discrepancy
    # cannot be mistaken for a truncated parse: the edge list holds no
    # duplicates or self-loops, the arithmetic is stated, and the
    # artefact's own header records both figures.
    canon = [_canon(e) for e in EDGES]
    assert len(set(canon)) == len(canon) == 179, "duplicate edges in the artefact"
    assert not [e for e in canon if e[0] == e[1]], "self-loop in the artefact"
    assert BRANCH_ROWS - len(canon) == 7

    from examples.ieee_118_bus import ieee_118_bus_topology as module

    header = (module.__doc__ or "") + pathlib.Path(module.__file__).read_text()[:400]
    assert "186" in header and "179" in header, (
        "the artefact does not record both the branch-row and edge counts, "
        "so the 7-edge gap reads as a parsing error"
    )


def test_the_topology_module_carries_no_weights():
    # PROVENANCE, enforced rather than documented. Only the topology comes
    # from the IEEE test case; the dynamics and weights are ours. If the
    # parser's output ever grew weight dictionaries -- line impedances,
    # say -- builder.py's two-module import would still work while the
    # case quietly became a different thing. This fails first.
    from examples.ieee_118_bus import ieee_118_bus_topology as topology_module

    for name in ("NODE_WEIGHTS", "EDGE_WEIGHTS", "POSITIONS"):
        assert not hasattr(topology_module, name), (
            f"{name} appeared in the parsed topology module; the case's "
            "weights must come from generate_ieee_118_bus.py alone"
        )


def test_full_system_is_genuinely_unstable():
    M = build_M(NODES, EDGES, NODE_WEIGHTS, EDGE_WEIGHTS)
    assert np.linalg.eigvalsh(M).min() <= 0


def test_faults_are_placed_where_intended():
    assert faulty_nodes(IEEE_118) == FAULTY
    # NO faulty edges, deliberately -- see generate_ieee_118_bus.py. This
    # case is for the scaling comparison, not for exercising reconnection.
    assert faulty_edges(IEEE_118) == []


def test_no_edge_carries_a_fault_weight():
    # The complement of the assertion above, against the artefact rather
    # than the generator: an empty FAULTY_EDGES must mean every branch
    # kept its drawn weight, not that the list merely went unread.
    assert all(weight > 0 for weight in EDGE_WEIGHTS.values())
    assert min(EDGE_WEIGHTS.values()) >= 1.0
    assert max(EDGE_WEIGHTS.values()) <= 10.0


def test_faults_are_real():
    graph = _graph()

    # Each faulty bus is unstable in EVERY island: its diagonal is
    # non-positive even at full original-graph coupling.
    for node in faulty_nodes(IEEE_118):
        diagonal = NODE_WEIGHTS[node] + sum(
            EDGE_WEIGHTS[_canon((node, j))] for j in graph.neighbors(node)
        )
        assert diagonal < 0 + 1e-9


def test_the_faulty_buses_are_well_connected():
    # The stated reason for choosing them: degree 6 or 7 where the average
    # is 3.03, so shedding them costs real coupling. Exactly ten buses
    # reach degree 6, and all four faults are among them.
    graph = _graph()
    degrees = {node: graph.degree(node) for node in FAULTY}
    assert degrees == {17: 6, 37: 6, 59: 6, 80: 7}

    average = 2 * graph.number_of_edges() / graph.number_of_nodes()
    assert average == pytest.approx(3.0339, abs=1e-3)

    # NOT "the four highest degree", which would be false: buses 49
    # (degree 9), 100 (8) and 12 (7) rank above three of these, and 100
    # and 12 are articulation points. The claim is a high-degree spread
    # that stays connected when shed, so the test pins the rank band
    # rather than a ranking the placement does not make.
    ranked = [node for _, node in sorted(
        ((d, n) for n, d in graph.degree()), reverse=True
    )]
    assert all(ranked.index(node) < 10 for node in FAULTY)
    assert sum(1 for _, d in graph.degree() if d >= 6) == 10


def test_none_of_the_faulty_buses_is_an_articulation_point():
    # Individually. That is ALL this establishes -- see the joint test
    # below, which is the one the placement actually depends on.
    graph = _graph()
    assert sorted(nx.articulation_points(graph)) == ARTICULATION_POINTS
    assert not [node for node in FAULTY if node in ARTICULATION_POINTS]

    for node in FAULTY:
        reduced = graph.copy()
        reduced.remove_node(node)
        assert nx.is_connected(reduced)


def test_shedding_all_four_faulty_buses_leaves_one_component():
    # THE PLACEMENT'S PREMISE, and the one that does not follow from the
    # test above. A set of individually non-cut vertices can still be a
    # cut set: in the IEEE 30-bus case buses 6 and 10 are each fine alone
    # but together strand {9, 11}. Here the four together leave a single
    # 114-bus component, so no island is forced by structure and the
    # island count stays a search outcome.
    graph = _graph()
    reduced = graph.copy()
    reduced.remove_nodes_from(FAULTY)

    components = list(nx.connected_components(reduced))
    assert len(components) == 1
    assert len(components[0]) == 114

    # And shedding them cuts 25 edges -- the coupling F1 is paying for.
    incident = [
        (u, v) for u, v in graph.edges() if u in FAULTY or v in FAULTY
    ]
    assert len(incident) == 25


def test_the_114_bus_remainder_is_stable():
    # Whether the instance has a feasible answer at all. If the remainder
    # were unstable there would be no stable island containing it, and the
    # search would have to fragment it further for reasons that have
    # nothing to do with the objective.
    graph = _graph()
    reduced = graph.copy()
    reduced.remove_nodes_from(FAULTY)

    nodes = sorted(reduced.nodes())
    assert _lambda_min(nodes, list(reduced.edges())) > 0


def test_a_fifth_faulty_bus_would_fragment_the_remainder():
    # Why the placement stops at four. Adding bus 77 breaks the remainder
    # into three pieces, which would make the island count a fact about
    # the fault set rather than a search outcome.
    graph = _graph()
    reduced = graph.copy()
    reduced.remove_nodes_from(FAULTY + [FIFTH])

    sizes = sorted(len(c) for c in nx.connected_components(reduced))
    assert sizes == [2, 31, 80]


def test_build_problem_draws_matching_sample_shapes():
    problem = build_problem(IEEE_118, build_topology(), build_config(), seed=0)

    assert problem.samples_1.shape == (problem.config.P_1, 118, 2)
    assert problem.samples_2.shape == (problem.config.P_2, 118, 2)
    assert problem.config.P_1 == problem.config.P_2


def test_the_generator_default_reproduces_the_committed_artefact():
    # Field by field, so a drift in the weights is caught as readily as a
    # drift in which buses were faulted.
    from examples.ieee_118_bus import generate_ieee_118_bus as generator

    result = generator.generate(generator.DEFAULT_SEED)

    assert generator.DEFAULT_SEED == 0
    assert result["seed"] == 0
    assert result["nodes"] == NODES
    assert result["edges"] == sorted(_canon(e) for e in EDGES)
    assert result["node_weights"] == NODE_WEIGHTS
    assert result["edge_weights"] == EDGE_WEIGHTS
    assert result["faulty_nodes"] == faulty_nodes(IEEE_118)
    assert result["faulty_edges"] == faulty_edges(IEEE_118)


def test_no_argument_main_writes_the_committed_bytes(tmp_path, monkeypatch):
    # The field-by-field check above compares generate()'s RETURN VALUE. A
    # regression in write_weights() -- a changed header, a dropped field, a
    # different repr -- or in main()'s argument handling would produce a
    # different FILE while that test still passed. This drives the real
    # no-argument entry point and compares bytes.
    import sys

    from examples.ieee_118_bus import generate_ieee_118_bus as generator

    committed = (
        pathlib.Path(generator.__file__).parent / "ieee_118_bus_weights.py"
    )
    written = tmp_path / "ieee_118_bus_weights.py"
    monkeypatch.setattr(generator, "OUTPUT_PATH", written)
    monkeypatch.setattr(sys, "argv", ["generate_ieee_118_bus.py"])

    assert generator.DEFAULT_SEED == 0

    generator.main()

    assert written.read_bytes() == committed.read_bytes()


def test_the_faulty_node_rule_matches_the_other_cases():
    # w_i = -(full original-graph neighbourhood coupling) - 1, the same
    # rule every other case uses. Checked against the artefact rather than
    # the generator, so a hand-edit to the weights file is caught too. No
    # branch is faulted here, so the coupling sum is exactly the drawn one
    # and the identity is exact.
    graph = _graph()

    for node in faulty_nodes(IEEE_118):
        coupling = sum(
            EDGE_WEIGHTS[_canon((node, j))] for j in graph.neighbors(node)
        )
        # approx, not ==. The rule is exact IN THE GENERATOR -- it
        # stores -sum - 1.0 -- but this test re-derives the sum from a
        # networkx neighbour iteration, which need not add the terms in
        # the generator's order, and float addition is not associative.
        # The same effect is why the fully-coupled diagonal comes out at
        # -0.9999999999999964 on one bus rather than exactly -1.0.
        assert NODE_WEIGHTS[node] == pytest.approx(-coupling - 1.0, abs=1e-12)
