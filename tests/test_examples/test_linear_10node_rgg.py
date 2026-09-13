"""Structural tests for the 10-node random-geometric-graph case.

Mirrors tests/test_examples/test_linear_20node.py where one exists, and
tests/test_examples/test_linear_30node.py otherwise. With random weights,
specific islands can't be hand-verified -- these check structural
invariants, not outcomes. None of these call solve().

TWO THINGS THIS CASE IS FOR, and both are tested here rather than left in
prose. First, it is drawn by the SAME rule as the 20-, 30- and 40-node
cases, so the generating parameters are pinned. Second, its fault
placement was chosen against a specific structure -- one articulation
point, a leaf behind a bridge, a four-edge bottleneck -- and those
premises are checked so a regenerated topology fails here rather than
silently invalidating the reasoning.
"""

import itertools
import pathlib

import networkx as nx
import numpy as np
import pytest

from examples.linear_10node_rgg.linear_10node_rgg_topology import (
    NODES, EDGES, NODE_WEIGHTS, EDGE_WEIGHTS, POSITIONS,
)
from examples.linear_10node_rgg.builder import (
    LINEAR_10_RGG, build_topology, build_M, build_problem, faulty_nodes,
    faulty_edges,
)
from examples.linear_10node_rgg.config import build_config
from examples.linear_10node_rgg.generate_linear_10node_rgg import (
    PLANE_SIZE, EDGE_RADIUS,
)

# The upper cluster and the four edges that hold the lower one on. Named
# because the fault placement is defined against them.
UPPER_CLUSTER = {1, 3, 5, 7, 8, 9}
LOWER_CLUSTER = {2, 4, 6, 10}
BOTTLENECK = [(1, 4), (1, 6), (1, 10), (8, 10)]


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


def test_counts_and_density():
    topology = build_topology()
    assert len(list(topology.nodes())) == 10
    assert len(list(topology.edges())) == len(EDGES) == 19
    # The generating parameter this case is built to share with the 30-,
    # 40- and 50-node generators. Pinned because it is the whole reason
    # the case exists: "the same rule at every size" is only true if the
    # rule's parameter is actually the same.
    assert PLANE_SIZE == pytest.approx(20.0 ** 0.5)
    assert 10 / PLANE_SIZE ** 2 == pytest.approx(0.500, abs=1e-9)


def test_average_degree_is_lower_than_the_larger_cases():
    # NOT a defect, and pinned so nobody "fixes" it. At N = 10 a larger
    # fraction of nodes sit near the boundary of the square and lose
    # neighbours within the radius, so holding node DENSITY fixed does not
    # hold degree fixed. The seed was not chosen to raise this figure.
    assert 2 * len(EDGES) / len(NODES) == pytest.approx(3.80)


def test_geometry_every_edge_within_the_radius():
    for u, v in EDGES:
        xu, yu = POSITIONS[u]
        xv, yv = POSITIONS[v]
        assert ((xu - xv) ** 2 + (yu - yv) ** 2) ** 0.5 <= EDGE_RADIUS + 1e-9


def test_the_edge_set_is_exactly_the_radius_rule():
    # The complement of the test above: every pair WITHIN the radius must
    # be an edge, not merely every edge within the radius. Together they
    # pin the unit-disc rule in both directions.
    derived = sorted(
        _canon((u, v)) for u, v in itertools.combinations(NODES, 2)
        if ((POSITIONS[u][0] - POSITIONS[v][0]) ** 2
            + (POSITIONS[u][1] - POSITIONS[v][1]) ** 2) ** 0.5 <= EDGE_RADIUS
    )
    assert derived == sorted(_canon(e) for e in EDGES)


def test_all_positions_lie_in_the_square():
    assert all(0.0 <= c <= PLANE_SIZE for p in POSITIONS.values() for c in p)


def test_full_system_is_genuinely_unstable():
    M = build_M(NODES, EDGES, NODE_WEIGHTS, EDGE_WEIGHTS)
    assert np.linalg.eigvalsh(M).min() <= 0


def test_faults_are_placed_where_intended():
    assert faulty_nodes(LINEAR_10_RGG) == [5]
    assert faulty_edges(LINEAR_10_RGG) == [(1, 7), (3, 8)]


def test_faults_are_real():
    graph = _graph()
    for node in faulty_nodes(LINEAR_10_RGG):
        diagonal = NODE_WEIGHTS[node] + sum(
            EDGE_WEIGHTS[_canon((node, j))] for j in graph.neighbors(node)
        )
        assert diagonal < 0 + 1e-9
    for edge in faulty_edges(LINEAR_10_RGG):
        assert EDGE_WEIGHTS[edge] == -20.0


def test_the_faulty_node_is_not_the_articulation_point():
    # Node 3 is the graph's ONLY articulation point, and it sits at the
    # near end of the bridge (3, 9): shedding it would strand the leaf 9
    # as a one-node island. That is the one node the placement had to
    # avoid, and this pins both halves of the reason.
    graph = _graph()
    assert sorted(nx.articulation_points(graph)) == [3]
    assert sorted(nx.bridges(graph)) == [(3, 9)]
    assert [n for n, d in graph.degree() if d == 1] == [9]
    assert 3 not in faulty_nodes(LINEAR_10_RGG)

    reduced = graph.copy()
    reduced.remove_nodes_from(faulty_nodes(LINEAR_10_RGG))
    assert nx.is_connected(reduced)


def test_the_faulty_node_was_chosen_over_its_structural_twin():
    # Nodes 5 and 7 are interchangeable structurally: same degree, and
    # each other's neighbour. The tiebreak was coupling -- node 5's full
    # neighbourhood carries more weight, so shedding it costs more. Pinned
    # because otherwise the choice looks arbitrary.
    graph = _graph()
    assert graph.degree(5) == graph.degree(7) == 4
    assert sorted(graph.neighbors(5)) == [1, 3, 7, 8]
    assert sorted(graph.neighbors(7)) == [1, 3, 5, 8]

    # Against the DRAWN weights, before the fault was applied to node 5.
    from examples.linear_10node_rgg.generate_linear_10node_rgg import (
        _generate_connected, DEFAULT_SEED,
    )
    _, rng, nodes, _, edges = _generate_connected(DEFAULT_SEED)
    _ = {n: float(rng.uniform(1.0, 5.0)) for n in nodes}
    healthy = {e: float(rng.uniform(1.0, 10.0)) for e in edges}
    coupling = {
        n: sum(healthy[_canon((n, j))] for j in graph.neighbors(n))
        for n in (5, 7)
    }
    assert coupling[5] > coupling[7]


def test_neither_endpoint_pair_of_the_faults_is_a_hidden_separator():
    # THE TRAP THE 30-BUS AND 118-BUS CASES BOTH HIT: a set of nodes that
    # are individually not articulation points can still be a cut set.
    # Neither 1 nor 8 is an articulation point, yet {1, 8} split this
    # graph 4/4, and {1, 10} strand {2, 4, 6}. This placement faults one
    # node so the question does not arise -- but the fact is recorded so
    # it is checked rather than remembered if the count ever rises.
    graph = _graph()
    for pair in ((1, 8), (1, 10)):
        assert not any(p in nx.articulation_points(graph) for p in pair)
        reduced = graph.copy()
        reduced.remove_nodes_from(pair)
        assert not nx.is_connected(reduced), f"{pair} was expected to separate"

    sizes = sorted(len(c) for c in nx.connected_components(
        nx.restricted_view(graph, [1, 8], [])))
    assert sizes == [4, 4]


def test_the_faulty_edges_avoid_the_bottleneck_and_the_faulty_node():
    graph = _graph()
    faults = [_canon(e) for e in faulty_edges(LINEAR_10_RGG)]

    for edge in faults:
        assert edge[0] in UPPER_CLUSTER and edge[1] in UPPER_CLUSTER
        assert edge not in [_canon(e) for e in BOTTLENECK]
        assert edge != (3, 9), "the bridge to the leaf must stay healthy"
        assert 5 not in edge, "a faulty edge must not touch the faulty node"

    # Vertex-disjoint, so losing both leaves no node at degree 1.
    assert not set(faults[0]) & set(faults[1])

    reduced = graph.copy()
    reduced.remove_nodes_from(faulty_nodes(LINEAR_10_RGG))
    reduced.remove_edges_from(faults)
    assert nx.is_connected(reduced)
    assert min(dict(reduced.degree()).values()) >= 1


def test_the_faulty_edges_force_a_split_the_graph_does_not():
    # THE CORRECTION. An earlier version of this test asserted that no
    # cut of at most three edges separates three or more nodes, and
    # concluded from it that the instance could not exercise
    # reconnection. The premise is true; the conclusion does not follow.
    #
    # A faulty edge does not need to form a graph CUT to force its
    # endpoints apart. Its -20 weight makes ANY island containing both
    # endpoints unstable, so the partition must separate them however
    # well connected the graph is -- and healthy edges crossing that
    # induced separation are then eligible reconnection candidates.
    graph = _graph()

    # The graph-theoretic fact, kept because it is true and worth pinning:
    # the connectivity bottlenecks are four edges wide, so two faulty
    # edges cannot open one.
    cuts = set()
    for size in range(3, 6):
        for side in itertools.combinations(NODES, size):
            side = set(side)
            other = set(NODES) - side
            if not nx.is_connected(graph.subgraph(side)):
                continue
            if not nx.is_connected(graph.subgraph(other)):
                continue
            cut = tuple(sorted(
                _canon((u, v)) for u, v in graph.edges()
                if (u in side) != (v in side)
            ))
            if len(cut) == 4:
                cuts.add(cut)
    assert sorted(cuts) == [
        ((1, 4), (1, 6), (1, 10), (8, 10)),   # 4/6, detaching {2,4,6,10}
        ((1, 5), (1, 7), (1, 8), (8, 10)),    # 5/5, detaching {1,2,4,6,10}
    ], "there are TWO four-edge cuts, not one"

    # ...and the separation the FAULTS induce, which is the one that
    # matters. Shedding node 5 and separating each faulty edge's endpoints
    # gives two connected, stable islands.
    island_a, island_b = {1, 2, 4, 6, 8, 10}, {3, 7, 9}
    reduced = graph.copy()
    reduced.remove_node(5)
    assert island_a | island_b == set(NODES) - {5}

    for island in (island_a, island_b):
        assert nx.is_connected(reduced.subgraph(island))
        induced = [
            (u, v) for u, v in reduced.edges() if u in island and v in island
        ]
        assert _lambda_min(island, induced) > 0

    for u, v in faulty_edges(LINEAR_10_RGG):
        assert (u in island_a) != (v in island_a), (
            f"faulty edge ({u},{v}) must have its endpoints separated"
        )


def test_a_healthy_edge_crosses_the_induced_separation():
    # WHY THIS CASE EXERCISES RECONNECTION. Three edges cross between the
    # two islands the faults induce: the two faulty ones, which are cut,
    # and (7,8), which is healthy -- so it is an eligible reconnection
    # edge joining two SUBSTANTIAL stable islands, six nodes and three,
    # not a one- or two-node stub.
    graph = _graph()
    island_a = {1, 2, 4, 6, 8, 10}
    reduced = graph.copy()
    reduced.remove_node(5)

    crossing = sorted(
        _canon((u, v)) for u, v in reduced.edges()
        if (u in island_a) != (v in island_a)
    )
    assert crossing == [(1, 7), (3, 8), (7, 8)]

    healthy = [e for e in crossing if EDGE_WEIGHTS[e] > 0]
    assert healthy == [(7, 8)]
    assert EDGE_WEIGHTS[(7, 8)] > 0

    # Both sides are substantial, which is what makes the candidate
    # meaningful rather than a stub reconnection.
    assert len(island_a) >= 3 and len(set(NODES) - {5} - island_a) >= 3


def test_the_reconnected_component_is_stable():
    # The union of the two islands joined by the healthy (7,8). Its
    # lambda_min is 3.017 -- a figure an earlier report quoted as evidence
    # that the instance stays ONE island and reconnects nothing. It is in
    # fact the reconnected component, so it was evidence of exactly the
    # opposite.
    graph = _graph()
    reduced = graph.copy()
    reduced.remove_node(5)
    reduced.remove_edges_from(faulty_edges(LINEAR_10_RGG))

    components = list(nx.connected_components(reduced))
    assert len(components) == 1
    nodes = sorted(components[0])
    assert nodes == [1, 2, 3, 4, 6, 7, 8, 9, 10]

    induced = [(u, v) for u, v in reduced.edges()]
    assert _lambda_min(nodes, induced) == pytest.approx(3.017465, abs=1e-5)


def test_build_problem_draws_matching_sample_shapes():
    problem = build_problem(LINEAR_10_RGG, build_topology(), build_config(), seed=0)
    assert problem.samples_1.shape == (problem.config.P_1, 10, 2)
    assert problem.samples_2.shape == (problem.config.P_2, 10, 2)
    assert problem.config.P_1 == problem.config.P_2


def test_the_generator_default_reproduces_the_committed_artefact():
    # Field by field, so a drift in the weights is caught as readily as
    # one in the topology.
    from examples.linear_10node_rgg import generate_linear_10node_rgg as generator

    result = generator.generate(generator.DEFAULT_SEED)

    assert generator.DEFAULT_SEED == 0
    # Seeds 0 and 1 draw disconnected graphs, so the retry loop lands on 2.
    assert result["seed"] == 2
    assert result["nodes"] == NODES
    assert result["edges"] == EDGES
    assert result["node_weights"] == NODE_WEIGHTS
    assert result["edge_weights"] == EDGE_WEIGHTS
    assert result["positions"] == POSITIONS
    assert result["faulty_nodes"] == faulty_nodes(LINEAR_10_RGG)
    assert result["faulty_edges"] == faulty_edges(LINEAR_10_RGG)


def test_the_retry_loop_actually_fires():
    # The recorded seed differs from DEFAULT_SEED only because seeds 0 and
    # 1 are disconnected. Asserted rather than assumed: if a future change
    # made seed 0 connected, the artefact would silently be a different
    # graph while the test above still passed.
    import networkx
    import numpy
    from examples.linear_10node_rgg import generate_linear_10node_rgg as generator

    for seed in (0, 1):
        rng = numpy.random.default_rng(seed)
        nodes, _, edges = generator._draw_positions_and_edges(rng)
        graph = networkx.Graph()
        graph.add_nodes_from(nodes)
        graph.add_edges_from(edges)
        assert not networkx.is_connected(graph), f"seed {seed} became connected"


def test_no_argument_main_writes_the_committed_bytes(tmp_path, monkeypatch):
    # The field-by-field check above compares generate()'s RETURN VALUE. A
    # regression in write_topology() -- a changed header, a dropped field,
    # a different repr -- or in main()'s argument handling would produce a
    # different FILE while that test still passed.
    import sys

    from examples.linear_10node_rgg import generate_linear_10node_rgg as generator

    committed = (
        pathlib.Path(generator.__file__).parent / "linear_10node_rgg_topology.py"
    )
    written = tmp_path / "linear_10node_rgg_topology.py"
    monkeypatch.setattr(generator, "OUTPUT_PATH", written)
    monkeypatch.setattr(sys, "argv", ["generate_linear_10node_rgg.py"])

    assert generator.DEFAULT_SEED == 0

    generator.main()

    assert written.read_bytes() == committed.read_bytes()


def test_the_faulty_node_rule_matches_the_other_cases():
    # w_i = -(full original-graph neighbourhood coupling) - 1. Checked
    # against the artefact rather than the generator, so a hand-edit to
    # the weights file is caught too. No faulty edge touches node 5, so
    # the coupling sum is the drawn one and the identity is exact up to
    # summation order.
    graph = _graph()
    for node in faulty_nodes(LINEAR_10_RGG):
        coupling = sum(
            EDGE_WEIGHTS[_canon((node, j))] for j in graph.neighbors(node)
        )
        assert NODE_WEIGHTS[node] == pytest.approx(-coupling - 1.0, abs=1e-12)
