"""Structural tests for the IEEE 30-bus case.

Mirrors tests/test_examples/test_linear_30node.py, with two deliberate
differences.

NO GEOMETRY TEST. The synthetic cases draw positions on a plane and
connect pairs within a fixed radius, so "every edge is within distance 2"
is a real invariant there. The IEEE case is a bus/branch list with no
coordinates: there are no POSITIONS to test, and an edge here means a
branch in the test case, not a distance.

EXTRA PLACEMENT TESTS. The fault placement was chosen against this
topology's structure -- articulation points, the one small edge cut -- and
those premises are recorded as comments in generate_ieee_30_bus.py. The
tests below check them, so a re-parse that changed the topology fails here
rather than silently invalidating the reasoning.

With random weights, specific islands can't be hand-verified -- these
check structural invariants, not outcomes. None of these call solve().
"""

import itertools
import pathlib

import networkx as nx
import numpy as np
import pytest

from examples.ieee_30_bus.ieee_30_bus_topology import NODES, EDGES
from examples.ieee_30_bus.ieee_30_bus_weights import NODE_WEIGHTS, EDGE_WEIGHTS
from examples.ieee_30_bus.builder import (
    IEEE_30, build_topology, build_M, build_problem, faulty_nodes,
    faulty_edges,
)
from examples.ieee_30_bus.config import build_config


# The two edges of the small cut. (27,28) is faulted; (24,25) is the
# reconnection candidate left healthy -- see generate_ieee_30_bus.py for
# why that half and not the other.
CUT = [(24, 25), (27, 28)]
FAULTED = (27, 28)
CANDIDATE = (24, 25)
SMALL_SIDE = [25, 26, 27, 29, 30]


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
    assert len(list(topology.nodes())) == 30
    assert len(list(topology.edges())) == len(EDGES) == 41


def test_the_topology_module_carries_no_weights():
    # PROVENANCE, enforced rather than documented. Only the topology comes
    # from the IEEE test case; the dynamics and weights are ours. If the
    # parser's output ever grew weight dictionaries -- line impedances, say
    # -- builder.py's two-module import would still work while the case
    # quietly became a different thing. This fails first.
    from examples.ieee_30_bus import ieee_30_bus_topology as topology_module

    for name in ("NODE_WEIGHTS", "EDGE_WEIGHTS", "POSITIONS"):
        assert not hasattr(topology_module, name), (
            f"{name} appeared in the parsed topology module; the case's "
            "weights must come from generate_ieee_30_bus.py alone"
        )


def test_full_system_is_genuinely_unstable():
    M = build_M(NODES, EDGES, NODE_WEIGHTS, EDGE_WEIGHTS)
    lambda_min = np.linalg.eigvalsh(M).min()
    assert lambda_min <= 0


def test_faults_are_placed_where_intended():
    # Placed explicitly against this topology's structure (see
    # generate_ieee_30_bus.py's FAULTY_NODES/FAULTY_EDGES), so the exact
    # identities are part of the fixture.
    assert faulty_nodes(IEEE_30) == [6, 10]
    assert faulty_edges(IEEE_30) == [FAULTED]
    # The other half of the cut stays healthy: it is the reconnection
    # candidate, and faulting it would remove the opportunity entirely.
    assert CANDIDATE not in faulty_edges(IEEE_30)
    assert EDGE_WEIGHTS[_canon(CANDIDATE)] > 0


def test_faults_are_real():
    graph = _graph()

    # Each faulty node is unstable in EVERY island: its diagonal is
    # non-positive even at full original-graph coupling.
    for node in faulty_nodes(IEEE_30):
        diagonal = NODE_WEIGHTS[node] + sum(
            EDGE_WEIGHTS[_canon((node, j))] for j in graph.neighbors(node)
        )
        assert diagonal < 0 + 1e-9

    # The faulty edge is strongly negative, so any island holding both
    # endpoints is unstable and the edge has to be cut.
    for edge in faulty_edges(IEEE_30):
        assert EDGE_WEIGHTS[edge] == -20.0


def test_the_faulty_buses_are_the_two_highest_degree():
    # The stated reason for choosing 6 and 10 over any other pair.
    graph = _graph()
    ranked = sorted(graph.degree(), key=lambda kv: (-kv[1], kv[0]))
    assert [node for node, _ in ranked[:2]] == [6, 10]
    assert [degree for _, degree in ranked[:2]] == [7, 6]
    # And a clear gap to third, or "the two highest" would be arbitrary.
    assert ranked[2][1] == 5


def test_neither_faulty_bus_is_an_articulation_point():
    # Individually. That is ALL this establishes: removing either bus
    # alone disconnects nothing, but the two TOGETHER are a separator --
    # see test_shedding_both_faulty_buses_disconnects_the_graph, which is
    # what actually happens when F1 sheds them both.
    graph = _graph()
    articulation = sorted(nx.articulation_points(graph))
    assert articulation == [9, 12, 25, 27]
    assert 6 not in articulation
    assert 10 not in articulation

    for node in (6, 10):
        reduced = graph.copy()
        reduced.remove_node(node)
        assert nx.is_connected(reduced)


def test_the_avoided_articulation_points_each_detach_a_stub():
    # Why all four were avoided. Three of them strand a single degree-1
    # leaf, which would give an island of one node and no eligible
    # reconnection edge; the fourth strands a connected pair.
    graph = _graph()

    for node, stranded in ((9, [11]), (12, [13]), (25, [26]), (27, [29, 30])):
        reduced = graph.copy()
        reduced.remove_node(node)
        smallest = min(nx.connected_components(reduced), key=len)
        assert sorted(smallest) == stranded


def test_shedding_both_faulty_buses_disconnects_the_graph():
    # THE PREMISE THAT DOES NOT HOLD, recorded because the placement's
    # reasoning stops one step short of it. Neither bus is an articulation
    # point, but {6, 10} JOINTLY separates the graph: bus 9's neighbours
    # are exactly {6, 10, 11} and bus 11's only neighbour is 9, so removing
    # both strands {9, 11}.
    #
    # Not a defect -- the pair is stable, and F1 sheds faulty nodes anyway
    # -- but it means no partition that sheds both faulty buses can keep a
    # single island, whatever the objective prefers. Anything reading the
    # island count as a search outcome has to account for this first.
    graph = _graph()
    reduced = graph.copy()
    reduced.remove_nodes_from([6, 10])

    components = sorted(
        (sorted(c) for c in nx.connected_components(reduced)), key=len
    )
    assert [len(c) for c in components] == [2, 26]
    assert components[0] == [9, 11]
    assert sorted(graph.neighbors(9)) == [6, 10, 11]
    assert sorted(graph.neighbors(11)) == [9]

    # And the stranded pair is stable, so it is a legitimate island rather
    # than an unstable fragment.
    assert _lambda_min([9, 11], [(9, 11)]) > 0


def test_the_small_cut_separates_the_graph_into_5_and_25():
    # The placement's premise: {(24,25), (27,28)} is a genuine edge cut,
    # and both sides are large enough to be meaningful islands.
    graph = _graph()
    reduced = graph.copy()
    reduced.remove_edges_from(CUT)

    components = sorted(
        (sorted(c) for c in nx.connected_components(reduced)), key=len
    )
    assert [len(c) for c in components] == [5, 25]
    assert components[0] == SMALL_SIDE


def test_it_is_the_only_small_cut_with_two_sizeable_sides():
    # THE CLAIM IS UNDER A CRITERION, and the criterion is load-bearing.
    # This graph is not short of small cuts: it has three bridges, each a
    # one-edge cut, and 26 minimal two-edge cuts. What is unique about
    # {(24,25),(27,28)} is that it is the only cut of fewer than three
    # edges whose BOTH sides hold at least 5 buses -- every other small
    # cut shaves off a leaf or a stub. Exhaustive over all 1- and 2-edge
    # subsets, 41 + 820 = 861 of them.
    graph = _graph()

    # The qualification, pinned: dropping the size criterion makes the
    # uniqueness claim false, so it cannot be quietly dropped later.
    assert sorted(_canon(e) for e in nx.bridges(graph)) == [
        (9, 11), (12, 13), (25, 26)
    ]

    found = []
    for size in (1, 2):
        for cut in itertools.combinations(EDGES, size):
            reduced = graph.copy()
            reduced.remove_edges_from(cut)
            if nx.is_connected(reduced):
                continue
            sizes = sorted(len(c) for c in nx.connected_components(reduced))
            if min(sizes) >= 5:
                found.append(sorted(_canon(e) for e in cut))

    assert found == [sorted(CUT)]


def test_the_reconnection_candidate_survives_the_cut():
    # (24,25) is the half of the cut left healthy, so it is the edge that
    # could join the two candidate islands. Eligibility needs it to be a
    # real edge, positively weighted, and to cross the boundary.
    assert _canon(CANDIDATE) in {_canon(e) for e in EDGES}
    assert EDGE_WEIGHTS[_canon(CANDIDATE)] > 0
    assert (CANDIDATE[0] in SMALL_SIDE) != (CANDIDATE[1] in SMALL_SIDE)

    # ...and the faulted half crosses the same boundary, which is what
    # makes cutting it the thing that creates the opportunity.
    assert (FAULTED[0] in SMALL_SIDE) != (FAULTED[1] in SMALL_SIDE)
    assert EDGE_WEIGHTS[_canon(FAULTED)] < 0

    # NECESSARY, NOT SUFFICIENT. decode_reconnection also requires the
    # edge's relay path to be "middle-cut" in the spanning tree, which no
    # structural test can establish. This pins the part that is a property
    # of the instance rather than of a particular tree.


def test_only_the_small_side_is_stable_before_shedding():
    # RENAMED from "both sides are stable before shedding", which is not
    # what this asserts and not what is true: the 25-bus side contains
    # both faulty buses, so it is unstable until F1 sheds them. Only the
    # 5-bus side is stable as it stands.
    #
    # Each side is taken as an island in its own right, with its induced
    # edges.
    graph = _graph()
    reduced = graph.copy()
    reduced.remove_edges_from(CUT)

    for component in nx.connected_components(reduced):
        induced = [
            (u, v) for u, v in reduced.edges() if u in component and v in component
        ]
        if component == set(SMALL_SIDE):
            assert _lambda_min(component, induced) > 0
        else:
            # The 25-node side still CONTAINS both faulty buses, so it is
            # unstable by construction -- that is what F1 sheds them for.
            assert _lambda_min(component, induced) < 0


def test_the_cut_sides_are_stable_once_the_faulty_buses_are_shed():
    # THE RECONNECTION OPPORTUNITY, in the form the placement is meant to
    # create. Shedding 6 and 10 and cutting BOTH halves of the cut leaves
    # four stable components: the 5-bus side, the 19-bus main component,
    # and two stubs -- {9,11}, stranded by the two faulty buses jointly,
    # and {8,28}, stranded because bus 8's only neighbours are 6 and the
    # faulty edge's endpoint 28.
    #
    # What matters is WHICH TWO the candidate joins. (24,25) has bus 24 in
    # the 19-bus component and bus 25 in the 5-bus one, so reconnecting it
    # joins two substantial islands. Under the previous placement -- the
    # halves swapped -- the surviving candidate (27,28) joined the 5-bus
    # side to the two-node stub {8,28} instead, and read-only local search
    # simply kept them as one 7-bus island and never reconnected anything,
    # leaving F4 and F6 unexercised. That is what this test guards.
    graph = _graph()
    reduced = graph.copy()
    reduced.remove_edges_from(CUT)
    reduced.remove_nodes_from([6, 10])

    components = sorted(
        (sorted(c) for c in nx.connected_components(reduced)), key=len
    )
    assert [len(c) for c in components] == [2, 2, 5, 19]
    assert [c for c in components if len(c) == 2] == [[8, 28], [9, 11]]
    assert [c for c in components if len(c) == 5] == [SMALL_SIDE]

    for component in components:
        induced = [
            (u, v) for u, v in reduced.edges()
            if u in component and v in component
        ]
        assert _lambda_min(component, induced) > 0, component

    # The candidate's endpoints land in the two SUBSTANTIAL islands, not
    # in a stub. This is the assertion the previous placement failed.
    location = {node: len(c) for c in components for node in c}
    assert location[CANDIDATE[0]] == 19
    assert location[CANDIDATE[1]] == 5


def test_every_faulty_edge_destabilises_any_island_holding_both_ends():
    # test_faults_are_real only checks the faulty edge weighs -20. That is
    # necessary, not sufficient: a large enough set of positive incident
    # weights could in principle outweigh a fixed negative edge, leaving an
    # island containing both endpoints stable and the "must be cut" premise
    # false.
    #
    # The sufficient condition is the Rayleigh quotient of M_S at
    # x = e_u - e_v, for the island's induced matrix M_S:
    #
    #     x^T M_S x / ||x||^2 = (M_uu + M_vv + 2 w_uv) / 2
    #
    # If that is negative, M_S has a negative eigenvalue, so the island is
    # unstable -- whatever else it contains.
    #
    # The quotient depends on which neighbours the island keeps, so this
    # computes an UPPER BOUND over every island containing both endpoints:
    # keep every positive incident edge (each raises a diagonal) and drop
    # every other negative one (dropping it also raises a diagonal). A
    # bound below zero proves the quotient is negative for every such
    # island, which is all the test needs. The bound is conservative, not
    # the attainable maximum.
    graph = _graph()

    for u, v in faulty_edges(IEEE_30):
        w_uv = EDGE_WEIGHTS[_canon((u, v))]

        def upper_bound_diagonal(node, partner):
            total = NODE_WEIGHTS[node] + EDGE_WEIGHTS[_canon((node, partner))]
            total += sum(
                max(EDGE_WEIGHTS[_canon((node, k))], 0.0)
                for k in graph.neighbors(node) if k != partner
            )
            return total

        bound = (
            upper_bound_diagonal(u, v) + upper_bound_diagonal(v, u) + 2 * w_uv
        ) / 2.0

        assert bound < 0, (
            f"edge ({u},{v}): this bound ({bound}) does not certify that "
            "every island holding both endpoints is unstable. A nonnegative "
            "bound does not exhibit a stable island -- the bound is "
            "conservative, so it may simply have failed to certify one."
        )


def test_build_problem_draws_matching_sample_shapes():
    problem = build_problem(IEEE_30, build_topology(), build_config(), seed=0)

    assert problem.samples_1.shape == (problem.config.P_1, 30, 2)
    assert problem.samples_2.shape == (problem.config.P_2, 30, 2)
    assert problem.config.P_1 == problem.config.P_2


def test_the_generator_default_reproduces_the_committed_artefact():
    # Field by field, so a drift in the weights is caught as readily as a
    # drift in which buses were faulted. Unlike the synthetic cases there
    # is no topology to compare -- the generator does not produce one.
    from examples.ieee_30_bus import generate_ieee_30_bus as generator

    result = generator.generate(generator.DEFAULT_SEED)

    assert generator.DEFAULT_SEED == 0
    assert result["seed"] == 0
    assert result["nodes"] == NODES
    assert result["edges"] == sorted(_canon(e) for e in EDGES)
    assert result["node_weights"] == NODE_WEIGHTS
    assert result["edge_weights"] == EDGE_WEIGHTS
    assert result["faulty_nodes"] == faulty_nodes(IEEE_30)
    assert result["faulty_edges"] == faulty_edges(IEEE_30)


def test_no_argument_main_writes_the_committed_bytes(tmp_path, monkeypatch):
    # The field-by-field check above compares generate()'s RETURN VALUE. A
    # regression in write_weights() -- a changed header, a dropped field, a
    # different repr -- or in main()'s argument handling would produce a
    # different FILE while that test still passed. This drives the real
    # no-argument entry point and compares bytes.
    import sys

    from examples.ieee_30_bus import generate_ieee_30_bus as generator

    committed = (
        pathlib.Path(generator.__file__).parent / "ieee_30_bus_weights.py"
    )
    written = tmp_path / "ieee_30_bus_weights.py"
    monkeypatch.setattr(generator, "OUTPUT_PATH", written)
    monkeypatch.setattr(sys, "argv", ["generate_ieee_30_bus.py"])

    # Pinned explicitly, not just implied by the bytes below.
    assert generator.DEFAULT_SEED == 0

    generator.main()

    assert written.read_bytes() == committed.read_bytes()


def test_the_faulty_node_rule_matches_the_other_cases():
    # w_i = -(full original-graph neighbourhood coupling) - 1, the same
    # rule generate_linear_30node.py uses. Checked against the artefact
    # rather than against the generator, so a hand-edit to the weights file
    # is caught too.
    graph = _graph()
    healthy_generator_draw = {
        node: sum(EDGE_WEIGHTS[_canon((node, j))] for j in graph.neighbors(node))
        for node in faulty_nodes(IEEE_30)
    }

    for node, coupling in healthy_generator_draw.items():
        # The faulty edge is not incident to either faulty bus, so the
        # coupling sum above is the drawn one and the identity is exact.
        # approx, not ==. The rule is exact IN THE GENERATOR -- it
        # stores -sum - 1.0 -- but this test re-derives the sum from a
        # networkx neighbour iteration, which need not add the terms in
        # the generator's order, and float addition is not associative.
        # The same effect is why the fully-coupled diagonal comes out at
        # -0.9999999999999964 on one bus rather than exactly -1.0.
        assert NODE_WEIGHTS[node] == pytest.approx(-coupling - 1.0, abs=1e-12)
        assert not any(
            node in edge for edge in faulty_edges(IEEE_30)
        ), "a faulty bus became incident to a faulty branch; revisit this test"
