"""Structural tests for the 40-node linear consensus example.

Follows tests/test_examples/test_linear_30node.py, plus tests specific to
this case: the fault placement here rests on structural facts about one
particular draw (its articulation points, its cluster boundary, and the
stability of the two sides that boundary creates), so those facts are
pinned rather than left in a comment.

With random weights, specific islands cannot be hand-verified -- these
check structural invariants, not outcomes. None of them call solve().
"""

import pathlib
from collections import Counter

import networkx as nx
import numpy as np
import pytest

from examples.linear_40node.linear_40node_topology import (
    NODES, EDGES, NODE_WEIGHTS, EDGE_WEIGHTS, POSITIONS,
)
from examples.linear_40node.builder import (
    LINEAR_40, build_topology, build_M, build_problem, faulty_nodes,
    faulty_edges,
)
from examples.linear_40node.config import build_config


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
    assert len(list(topology.nodes())) == 40
    # The LITERAL, not len(EDGES) -- comparing the artefact against itself
    # passes for any draw and would not notice a regenerated topology.
    assert len(list(topology.edges())) == 109


def _distance(u, v):
    xu, yu = POSITIONS[u]
    xv, yv = POSITIONS[v]
    return ((xu - xv) ** 2 + (yu - yv) ** 2) ** 0.5


def test_geometry_edge_set_is_exactly_the_radius_2_pairs():
    # BOTH directions. Checking only that listed edges are within radius 2
    # would pass on any subset of the correct edge set -- including one
    # missing edges, which changes the topology without breaking anything
    # a one-directional test can see.
    listed = {_canon(edge) for edge in EDGES}
    expected = {
        _canon((u, v))
        for i, u in enumerate(NODES) for v in NODES[i + 1:]
        if _distance(u, v) <= 2.0
    }

    assert listed == expected

    # The margin on each side, so a future regeneration that lands a pair
    # near the boundary is visible rather than silently reclassified.
    included = max(_distance(u, v) for u, v in listed)
    excluded = min(
        _distance(u, v)
        for i, u in enumerate(NODES) for v in NODES[i + 1:]
        if _canon((u, v)) not in listed
    )
    assert included == pytest.approx(1.98486, abs=1e-5)
    assert excluded == pytest.approx(2.01788, abs=1e-5)
    assert included < 2.0 < excluded


def test_full_system_is_genuinely_unstable():
    # The VALUE, not just the sign. lambda_min <= 0 would pass on a system
    # that is barely unstable, or on a differently-faulted one -- neither
    # of which is this fixture.
    M = build_M(NODES, EDGES, NODE_WEIGHTS, EDGE_WEIGHTS)
    eigenvalues = np.linalg.eigvalsh(M)

    assert eigenvalues.min() == pytest.approx(-28.075, abs=1e-3)
    assert (eigenvalues <= 0).sum() == 3


def test_faults_are_placed_where_intended():
    # Unlike the 20-node case, faults here are placed explicitly
    # (generate_linear_40node.py's FAULTY_NODES/FAULTY_EDGES), so the exact
    # identities are part of the fixture, not an artifact of "first n nodes".
    assert faulty_nodes(LINEAR_40) == [8, 24]
    assert faulty_edges(LINEAR_40) == [(10, 16), (17, 38)]


def test_faults_are_real():
    neighbors = {node: set() for node in NODES}
    for u, v in EDGES:
        neighbors[u].add(v)
        neighbors[v].add(u)

    # Each faulty node is unstable in EVERY island: its diagonal is
    # negative even at full original-graph coupling. Pinned at exactly
    # -1.0, which is what the rule w_i = -(sum of w_ij) - 1 produces --
    # "any negative value" would also pass if the rule were applied over
    # the wrong neighbourhood.
    for node in faulty_nodes(LINEAR_40):
        diagonal = NODE_WEIGHTS[node] + sum(
            EDGE_WEIGHTS[_canon((node, j))] for j in neighbors[node]
        )
        assert diagonal == pytest.approx(-1.0, abs=1e-9)

    # Each faulty edge is strongly negative, so any island holding both
    # endpoints is unstable and the edge has to be cut.
    for edge in faulty_edges(LINEAR_40):
        assert EDGE_WEIGHTS[edge] == -20.0


def test_reconnection_candidate_survives_the_cut():
    # The placement's whole purpose: the two faulty edges are two of the
    # three crossing edges between the cluster
    # {4,7,16,19,27,32,33,38,39,40} and the rest, leaving (26,38) as the
    # only healthy link across. That is what can give the search two stable
    # islands with an eligible cut edge between them -- the precondition
    # for a non-empty E^rec.
    cluster = {4, 7, 16, 19, 27, 32, 33, 38, 39, 40}
    crossing = sorted(
        _canon((u, v)) for u, v in EDGES if (u in cluster) != (v in cluster)
    )
    assert crossing == [(10, 16), (17, 38), (26, 38)]

    healthy = [edge for edge in crossing if EDGE_WEIGHTS[edge] > 0]
    assert healthy == [(26, 38)]


def test_both_sides_of_the_cut_stay_connected():
    # A cut that fragments either side gives islands the reconnection
    # candidate cannot usefully join. Checked here because it is a property
    # of THIS draw: it was the reason the 5-node group behind articulation
    # point 25 was rejected as a cluster candidate.
    cluster = {4, 7, 16, 19, 27, 32, 33, 38, 39, 40}
    graph = nx.Graph()
    graph.add_nodes_from(NODES)
    graph.add_edges_from(EDGES)

    assert nx.is_connected(graph.subgraph(cluster))
    assert nx.is_connected(graph.subgraph(set(NODES) - cluster))


def test_faulty_nodes_avoid_articulation_points():
    # Shedding an articulation point strands whatever sits behind it. What
    # each one detaches on this draw:
    #
    #     2   ->  {11, 23}                  (connected, via (11,23))
    #     10  ->  {18, 36}                  (connected, via (18,36))
    #     23  ->  {11}
    #     25  ->  {3}  and  {2, 11, 23, 28} (the 4-group is connected)
    #     36  ->  {18}
    #
    # So they are NOT all leaf-stranding, and the group behind 25 is not
    # internally disconnected -- an earlier version of this comment claimed
    # both. The reason to avoid them stands regardless: a fault at an
    # articulation point splits the graph on its own terms rather than at
    # the targeted bottleneck, and the detached pieces are 1-4 nodes cut
    # off by node removal, so they offer no reconnection candidate. See the
    # generator's reasoning.
    graph = nx.Graph()
    graph.add_nodes_from(NODES)
    graph.add_edges_from(EDGES)

    articulation = set(nx.articulation_points(graph))
    assert articulation == {2, 10, 23, 25, 36}
    assert not articulation & set(faulty_nodes(LINEAR_40))


def test_build_problem_draws_matching_sample_shapes():
    problem = build_problem(LINEAR_40, build_topology(), build_config(), seed=0)

    assert problem.samples_1.shape == (problem.config.P_1, 40, 2)
    assert problem.samples_2.shape == (problem.config.P_2, 40, 2)
    assert problem.config.P_1 == problem.config.P_2


# ---------------------------------------------------------------------------
# The properties the fault placement rests on
#
# The generator's reasoning claims the cut creates a FEASIBLE reconnection
# opportunity: two genuinely stable candidate islands joined by one healthy
# cut edge. Feasible, not forced -- nothing here says a solver will choose
# that partition. These pin the structural half, which is the half that can
# be checked without a solve.
# ---------------------------------------------------------------------------

CLUSTER = frozenset({4, 7, 16, 19, 27, 32, 33, 38, 39, 40})


def _graph():
    graph = nx.Graph()
    graph.add_nodes_from(NODES)
    graph.add_edges_from(EDGES)
    return graph


def _lambda_min(nodes):
    nodes = sorted(nodes)
    member = set(nodes)
    induced = [
        _canon((u, v)) for u, v in EDGES if u in member and v in member
    ]
    return float(np.linalg.eigvalsh(
        build_M(nodes, induced, NODE_WEIGHTS, EDGE_WEIGHTS)
    ).min())


def test_cutting_all_three_crossings_separates_the_cluster_exactly():
    # Two components, 10 and 30, with the cluster as one of them. If the
    # cut left either side in pieces, the reconnection candidate would
    # join fragments rather than islands.
    graph = _graph()
    graph.remove_edges_from([(10, 16), (17, 38), (26, 38)])
    components = [set(c) for c in nx.connected_components(graph)]

    assert sorted(len(c) for c in components) == [10, 30]
    assert set(CLUSTER) in components


def test_the_cluster_is_a_stable_candidate_island():
    assert nx.is_connected(_graph().subgraph(CLUSTER))
    assert _lambda_min(CLUSTER) == pytest.approx(2.982, abs=1e-3)


def test_the_complement_stays_stable_after_shedding_the_faulty_nodes():
    # Shedding 8 and 24 must not disconnect the other side, or what is
    # left is several islands rather than the one the placement intends.
    complement = set(NODES) - set(CLUSTER) - set(faulty_nodes(LINEAR_40))

    assert len(complement) == 28
    assert nx.is_connected(_graph().subgraph(complement))
    assert _lambda_min(complement) == pytest.approx(2.509, abs=1e-3)


def test_the_shed_group_is_connected_and_unstable():
    # {8, 24} as a single unstable island is only coherent if the two are
    # adjacent; otherwise they are two separate unstable islands.
    shed = set(faulty_nodes(LINEAR_40))

    assert nx.is_connected(_graph().subgraph(shed))
    assert _lambda_min(shed) < 0


def test_the_reconnection_candidate_joins_the_two_stable_sides():
    # (26,38) is only an eligible reconnection if its endpoints land in
    # the two DISTINCT stable candidate islands -- E^rec requires exactly
    # that. An endpoint in the shed group would disqualify it.
    complement = set(NODES) - set(CLUSTER) - set(faulty_nodes(LINEAR_40))

    assert 38 in CLUSTER
    assert 26 in complement
    assert EDGE_WEIGHTS[_canon((26, 38))] > 0


def test_faulting_the_two_edges_does_not_by_itself_force_the_split():
    # Stated because the generator's comment is careful about it: with
    # only the two faulted crossings removed the graph is STILL connected,
    # through the healthy (26,38). The placement creates an opportunity;
    # it does not compel the partition.
    graph = _graph()
    graph.remove_edges_from(faulty_edges(LINEAR_40))

    assert nx.is_connected(graph)


# ---------------------------------------------------------------------------
# Artefact/generator agreement
# ---------------------------------------------------------------------------

def test_the_generator_default_reproduces_the_committed_artefact():
    # The documented no-argument command must rebuild exactly what is
    # committed. It did not: DEFAULT_SEED was 0 while the artefact came
    # from seed 36, and generate(0) RAISES, because the edges FAULTY_EDGES
    # names do not exist in that draw. Field by field, so a drift in the
    # weights is caught as readily as one in the topology.
    from examples.linear_40node import generate_linear_40node as generator

    result = generator.generate(generator.DEFAULT_SEED)

    assert generator.DEFAULT_SEED == 36
    assert result["seed"] == 36
    assert result["nodes"] == NODES
    assert result["edges"] == EDGES
    assert result["node_weights"] == NODE_WEIGHTS
    assert result["edge_weights"] == EDGE_WEIGHTS
    assert result["positions"] == POSITIONS
    assert result["faulty_nodes"] == faulty_nodes(LINEAR_40)
    assert result["faulty_edges"] == faulty_edges(LINEAR_40)


def test_no_argument_main_writes_the_committed_bytes(tmp_path, monkeypatch):
    # The field-by-field check above compares generate()'s RETURN VALUE. A
    # regression in write_topology() -- a changed header, a dropped field,
    # a different repr -- or in main()'s argument handling would produce a
    # different FILE while that test still passed. This drives the real
    # no-argument entry point and compares bytes, which is what makes
    # "running it with no argument reproduces the artefact" enforced
    # rather than asserted.
    import sys

    from examples.linear_40node import generate_linear_40node as generator

    committed = (
        pathlib.Path(generator.__file__).parent / "linear_40node_topology.py"
    )
    written = tmp_path / "linear_40node_topology.py"
    monkeypatch.setattr(generator, "OUTPUT_PATH", written)
    monkeypatch.setattr(sys, "argv", ["generate_linear_40node.py"])

    # Pinned explicitly, not just implied by the bytes below: the byte
    # comparison alone cannot catch a seed regression whose retry loop
    # happens to land on the right draw anyway -- a generator whose
    # DEFAULT_SEED changed but whose loop advanced to the same
    # successful seed would write identical bytes.
    assert generator.DEFAULT_SEED == 36

    generator.main()

    assert written.read_bytes() == committed.read_bytes()


def test_every_faulty_edge_destabilises_any_island_holding_both_ends():
    # test_faults_are_real only checks each faulty edge weighs -20. That is
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
    # island, which is all the test needs.
    #
    # The bound is CONSERVATIVE, not the attainable maximum. It maximises
    # the two diagonals independently, but a neighbour shared by u and v
    # contributes to both at once, so some combinations it assumes are not
    # realisable by any single island. On the 30-node fixture that gap is
    # visible -- its bounds run -13.19 to -18.58 while the true maxima run
    # -18.07 to -23.56.
    #
    # On THIS fixture the bound happens to be attained, and also coincides
    # with the full-neighbourhood value, because no node here carries two
    # faulty edges. That is a property of this placement, not of the
    # construction: the helper is the same conservative bound either way,
    # and is asserted as a bound, not as an exact worst case.
    graph = nx.Graph()
    graph.add_nodes_from(NODES)
    graph.add_edges_from(EDGES)

    for u, v in faulty_edges(LINEAR_40):
        w_uv = EDGE_WEIGHTS[_canon((u, v))]

        def upper_bound_diagonal(node, partner):
            # The faulty edge is present by construction; every other
            # incident edge is included only if it raises the diagonal.
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


def test_faulty_edge_rayleigh_quotients_at_full_coupling():
    # The concrete values, at the FULL neighbourhood. Pinned separately
    # from the bound above because on this fixture no node carries two
    # faulty edges, so the two coincide -- and if a future placement broke
    # that, the divergence should be visible rather than silent. The
    # assertion below that no node carries two faulty edges is what makes
    # that coincidence checked rather than assumed.
    graph = nx.Graph()
    graph.add_nodes_from(NODES)
    graph.add_edges_from(EDGES)

    def full_quotient(u, v):
        def diagonal(node):
            return NODE_WEIGHTS[node] + sum(
                EDGE_WEIGHTS[_canon((node, k))] for k in graph.neighbors(node)
            )
        return (diagonal(u) + diagonal(v) + 2 * EDGE_WEIGHTS[_canon((u, v))]) / 2.0

    assert full_quotient(10, 16) == pytest.approx(-24.0953, abs=1e-4)
    assert full_quotient(17, 38) == pytest.approx(-9.3288, abs=1e-4)

    # The premise those two values rest on: with no node carrying two
    # faulty edges, dropping a neighbour can never REMOVE a negative
    # incident weight, so the full-neighbourhood value is also the bound
    # and the true maximum. The 30-node fixture violates this (nodes 24
    # and 25 each carry two), which is why it needs the bound instead.
    incident = Counter()
    for u, v in faulty_edges(LINEAR_40):
        incident[u] += 1
        incident[v] += 1
    assert max(incident.values()) == 1
