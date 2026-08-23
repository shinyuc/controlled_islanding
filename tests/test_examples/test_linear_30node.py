"""Structural tests for the 30-node linear consensus example, mirroring
tests/test_examples/test_linear_50node.py one for one. With random
weights, specific islands can't be hand-verified -- these check
structural invariants, not outcomes. None of these call solve().
"""

import pathlib

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


def test_the_generator_default_reproduces_the_committed_artefact():
    # Same defect as the 40-node case had: DEFAULT_SEED was 0 while this
    # artefact came from seed 26, and generate(0) RAISES, because the edges
    # FAULTY_EDGES names do not exist in that draw -- so the documented
    # no-argument command did not work. Field by field, so a drift in the
    # weights is caught as readily as one in the topology.
    from examples.linear_30node import generate_linear_30node as generator

    result = generator.generate(generator.DEFAULT_SEED)

    assert generator.DEFAULT_SEED == 26
    assert result["seed"] == 26
    assert result["nodes"] == NODES
    assert result["edges"] == EDGES
    assert result["node_weights"] == NODE_WEIGHTS
    assert result["edge_weights"] == EDGE_WEIGHTS
    assert result["positions"] == POSITIONS
    assert result["faulty_nodes"] == faulty_nodes(LINEAR_30)
    assert result["faulty_edges"] == faulty_edges(LINEAR_30)


def test_no_argument_main_writes_the_committed_bytes(tmp_path, monkeypatch):
    # The field-by-field check above compares generate()'s RETURN VALUE. A
    # regression in write_topology() -- a changed header, a dropped field,
    # a different repr -- or in main()'s argument handling would produce a
    # different FILE while that test still passed. This drives the real
    # no-argument entry point and compares bytes, which is what makes
    # "running it with no argument reproduces the artefact" enforced
    # rather than asserted.
    import sys

    from examples.linear_30node import generate_linear_30node as generator

    committed = (
        pathlib.Path(generator.__file__).parent / "linear_30node_topology.py"
    )
    written = tmp_path / "linear_30node_topology.py"
    monkeypatch.setattr(generator, "OUTPUT_PATH", written)
    monkeypatch.setattr(sys, "argv", ["generate_linear_30node.py"])

    # Pinned explicitly, not just implied by the bytes below: the byte
    # comparison alone cannot catch a seed regression whose retry loop
    # happens to land on the right draw anyway. See the 50-node case,
    # where exactly that is possible.
    assert generator.DEFAULT_SEED == 26

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
    # realisable by any single island. Brute-forcing the neighbour choices
    # gives the true maxima, which sit below the bound wherever a shared
    # neighbour carries edges the bound would treat inconsistently:
    #
    #     edge       bound      true max
    #     (10,25)  -13.1911     -18.0681
    #     (11,25)  -17.5713     -22.4483
    #     (12,24)  -18.5815     -23.5627
    #     (18,24)  -17.4282     -22.4094
    #
    # Both are negative, so the conclusion is unaffected -- the bound is
    # simply the cheap sufficient check.
    #
    # Taking the full-neighbourhood value instead would be unsound here:
    # nodes carrying TWO faulty edges exist in this fixture (24 and 25),
    # and excluding one of them raises the quotient by 10.
    graph = nx.Graph()
    graph.add_nodes_from(NODES)
    graph.add_edges_from(EDGES)

    for u, v in faulty_edges(LINEAR_30):
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
