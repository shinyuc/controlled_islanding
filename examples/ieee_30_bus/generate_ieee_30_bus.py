"""Seeded WEIGHT generator for the IEEE 30-bus case.

Writes examples/ieee_30_bus/ieee_30_bus_weights.py. Mirrors
examples/linear_30node/generate_linear_30node.py's RULES exactly --
w_i ~ Uniform[1,5], w_ij ~ Uniform[1,10], one seed for everything, node
faults set to -(full-neighbourhood coupling) - 1, faulty edges set to
-20.

The same rules do NOT make the two 30-node cases differ in topology
alone, and it would be wrong to read them that way. The generator seeds
differ (0 here, 26 there), so every realised weight differs; the faulty
buses differ (6 and 10 against 5 and 16); and the fault counts differ
(one faulty edge against four). What is shared is the FORM of the
distributions and of the fault transformation, plus the AlgorithmConfig
and the sampling seed, which is enough to make the two cases comparable
but not enough to make either a controlled variant of the other.

WHY THIS GENERATOR DRAWS WEIGHTS ONLY
-------------------------------------
Every other generator in examples/ draws positions, derives edges from a
radius, retries the seed until the graph is connected, and only then draws
weights, writing all five arrays to one artefact. Here the topology is
given: it is the IEEE 30-bus test case, parsed from the IEEE Common Data
Format by parse_ieee_30_bus.py. Three consequences, each deliberate:

  * NO GEOMETRY. The IEEE case is a bus/branch list with no coordinates,
    so there are no POSITIONS to draw and none in the artefact. Anything
    that reads POSITIONS -- the geometric edge-length test the synthetic
    cases carry, for instance -- has no counterpart here.

  * NO CONNECTIVITY RETRY. There is one topology and it is connected, so
    there is no draw to reject and no seed to advance past. The generator
    ASSERTS connectivity rather than assuming it: the check is cheap, and
    if a re-parse ever produced something disconnected, failing here is
    better than every downstream case quietly solving a different problem.

  * SEPARATE OUTPUT FILE. The weights go to ieee_30_bus_weights.py, NOT
    into ieee_30_bus_topology.py. The topology file is the parser's
    output and says so; having two generators write one file would put
    our seeded draw and derived IEEE data under one provenance claim, and
    re-running either would silently discard the other's work. It also
    keeps the repository's existing copyright separation intact --
    .gitignore excludes the CDF source and the parser, and commits only
    the derived topology. This file and its output are ours and are
    committed, exactly like the synthetic generators.

So builder.py imports NODES/EDGES from the topology module and
NODE_WEIGHTS/EDGE_WEIGHTS from this one. That split is the honest
description of where each part came from.

Usage:
    python examples/ieee_30_bus/generate_ieee_30_bus.py [seed]

Default: seed=0 -- the seed the committed artefact was generated with.
Unlike the synthetic cases, ANY seed produces a valid artefact here: the
faults name buses and branches of a fixed topology, so there is no draw
that can fail to contain them and no reason to prefer one seed's shape
over another's. Running with no argument reproduces the artefact exactly,
which tests/test_examples/test_ieee_30_bus.py checks field by field and
byte for byte.
"""

import sys
from pathlib import Path

import networkx as nx
import numpy as np

from examples.ieee_30_bus.ieee_30_bus_topology import NODES, EDGES

HERE = Path(__file__).parent
OUTPUT_PATH = HERE / "ieee_30_bus_weights.py"

# The seed the COMMITTED artefact was generated with, so the documented
# no-argument command reproduces it. 0 is usable here where it was not for
# the 30- and 40-node synthetic cases: those rejected it because its draw
# did not contain the edges FAULTY_EDGES named, and this topology is fixed.
DEFAULT_SEED = 0

# Explicit fault placement, chosen against this topology's structure. Every
# figure below is checked by tests/test_examples/test_ieee_30_bus.py rather
# than asserted here.
#
# FAULTY_NODES: buses 6 and 10, degrees 7 and 6 -- the two highest in the
# graph (next is 12 at 5). Shedding the two best-connected buses exercises
# F1 where it costs the most coupling.
#
# NEITHER IS AN ARTICULATION POINT, BUT THE PAIR IS A SEPARATOR. Removing
# either one alone leaves the graph connected. Removing BOTH strands
# {9, 11}: bus 9's neighbours are exactly {6, 10, 11} and bus 11's only
# neighbour is 9. So any partition that sheds both faulty buses has at
# least two stable islands before the objective expresses any preference,
# and the island count here is not a pure search outcome. The stranded
# pair is stable, so this costs nothing but has to be read correctly.
#
# The articulation points are 9, 12, 25 and 27, and all four were avoided.
# Removing 9, 12 or 25 detaches a single degree-1 leaf -- 11, 13 and 26
# respectively -- which would strand an island of one node with no
# eligible reconnection edge: a trivial island, not a useful test.
# Removing 27 detaches the connected pair {29, 30}, less degenerate but
# still a two-node stub.
#
# FAULTY_EDGES: (27, 28), one half of the two-edge cut
# {(24,25), (27,28)}, which separates {25,26,27,29,30} from the other 25
# buses. Faulting (27,28) leaves (24,25) as the reconnection candidate.
#
# WHICH HALF TO FAULT IS THE WHOLE DECISION, and the other choice does not
# work. Faulting (24,25) instead leaves (27,28) as the candidate -- but
# bus 28's only other neighbour is the faulty bus 6, so once 6 is shed,
# (27,28) joins the 5-bus side to the two-node stub {8,28} rather than to
# anything substantial. Read-only local search on that placement returned
# three stable islands of 19, 7 and 2 buses with NO reconnection at all:
# the candidate ended up INTERNAL to a 7-bus island spanning the 5-bus
# side and the stub, so F4 and F6 were never exercised. That is the
# failure this placement exists to avoid.
#
# Faulting (27,28) puts the stub on the far side of a forced cut instead.
# After shedding 6 and 10 and cutting the faulty edge, the graph is
# {8,28}, {9,11} and one 24-bus component; cutting the candidate (24,25)
# then splits that component into a 5-bus island (lambda_min 3.483) and a
# 19-bus island (lambda_min 1.828), and (24,25) joins those two. That is
# a reconnection between two substantial islands, which is what F4 and F6
# need to be exercised at all.
#
# The forcing is the same strength as the synthetic 30-node case's, not
# weaker. There, five edges crossed the target boundary and four were
# faulted; here two cross and one is faulted. In BOTH cases exactly one
# healthy crossing survives the forced cuts, so the marginal decision --
# cut the last crossing and collect the reconnection reward, or keep it
# and stay one island -- has the same shape. What the placement
# establishes is that the opportunity EXISTS; whether either solver takes
# it is an outcome of the search, not a property of the placement.
#
# "THE ONLY SMALL CUT" IS A CLAIM UNDER A CRITERION, not an unqualified
# one. This graph has three bridges -- (9,11), (12,13), (25,26) -- each
# of which is a one-edge cut, and 26 minimal two-edge cuts. What is
# unique about {(24,25), (27,28)} is that it is the only cut of fewer
# than three edges whose BOTH sides hold at least five buses. Every other
# small cut shaves off a leaf or a stub.
FAULTY_NODES = [6, 10]
FAULTY_EDGES = [(27, 28)]

# Faulty edges get a strongly negative coupling, so any island containing
# both endpoints is unstable and the edge must be cut.
FAULTY_EDGE_WEIGHT = -20.0


def _canon(edge):
    i, j = edge
    return (i, j) if i < j else (j, i)


def generate(seed):
    nodes = list(NODES)
    edges = sorted(_canon(edge) for edge in EDGES)

    graph = nx.Graph()
    graph.add_nodes_from(nodes)
    graph.add_edges_from(edges)
    # ASSERTED, not assumed -- see the module docstring. There is no retry
    # loop to fall back on, so a disconnected topology has to stop here.
    if not nx.is_connected(graph):
        raise ValueError(
            "the IEEE 30-bus topology is not connected; every builder and "
            "test downstream assumes it is"
        )

    # One stream, nodes before edges, exactly as the synthetic generators
    # order it -- so the two 30-node cases' weights are drawn by the same
    # procedure and differ only in how many draws each takes.
    rng = np.random.default_rng(seed)
    node_weights = {node: float(rng.uniform(1.0, 5.0)) for node in nodes}
    edge_weights = {edge: float(rng.uniform(1.0, 10.0)) for edge in edges}

    edge_set = set(edges)
    unknown = [node for node in FAULTY_NODES if node not in nodes]
    if unknown:
        raise ValueError(f"FAULTY_NODES references {unknown}, not in the graph")
    faulty_edges = [_canon(edge) for edge in FAULTY_EDGES]
    missing = [edge for edge in faulty_edges if edge not in edge_set]
    if missing:
        raise ValueError(
            f"FAULTY_EDGES references {missing}, which are not branches of "
            "the IEEE 30-bus case -- the placement is tied to this specific "
            "topology, so it must be revisited if the parser's output changes"
        )

    neighbors = {node: set() for node in nodes}
    for u, v in edges:
        neighbors[u].add(v)
        neighbors[v].add(u)

    # Node faults first, off the drawn (healthy) couplings, so w_i is
    # -(full original-graph neighbourhood coupling) - 1 exactly as in the
    # other cases. Node 6 IS incident to no faulty edge here and node 10
    # likewise -- (24,25) touches neither -- so the order does not actually
    # matter, but fixing it keeps the rule unambiguous if that changes.
    for node in FAULTY_NODES:
        full_coupling = sum(
            edge_weights[_canon((node, j))] for j in neighbors[node]
        )
        node_weights[node] = -full_coupling - 1.0

    for edge in faulty_edges:
        edge_weights[edge] = FAULTY_EDGE_WEIGHT

    return {
        "seed": seed,
        "nodes": nodes,
        "edges": edges,
        "node_weights": node_weights,
        "edge_weights": edge_weights,
        "faulty_nodes": list(FAULTY_NODES),
        "faulty_edges": faulty_edges,
    }


def write_weights(result, output_path):
    seed = result["seed"]
    content = (
        "# Weights for the IEEE 30-bus case, randomly generated (seeded).\n"
        f"# Generated by generate_ieee_30_bus.py, seed={seed}.\n"
        f"# Faulty nodes: {result['faulty_nodes']}. "
        f"Faulty edges: {result['faulty_edges']}.\n"
        '"""Linear-consensus weights for the IEEE 30-bus topology.\n'
        "\n"
        "GENERATED, SEEDED DATA -- do not hand-edit; re-run\n"
        "generate_ieee_30_bus.py instead.\n"
        "\n"
        "Only the TOPOLOGY of this case comes from the IEEE 30-bus test\n"
        "case (see ieee_30_bus_topology.py). The dynamics are linear\n"
        "consensus and these weights are drawn from one seed by exactly\n"
        "the rules the synthetic cases use -- w_i ~ U[1,5],\n"
        "w_ij ~ U[1,10] -- so nothing here is IEEE data. Faults are then\n"
        "applied on top: the faulty nodes' w_i is set to minus their full\n"
        "original-graph neighbourhood coupling minus one, and the faulty\n"
        "branches are set to -20.\n"
        "\n"
        "There are no POSITIONS: the IEEE case is a bus/branch list with\n"
        "no coordinates.\n"
        '"""\n'
        "\n"
        f"NODE_WEIGHTS = {result['node_weights']!r}\n"
        "\n"
        f"EDGE_WEIGHTS = {result['edge_weights']!r}\n"
    )
    Path(output_path).write_text(content)


def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_SEED

    result = generate(seed)
    write_weights(result, OUTPUT_PATH)

    graph = nx.Graph()
    graph.add_nodes_from(result["nodes"])
    graph.add_edges_from(result["edges"])

    n_nodes = len(result["nodes"])
    n_edges = len(result["edges"])

    print(f"Wrote {OUTPUT_PATH}")
    print(f"Seed: {result['seed']}")
    print(f"Nodes: {n_nodes}")
    print(f"Edges: {n_edges}")
    print(f"Connected: {nx.is_connected(graph)}")
    print(f"Average degree: {2 * n_edges / n_nodes:.2f}")
    print(f"Faulty nodes: {result['faulty_nodes']}")
    print(f"Faulty edges: {result['faulty_edges']}")


if __name__ == "__main__":
    main()
