"""Seeded WEIGHT generator for the IEEE 118-bus case.

Writes examples/ieee_118_bus/ieee_118_bus_weights.py. Mirrors
examples/linear_30node/generate_linear_30node.py's RULES exactly --
w_i ~ Uniform[1,5], w_ij ~ Uniform[1,10], one seed for everything, node
faults set to -(full-neighbourhood coupling) - 1, faulty edges set to
-20.

Sharing those rules does NOT make this case a controlled variant of any
other, and it would be wrong to read it that way. Every instance-level
quantity differs from every other case: the topology, the realised
weights (each case draws from its own seed), the fault locations, and the
fault counts -- this case has four faulty buses and NO faulty edges,
where the others have two and one, or two and four. What is shared is the
FORM of the distributions and of the fault transformation, plus the
AlgorithmConfig and the sampling seed. That is enough to place all the
cases on one scaling curve; it is not enough to attribute a difference
between any two of them to a single cause.

WHY THIS GENERATOR DRAWS WEIGHTS ONLY
-------------------------------------
Every other generator in examples/ draws positions, derives edges from a
radius, retries the seed until the graph is connected, and only then draws
weights, writing all five arrays to one artefact. Here the topology is
given: it is the IEEE 118-bus test case, parsed from the IEEE Common Data
Format by parse_ieee_118_bus.py. Three consequences, each deliberate:

  * NO GEOMETRY. The IEEE case is a bus/branch list with no coordinates,
    so there are no POSITIONS to draw and none in the artefact. Anything
    that reads POSITIONS -- the geometric edge-length test the synthetic
    cases carry, for instance -- has no counterpart here.

  * NO CONNECTIVITY RETRY. There is one topology and it is connected, so
    there is no draw to reject and no seed to advance past. The generator
    ASSERTS connectivity rather than assuming it: the check is cheap, and
    if a re-parse ever produced something disconnected, failing here is
    better than every downstream case quietly solving a different problem.

  * SEPARATE OUTPUT FILE. The weights go to ieee_118_bus_weights.py, NOT
    into ieee_118_bus_topology.py. The topology file is the parser's
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
    python examples/ieee_118_bus/generate_ieee_118_bus.py [seed]

Default: seed=0 -- the seed the committed artefact was generated with.
Unlike the synthetic cases, ANY seed produces a valid artefact here: the
faults name buses of a fixed topology, so there is no draw that can fail
to contain them and no reason to prefer one seed's shape
over another's. Running with no argument reproduces the artefact exactly,
which tests/test_examples/test_ieee_118_bus.py checks field by field and
byte for byte.
"""

import sys
from pathlib import Path

import networkx as nx
import numpy as np

from examples.ieee_118_bus.ieee_118_bus_topology import NODES, EDGES

HERE = Path(__file__).parent
OUTPUT_PATH = HERE / "ieee_118_bus_weights.py"

# The seed the COMMITTED artefact was generated with, so the documented
# no-argument command reproduces it. 0 is usable here where it was not for
# the 30- and 40-node synthetic cases: those rejected it because its draw
# did not contain the edges FAULTY_EDGES named, and this topology is fixed.
DEFAULT_SEED = 0

# Explicit fault placement, chosen against this topology's structure. Every
# figure below is checked by tests/test_examples/test_ieee_118_bus.py rather
# than asserted here.
#
# FAULTY_NODES: buses 17, 37, 59 and 80, of degrees 6, 6, 6 and 7. Well
# connected -- the graph's average degree is 3.03 -- so shedding them
# removes 25 edges and exercises F1 where it costs real coupling. They are
# NOT the four highest-degree buses: 49 (degree 9), 100 (8) and 12 (7) are
# higher, and 100 and 12 are articulation points. The criterion was a
# high-degree spread that stays connected when shed, not the degree
# ranking.
#
# The faulted rule makes each bus's fully-coupled diagonal -1 BY
# CONSTRUCTION: w_i is set to minus the neighbourhood coupling minus one,
# so w_i + sum_j w_ij is -1 in exact arithmetic. In floating point the two
# sums are formed in different orders and the result lands within an ulp
# or two -- bus 17 comes out at -0.9999999999999964. Every test here
# compares with a tolerance for that reason; nothing depends on the
# equality being exact, only on the diagonal being safely negative.
#
# NONE IS AN ARTICULATION POINT, AND THE FOUR TOGETHER ARE NOT A
# SEPARATOR. Both halves matter, and the second is the one that is easy to
# get wrong. The nine articulation points are 8, 9, 12, 68, 71, 85, 86,
# 100 and 110; none of the four is among them, so removing any one alone
# leaves the graph connected. Removing ALL FOUR also leaves a single
# connected component, of 114 buses.
#
# That joint check is not implied by the individual one, and this
# repository has already been caught by the gap. In the IEEE 30-bus case
# buses 6 and 10 are individually not articulation points, yet removing
# both strands {9, 11} -- because bus 9's only neighbours are 6, 10 and
# 11. A set of non-cut vertices can still be a cut set. So the four here
# are verified jointly.
#
# FOUR IS WHERE IT STOPS BEING CLEAN. Adding a fifth high-degree bus, 77,
# fragments the remainder into 80, 31 and 2 buses. That is why the
# placement is four and not five: a fragmented remainder would make the
# island count a fact about the fault set rather than a search outcome,
# which is exactly the confound the 30-bus case documents.
#
# FAULTY_EDGES: NONE, deliberately. This instance exists to push the
# SCALING comparison between the local search and the MILP at a size no
# other case reaches, and adding a faulty edge to manufacture a
# reconnection would change what is being measured without making the
# reconnection any more representative.
#
# WHAT THIS DOES AND DOES NOT IMPLY. An empty faulty-edge list does NOT
# force E^rec to be empty, and it would be wrong to predict F4 = F6 = 0
# from it. E^rec holds original edges whose endpoints land in DISTINCT
# stable islands, and a perfectly healthy edge qualifies if the chosen
# partition happens to cut it. Nothing here forbids that; the search is
# free to split the 114-bus remainder and reconnect across the split if
# the objective rewards it.
#
# The accurate statement is narrower: this instance is not DESIGNED to
# induce or demonstrate reconnection. Nothing about the placement steers
# the search toward a split, so reconnection is neither guaranteed nor
# ruled out, and whichever way it comes out carries no information about
# the mechanism. Reconnection is demonstrated on the small cases, where a
# bottleneck can be targeted and the outcome inspected by hand.
FAULTY_NODES = [17, 37, 59, 80]
FAULTY_EDGES = []

# Faulty edges get a strongly negative coupling, so any island containing
# both endpoints is unstable and the edge must be cut.
#
# UNUSED ON THIS CASE: FAULTY_EDGES is empty, so the loop that applies
# this never runs and no branch weight departs from its drawn value in
# [1, 10]. It is kept, at the same value the other cases use, so that
# adding a faulty branch here is a one-line change that behaves
# identically to every other case rather than an invitation to invent a
# fresh convention. tests/test_examples/test_ieee_118_bus.py asserts that
# no edge in the artefact carries it.
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
            "the IEEE 118-bus topology is not connected; every builder and "
            "test downstream assumes it is"
        )

    # One stream, nodes before edges, exactly as the synthetic generators
    # order it -- so every case's weights are drawn by the same procedure
    # and differ only in the seed and in how many draws each takes.
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
            "the IEEE 118-bus case -- the placement is tied to this specific "
            "topology, so it must be revisited if the parser's output changes"
        )

    neighbors = {node: set() for node in nodes}
    for u, v in edges:
        neighbors[u].add(v)
        neighbors[v].add(u)

    # Node faults first, off the drawn (healthy) couplings, so w_i is
    # -(full original-graph neighbourhood coupling) - 1 exactly as in the
    # other cases. The edge-fault loop below is a no-op here, since
    # FAULTY_EDGES is empty; the ordering is kept anyway so the rule stays
    # the one every other case follows if an edge is ever added.
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
        "# Weights for the IEEE 118-bus case, randomly generated (seeded).\n"
        f"# Generated by generate_ieee_118_bus.py, seed={seed}.\n"
        f"# Faulty nodes: {result['faulty_nodes']}. "
        f"Faulty edges: {result['faulty_edges']}.\n"
        '"""Linear-consensus weights for the IEEE 118-bus topology.\n'
        "\n"
        "GENERATED, SEEDED DATA -- do not hand-edit; re-run\n"
        "generate_ieee_118_bus.py instead.\n"
        "\n"
        "Only the TOPOLOGY of this case comes from the IEEE 118-bus test\n"
        "case (see ieee_118_bus_topology.py). The dynamics are linear\n"
        "consensus and these weights are drawn from one seed by exactly\n"
        "the rules the synthetic cases use -- w_i ~ U[1,5],\n"
        "w_ij ~ U[1,10] -- so nothing here is IEEE data. Faults are then\n"
        "applied on top: the faulty nodes' w_i is set to minus their full\n"
        "original-graph neighbourhood coupling minus one, and the faulty\n"
        "branches are set to -20 -- though this case has none, so every\n"
        "branch weight here is the drawn value.\n"
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
