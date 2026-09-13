"""Seeded generator for the 10-node random-geometric-graph example.

Writes examples/linear_10node_rgg/linear_10node_rgg_topology.py. Mirrors
examples/linear_20node/generate_linear_20node.py, and exists so the
benchmark's smallest size is drawn by the SAME process as the 20-, 30-
and 40-node cases.

WHY THIS EXISTS ALONGSIDE examples/linear_10node/. That case's topology
is hand-specified, not drawn. Putting it in a scaling table beside three
generated instances implies a homogeneity that does not hold, and the
honest fix is another instance at the same size rather than a caveat.
Both are kept: the hand-specified one is still the fixture several tests
and specs are written against.

THE RULE, applied without selection. Node positions uniform on a square,
an edge wherever the Euclidean distance is at most EDGE_RADIUS = 2,
redrawn from scratch at the next seed until the graph is connected. The
side is sqrt(20) = 4.4721, which puts node density at 10 / 4.4721^2 =
0.500 -- the figure the 30-, 40- and 50-node generators state and hold.

THE RESULTING AVERAGE DEGREE IS 3.80, below the 20-node case's 4.20 and
the 30-/40-node cases' ~5.45. That is a finite-size effect of random
geometric graphs, not a defect: at small N a larger fraction of nodes sit
near the boundary of the square and so have fewer neighbours within the
radius. Holding DENSITY fixed therefore does not hold degree fixed, and
the 20-node case already departs the other way. The seed was NOT chosen
to bring the degree closer to the larger cases -- "we applied the same
rule at each size" is a claim that survives scrutiny, and "we applied the
rule, then picked the seed whose degree looked most like the others"
invites the question of what the other seeds gave.

Usage:
    python examples/linear_10node_rgg/generate_linear_10node_rgg.py [seed]

Default: seed=0. Seeds 0 and 1 draw disconnected graphs (two components
each), so the retry loop advances to seed 2, which is what the committed
artefact records. Running with no argument reproduces it exactly, which
tests/test_examples/test_linear_10node_rgg.py checks field by field and
byte for byte.
"""

import sys
from pathlib import Path

import networkx as nx
import numpy as np

HERE = Path(__file__).parent
OUTPUT_PATH = HERE / "linear_10node_rgg_topology.py"

N_NODES = 10
# sqrt(N / 0.500): the side that puts this case at the same node density
# as the 30-, 40- and 50-node generators, which is the parameter those
# generators are written to control.
PLANE_SIZE = 20.0 ** 0.5
EDGE_RADIUS = 2.0
# The seed the retry loop STARTS from. The artefact records the seed it
# reached, which is 2 -- see the module docstring.
DEFAULT_SEED = 0

# Explicit fault placement, chosen against this topology's structure.
# Every figure below is checked by
# tests/test_examples/test_linear_10node_rgg.py rather than asserted here.
#
# FAULTY_NODES: node 5. Interior, degree 4, and NOT the graph's only
# articulation point. Chosen over the structurally equivalent node 7 --
# same degree, same neighbour set apart from each other -- because its
# full-neighbourhood coupling is larger (29.00 against 23.77), so
# shedding it removes more coupling and exercises F1 where it costs most.
#
# NODE 3 IS AVOIDED, and it is the one node that must be. It is the only
# articulation point, and it sits at the near end of the bridge (3, 9):
# shedding 3 strands the leaf 9 as a one-node island, which is a
# degenerate instance rather than a useful one.
#
# JOINTLY, NOT ONLY INDIVIDUALLY. Three node pairs separate this graph
# even though no member of any of them is an articulation point:
#
#     {1, 8}   -> 4 / 4, detaching {2, 4, 6, 10}
#     {1, 10}  -> 3 / 5, detaching {2, 4, 6}
#     {6, 10}  -> 1 / 7, isolating {2}
#
# The last is trivial for the substantial-component analysis -- a single
# node -- but belongs with the others: a set of non-cut vertices can
# still be a cut set, and the 30-bus and 118-bus cases were both shaped
# by exactly that trap. This placement faults a single node so the
# question does not arise here, but it would if the count ever rose.
#
# FAULTY_EDGES: (1, 7) and (3, 8). Both lie inside the upper cluster
# {1, 3, 5, 7, 8, 9}, neither is incident to node 5, and they are
# vertex-disjoint, so no node is reduced to degree 1 by losing both. They
# are the heaviest such pair (8.64 + 9.03), so the fault again costs real
# coupling.
#
# DELIBERATELY AWAY FROM THE CONNECTIVITY BOTTLENECKS. There are TWO
# four-edge cuts that split this graph into substantial halves:
#
#     {(1,4), (1,6), (1,10), (8,10)}   ->  4 / 6, detaching {2,4,6,10}
#     {(1,5), (1,7), (1,8), (8,10)}    ->  5 / 5, detaching {1,2,4,6,10}
#
# and (1,7) does lie on the second. Neither faulty edge was chosen to sit
# on a cut, and two faulty edges cannot open a four-edge cut on their own.
#
# THAT DOES NOT MEAN NO RECONNECTION, and an earlier version of this
# comment drew exactly that false conclusion. THE REASONING ERROR IS
# WORTH RECORDING, because it is easy to repeat: a faulty edge does not
# have to form a graph CUT to force its endpoints into different islands.
# Its -20 weight makes any island containing BOTH endpoints unstable, so
# the partition must separate them however well connected the graph is.
# Healthy edges that happen to cross that induced separation then become
# eligible reconnection candidates. Asking only "do the faulty edges
# disconnect the graph?" misses this completely.
#
# WHAT ACTUALLY HAPPENS HERE. Shedding node 5 and separating the faulty
# edges' endpoints gives two stable islands
#
#     {1, 2, 4, 6, 8, 10}   lambda_min 3.192
#     {3, 7, 9}             lambda_min 2.831
#
# with three edges crossing between them: (1,7) and (3,8), both faulty
# and so cut, and (7,8), HEALTHY -- an eligible reconnection edge. The
# local search converges in two inner solves, returns exactly those
# islands and reconnects (7,8).
#
# So this instance exercises BOTH islanding and reconnection. Note that
# the union of the two islands joined by (7,8) has lambda_min 3.017: that
# figure is the RECONNECTED COMPONENT, and citing it as evidence of a
# single un-split island was the other half of the earlier mistake.
FAULTY_NODES = [5]
FAULTY_EDGES = [(1, 7), (3, 8)]

# Faulty edges get a strongly negative coupling, so any island containing
# both endpoints is unstable and the edge must be cut.
FAULTY_EDGE_WEIGHT = -20.0


def _canon(edge):
    i, j = edge
    return (i, j) if i < j else (j, i)


def _draw_positions_and_edges(rng):
    nodes = list(range(1, N_NODES + 1))
    positions = {
        node: tuple(float(c) for c in rng.uniform(0.0, PLANE_SIZE, size=2))
        for node in nodes
    }

    edges = set()
    for a in range(len(nodes)):
        for b in range(a + 1, len(nodes)):
            i, j = nodes[a], nodes[b]
            xi, yi = positions[i]
            xj, yj = positions[j]
            distance = ((xi - xj) ** 2 + (yi - yj) ** 2) ** 0.5
            if distance <= EDGE_RADIUS:
                edges.add(_canon((i, j)))

    return nodes, positions, sorted(edges)


def _generate_connected(seed):
    """Draw for `seed`; if disconnected, advance the seed and redraw.

    Returns (successful_seed, rng, nodes, positions, edges) with `rng`
    left mid-stream, so weight generation continues from the SAME seeded
    stream and "all randomness flows from one seed" stays true.
    """
    candidate_seed = seed
    while True:
        rng = np.random.default_rng(candidate_seed)
        nodes, positions, edges = _draw_positions_and_edges(rng)

        graph = nx.Graph()
        graph.add_nodes_from(nodes)
        graph.add_edges_from(edges)

        if nx.is_connected(graph):
            return candidate_seed, rng, nodes, positions, edges

        candidate_seed += 1


def generate(seed):
    successful_seed, rng, nodes, positions, edges = _generate_connected(seed)

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
            f"FAULTY_EDGES references {missing}, which are not edges of this "
            "draw -- the placement is tied to a specific topology, so it must "
            "be revisited if the seed or plane changes"
        )

    neighbors = {node: set() for node in nodes}
    for u, v in edges:
        neighbors[u].add(v)
        neighbors[v].add(u)

    # Node faults first, off the drawn (healthy) couplings, so w_i is
    # -(full original-graph neighbourhood coupling) - 1 exactly as in the
    # other cases. No faulty edge is incident to node 5 here, so the
    # order does not actually matter -- but fixing it keeps the rule
    # unambiguous if that ever changes.
    for node in FAULTY_NODES:
        full_coupling = sum(
            edge_weights[_canon((node, j))] for j in neighbors[node]
        )
        node_weights[node] = -full_coupling - 1.0

    for edge in faulty_edges:
        edge_weights[edge] = FAULTY_EDGE_WEIGHT

    return {
        "seed": successful_seed,
        "nodes": nodes,
        "edges": edges,
        "node_weights": node_weights,
        "edge_weights": edge_weights,
        "positions": positions,
        "faulty_nodes": list(FAULTY_NODES),
        "faulty_edges": faulty_edges,
    }


def write_topology(result, output_path):
    seed = result["seed"]
    content = (
        "# 10-node random geometric graph example, randomly generated "
        "(seeded).\n"
        f"# Generated by generate_linear_10node_rgg.py, seed={seed}.\n"
        f"# Faulty nodes: {result['faulty_nodes']}. "
        f"Faulty edges: {result['faulty_edges']}.\n"
        '"""10-node random-geometric-graph topology, randomly generated '
        '(seeded).\n'
        "\n"
        "Positions are drawn uniformly on a sqrt(20) x sqrt(20) plane; an\n"
        "edge connects every pair of nodes within Euclidean distance 2.\n"
        "The side puts node density at 0.500, the figure the 30-, 40- and\n"
        "50-node generators hold. The resulting average degree is 3.80,\n"
        "BELOW the larger cases' 4.20 and ~5.45: that is a finite-size\n"
        "effect of random geometric graphs, not a defect, and the seed was\n"
        "not chosen to hide it. See generate_linear_10node_rgg.py.\n"
        "\n"
        "The seed recorded here is the one that produced a connected graph\n"
        "(seeds 0 and 1 are disconnected). Weights are part of the\n"
        "artifact, not just the topology -- this case is not reproducible\n"
        "from NODES/EDGES alone. Both this module and its generator are\n"
        "committed; re-run the generator rather than hand-editing.\n"
        '"""\n'
        "\n"
        f"NODES = {result['nodes']!r}\n"
        "\n"
        f"EDGES = {result['edges']!r}\n"
        "\n"
        f"NODE_WEIGHTS = {result['node_weights']!r}\n"
        "\n"
        f"EDGE_WEIGHTS = {result['edge_weights']!r}\n"
        "\n"
        f"POSITIONS = {result['positions']!r}\n"
    )
    Path(output_path).write_text(content)


def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_SEED

    result = generate(seed)
    write_topology(result, OUTPUT_PATH)

    graph = nx.Graph()
    graph.add_nodes_from(result["nodes"])
    graph.add_edges_from(result["edges"])

    n_nodes = len(result["nodes"])
    n_edges = len(result["edges"])

    print(f"Wrote {OUTPUT_PATH}")
    print(f"Successful seed: {result['seed']} (started from {seed})")
    print(f"Nodes: {n_nodes}")
    print(f"Edges: {n_edges}")
    print(f"Plane: {PLANE_SIZE:.4f} x {PLANE_SIZE:.4f} "
          f"(density {n_nodes / PLANE_SIZE ** 2:.3f})")
    print(f"Connected: {nx.is_connected(graph)}")
    print(f"Average degree: {2 * n_edges / n_nodes:.2f}")
    print(f"Faulty nodes: {result['faulty_nodes']}")
    print(f"Faulty edges: {result['faulty_edges']}")


if __name__ == "__main__":
    main()
