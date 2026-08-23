"""Seeded generator for the 30-node linear consensus example.

Writes examples/linear_30node/linear_30node_topology.py. Unlike the IEEE
30-bus case, both this generator and its output are committed -- the
generator is our own code and seeded, so there's no copyright or
reproducibility reason to hide it. Mirrors
examples/linear_50node/generate_linear_50node.py so the two cases are
directly comparable.

The plane is 7.75x7.75 rather than the 50-node case's 10x10. That is the
whole point of the size: at the same edge radius, node density -- and
hence average degree -- has to match, or a runtime comparison across the
cases measures density rather than size. Area scales with node count, so
30/7.75^2 = 0.500 nodes per unit area reproduces the 50-node case's
50/10^2 = 0.500, and the expected degree at radius 2 is preserved. The
50-node case has 136 edges over 50 nodes (average degree ~5.4); the
target here is roughly 82 (= 136 * 30/50). Both dimensions bear on cost: edge count drives the
certificate families and the local search's neighbourhood, while node
count drives the MILP's assignment binaries and the sample arrays.
Holding density fixed is what keeps a cross-case comparison about
size rather than about shape.

Faults are placed EXPLICITLY here (FAULTY_NODES / FAULTY_EDGES below)
rather than taken as "the first n nodes" like the 20- and 50-node cases.
Arbitrary faults do not fragment a graph at this density, so those cases
converge to a single stable island and never produce a reconnected edge,
leaving F4 and F6 unexercised. Explicit placement targets a known
bottleneck instead -- see the constants for the reasoning.

Usage:
    python examples/linear_30node/generate_linear_30node.py [seed]

Default: seed=26 -- the seed the committed artifact was generated with,
and the draw that hits the target density (82 edges, average degree 5.47).
Running with no argument therefore reproduces the artifact exactly, which
tests/test_examples/test_linear_30node.py checks field by field.
"""

import sys
from pathlib import Path

import networkx as nx
import numpy as np

HERE = Path(__file__).parent
OUTPUT_PATH = HERE / "linear_30node_topology.py"

N_NODES = 30
PLANE_SIZE = 7.75
EDGE_RADIUS = 2.0
# The seed the COMMITTED artefact was generated with, so the documented
# no-argument command reproduces it. Not 0: seed 0's draw does not contain
# the edges FAULTY_EDGES names, so generate(0) raises rather than
# producing anything. The connectivity-retry loop cannot rescue that --
# it only advances past DISCONNECTED draws, and seed 0's draw is connected,
# just differently shaped.
DEFAULT_SEED = 26

# Explicit fault placement, chosen against this topology's structure.
#
# FAULTY_NODES: 5 and 16 sit in the dense lower cluster (degrees 7 and 9),
# so shedding them exercises F1 without disconnecting anything. They
# deliberately avoid nodes 3 and 19, this graph's only two articulation
# points, whose removal would strand the leaves 7 and 13 respectively --
# stranding a leaf produces a trivial island rather than a useful test.
#
# FAULTY_EDGES: the crossing edges between the upper cluster
# {9,10,11,12,18,27,28} and the rest. Exactly five such edges exist --
# (10,25), (11,22), (11,25), (12,24), (18,24) -- and four are faulted,
# leaving (11,22) healthy as the single reconnection candidate.
#
# What this CREATES is a feasible reconnection opportunity, not a forced
# outcome. Faulting four of the five does not force the split: the graph
# stays connected through the healthy (11,22), and forcing the four faulty
# endpoint pairs apart does not uniquely determine that the cluster
# becomes one island. What it does establish is that cutting all five
# crossings separates the cluster from its complement, so (11,22) joins
# two candidate islands and would be ELIGIBLE for reconnection -- the
# precondition for E^rec to be non-empty. Whether either solver selects
# that partition, and therefore whether F4 and F6 come out nonzero, is an
# outcome of the search, not a property of the placement.
FAULTY_NODES = [5, 16]
FAULTY_EDGES = [(10, 25), (11, 25), (12, 24), (18, 24)]

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
    """Draw positions/edges for `seed`; if the resulting graph isn't
    connected, increment the seed and redraw from scratch, repeating
    until connected. Returns (successful_seed, rng, nodes, positions,
    edges) -- `rng` is left mid-stream so weight generation (step 4)
    continues drawing from the SAME seeded stream, keeping "all
    randomness flows from one seed" true for the whole generation.
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
    # other cases. The two fault sets are disjoint here (no faulty edge is
    # incident to a faulty node), so the order does not actually matter --
    # but fixing it keeps the rule unambiguous if that ever changes.
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
        "# 30-node linear consensus example, randomly generated (seeded).\n"
        f"# Generated by generate_linear_30node.py, seed={seed}.\n"
        f"# Faulty nodes: {result['faulty_nodes']}. "
        f"Faulty edges: {result['faulty_edges']}.\n"
        '"""30-node linear consensus topology, randomly generated (seeded).\n'
        "\n"
        "Positions are drawn uniformly on a 7.75x7.75 plane; an edge\n"
        "connects every pair of nodes within Euclidean distance 2. The\n"
        "plane is smaller than the 50-node case's 10x10 so that node\n"
        "density, and hence average degree, is comparable -- see\n"
        "generate_linear_30node.py. Faults are placed explicitly (see that\n"
        "generator's FAULTY_NODES/FAULTY_EDGES and the reasoning beside\n"
        "them), not taken as the first n nodes. The seed recorded here is\n"
        "the one that produced a connected graph (see that generator's\n"
        "connectivity-retry loop). Weights are part of the artifact, not\n"
        "just the topology -- this case is not reproducible from\n"
        "NODES/EDGES alone. Both this module and its generator are\n"
        "committed; re-run the generator to reproduce or vary the\n"
        "seed or fault placement instead of hand-editing this file.\n"
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
    print(f"Successful seed: {result['seed']}")
    print(f"Nodes: {n_nodes}")
    print(f"Edges: {n_edges}")
    print(f"Connected: {nx.is_connected(graph)}")
    print(f"Average degree: {2 * n_edges / n_nodes:.2f}")
    print(f"Faulty nodes: {result['faulty_nodes']}")
    print(f"Faulty edges: {result['faulty_edges']}")


if __name__ == "__main__":
    main()
