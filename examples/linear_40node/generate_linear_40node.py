"""Seeded generator for the 40-node linear consensus example.

Writes examples/linear_40node/linear_40node_topology.py. Both this
generator and its output are committed -- the generator is our own code
and seeded, so there is no copyright or reproducibility reason to hide it.
Mirrors examples/linear_30node/generate_linear_30node.py so the cases are
directly comparable.

The plane is 8.94x8.94, between the 30-node case's 7.75x7.75 and the
50-node case's 10x10. That is the whole point of the size: at the same
edge radius, node density -- and hence average degree -- has to match, or
a runtime comparison across the cases measures density rather than size.
Area scales with node count, so 40/8.94^2 = 0.500 nodes per unit area
reproduces the 50-node case's 50/10^2 = 0.500 and the 30-node case's
30/7.75^2 = 0.500, preserving the expected degree at radius 2. The 50-node
case has 136 edges over 50 nodes (average degree 5.44); the target here is
roughly 109 (= 136 * 40/50). Both dimensions bear on cost: edge count drives the
certificate families and the local search's neighbourhood, while node
count drives the MILP's assignment binaries and the sample arrays.
Holding density fixed is what keeps a cross-case comparison about
size rather than about shape.

Faults are placed EXPLICITLY here (FAULTY_NODES / FAULTY_EDGES below)
rather than taken as "the first n nodes" like the 20- and 50-node cases.
Arbitrary faults do not fragment a graph at this density, so those cases
converge to a single stable island and never produce a reconnected edge,
leaving F4 and F6 unexercised. Explicit placement targets a measured
bottleneck instead -- the placement was chosen only after generating the
healthy graph and inspecting its articulation points and cluster
boundaries, since it is tied to this specific draw. See the constants for
the reasoning, including the two candidate clusters that were rejected.

Usage:
    python examples/linear_40node/generate_linear_40node.py [seed]

Default: seed=36 -- the seed the committed artifact was generated with,
and the draw that hits the target density (109 edges, average degree 5.45;
the 50-node case is 5.44). Running with no argument therefore reproduces
the artifact exactly, which tests/test_examples/test_linear_40node.py
checks field by field.
"""

import sys
from pathlib import Path

import networkx as nx
import numpy as np

HERE = Path(__file__).parent
OUTPUT_PATH = HERE / "linear_40node_topology.py"

N_NODES = 40
PLANE_SIZE = 8.94
EDGE_RADIUS = 2.0
# The seed the COMMITTED artefact was generated with, so the documented
# no-argument command reproduces it. Not 0: seed 0's draw does not contain
# the edges FAULTY_EDGES names, so generate(0) raises rather than
# producing anything. The connectivity-retry loop cannot rescue that --
# it only advances past DISCONNECTED draws, and seed 0's draw is connected,
# just differently shaped.
DEFAULT_SEED = 36

# Explicit fault placement, chosen against this draw's structure.
#
# Arbitrary faults do not fragment a graph at this density, so a case with
# faults scattered at random converges to a single stable island and never
# produces a reconnected edge -- leaving F4 and F6 unexercised. That is
# what happens in the 20- and 50-node cases, which take "the first n
# nodes" as faulty. Placement here targets a measured bottleneck instead.
#
# FAULTY_EDGES: two of the THREE edges crossing the cluster
# {4, 7, 16, 19, 27, 32, 33, 38, 39, 40} -- (10,16), (17,38) and (26,38).
# (26,38) is deliberately left HEALTHY as the single reconnection
# candidate. This is the fault-(n-1)-of-n pattern that produced a
# reconnection at 30 nodes.
#
# What this CREATES is a feasible reconnection opportunity, not a forced
# outcome. Faulting two of the three crossings does not force the split:
# the graph stays connected through the healthy (26,38), and forcing the
# pairs (10,16) and (17,38) apart does not uniquely determine that the
# 10-node cluster becomes one island.
#
# What IS established, computed on this draw:
#
#   - cutting all THREE crossings separates the cluster from its
#     complement exactly, leaving components of 10 and 30 nodes;
#   - the cluster is connected with lambda_min = 2.982, so it is a
#     genuinely stable candidate island;
#   - after shedding the faulty nodes 8 and 24, the complement is still
#     connected with lambda_min = 2.509, so it is too;
#   - {8, 24} is itself connected and unstable (lambda_min = -57.4), so
#     shedding it as one unstable island is coherent.
#
# So (26,38) joins two genuinely stable candidate islands and would be
# ELIGIBLE for reconnection under the E^rec rules. Whether either solver
# actually selects that partition -- and therefore whether F4 and F6 come
# out nonzero -- is an outcome of the search, not a property of the
# placement. tests/test_examples/test_linear_40node.py pins the structural
# facts above; only a run can settle the rest.
#
# Two better-looking clusters were rejected:
#
#   The modularity-cleanest bottleneck is {2,3,11,13,23,25,28,34,35,37},
#   which Louvain, greedy modularity and a Fiedler bisection all agree on
#   independently. Its cut is only TWO edges, (5,35) and (13,14), so
#   fault-(n-1) means faulting one and leaving one. That is too weak to
#   force anything: the cluster stays attached by a healthy edge, and the
#   search can keep everything in a single island for the price of one cut
#   edge in F3 -- which would leave F4/F6 unexercised, the very outcome
#   this placement exists to avoid.
#
#   {10,18,36} has a 3-edge cut, but all three crossings are incident to
#   node 10, so cutting them isolates one node's neighbourhood rather than
#   splitting a cluster, and the resulting island is 3 nodes.
#
# FAULTY_NODES: 24 (degree 11) and 8 (degree 10), both on the MAIN side of
# the cut. 24 is the single highest-degree node; 8 is one of TWO nodes tied
# at degree 10 -- node 9 is the other, and either would have served. Same
# rationale as the 30-node placement: node faults land away from the
# bottleneck so the edge faults, not the node faults, are what opens the
# split, and shedding a high-degree node exercises F1 without
# disconnecting anything.
#
# The five articulation points -- 2, 10, 23, 25, 36 -- were all avoided.
# Read off the graph, each detaches:
#
#     remove 2   ->  {11, 23}          (connected, via (11,23))
#     remove 10  ->  {18, 36}          (connected, via (18,36))
#     remove 23  ->  {11}
#     remove 25  ->  {3}  and  {2, 11, 23, 28}
#                    (the 4-group is connected, via (2,23),(2,28),(11,23))
#     remove 36  ->  {18}
#
# So these are not all leaf-stranding, and the 4-group behind 25 IS
# internally connected. The reason to avoid them is simpler than that: a
# fault at an articulation point splits the graph on ITS terms rather than
# at the bottleneck this placement targets, adding a second, competing
# partition pressure and confounding what the edge faults are meant to
# test. The detached pieces are also 1-4 nodes -- far smaller than the
# 10-node cluster -- so they make thin islands and, being cut off by node
# removal rather than by a cut edge, offer no reconnection candidate.
#
# Both lists are validated against the drawn graph in generate(): an
# unknown node or a non-edge raises rather than being silently ignored,
# because a placement is tied to one specific draw and must be revisited
# if the seed or the plane changes.
FAULTY_NODES = [8, 24]
FAULTY_EDGES = [(10, 16), (17, 38)]

# Faulty nodes get w_i = -(sum of w_ij over the FULL neighbourhood) - 1,
# so the node is unstable in isolation and in any island that keeps it.
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
        "# 40-node linear consensus example, randomly generated (seeded).\n"
        f"# Generated by generate_linear_40node.py, seed={seed}.\n"
        f"# Faulty nodes: {result['faulty_nodes']}. "
        f"Faulty edges: {result['faulty_edges']}.\n"
        '"""40-node linear consensus topology, randomly generated (seeded).\n'
        "\n"
        "Positions are drawn uniformly on an 8.94x8.94 plane; an edge\n"
        "connects every pair of nodes within Euclidean distance 2. The\n"
        "plane is sized so that node density, and hence average\n"
        "degree, matches the 30- and 50-node cases -- see\n"
        "generate_linear_40node.py. Faults are placed explicitly (see that\n"
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
