"""Local search machinery for Algorithm 1: initial spanning tree seed and
decoding a spanning tree of Gbar into an islanding.
"""

import random

import networkx as nx

from islanding.augmented import SOURCE, SINK


def initial_spanning_tree(gbar, seed=0):
    """Seeded-random spanning tree of gbar that always contains (SOURCE, SINK).

    Assigns (SOURCE, SINK) a weight lower than every other edge and every
    other edge a random weight in [0, 1) drawn from a seeded RNG, then
    takes the minimum spanning tree. The globally minimum-weight edge is
    always part of an MST, so (SOURCE, SINK) is guaranteed to be included.
    """
    rng = random.Random(seed)
    weighted = gbar.copy()
    for u, v in weighted.edges():
        weighted[u][v]["weight"] = rng.random()
    weighted[SOURCE][SINK]["weight"] = -1.0

    return nx.minimum_spanning_tree(weighted, weight="weight")


def decode_islanding(tree, topology):
    """Decode a spanning tree of Gbar into stable islands and failed nodes.

    Removing the (SOURCE, SINK) edge splits the tree into a SOURCE-side
    component (stable nodes) and a SINK-side component (failed nodes, Z).
    Within the SOURCE-side component, SOURCE itself must be dropped before
    taking connected components: SOURCE is adjacent to every original
    node, so leaving it in would merge all islands into one. What remains
    connects original nodes only through intact relay paths, so its
    connected components -- projected onto original nodes -- are exactly
    the stable islands.
    """
    split = tree.copy()
    split.remove_edge(SOURCE, SINK)
    components = list(nx.connected_components(split))
    sink_component = next(c for c in components if SINK in c)
    source_component = next(c for c in components if SOURCE in c)

    original_nodes = set(topology.nodes())
    Z = frozenset(sink_component & original_nodes)

    stable_remainder = source_component - {SOURCE}
    island_subgraph = split.subgraph(stable_remainder)

    islands = []
    for component in nx.connected_components(island_subgraph):
        projected = frozenset(component & original_nodes)
        if projected:
            islands.append(projected)
    islands.sort(key=min)

    return islands, Z
