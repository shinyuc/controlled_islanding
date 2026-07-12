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


def decode_reconnection(tree, topology):
    """Decode a spanning tree into post-islanding reconnected components.

    Starts from decode_islanding's stable islands, then reconnects pairs
    of islands whose original edge's relay path is "middle-cut" -- both
    endpoints still attach to their own relay, but the relay-to-relay
    edge itself is missing from the tree. Reconnected islands are merged
    into one component per connected group.

    Each component's edges come from two disjoint sources: the original
    edges induced WITHIN each constituent island (computed per island,
    never over the merged node set -- an edge whose endpoints land in the
    same merged component only because two OTHER islands got reconnected
    must not be readmitted), plus the reconnected edges joining islands
    within that group.
    """
    islands, Z = decode_islanding(tree, topology)

    node_to_island = {}
    for index, island in enumerate(islands):
        for node in island:
            node_to_island[node] = index

    reconnected_edges = []
    for u, v in topology.edges():
        u, v = (u, v) if u < v else (v, u)
        if u not in node_to_island or v not in node_to_island:
            continue
        idx_u, idx_v = node_to_island[u], node_to_island[v]
        if idx_u == idx_v:
            continue
        relay_u, relay_v = ("n", u, v, u), ("n", u, v, v)
        if (tree.has_edge(u, relay_u) and tree.has_edge(relay_v, v)
                and not tree.has_edge(relay_u, relay_v)):
            reconnected_edges.append((u, v, idx_u, idx_v))

    merge_graph = nx.Graph()
    merge_graph.add_nodes_from(range(len(islands)))
    for _, _, idx_u, idx_v in reconnected_edges:
        merge_graph.add_edge(idx_u, idx_v)

    components = []
    for group in nx.connected_components(merge_graph):
        constituent_islands = tuple(sorted((islands[i] for i in group), key=min))

        component = nx.Graph()
        for island in constituent_islands:
            component.add_nodes_from(island)
            for u, v in topology.edges():
                u, v = (u, v) if u < v else (v, u)
                if u in island and v in island:
                    component.add_edge(u, v)
        for u, v, idx_u, idx_v in reconnected_edges:
            if idx_u in group and idx_v in group:
                component.add_edge(u, v)

        component.graph["islands"] = constituent_islands
        components.append(component)

    components.sort(key=lambda component: min(component.nodes()))

    return components, Z
