"""Local search machinery for Algorithm 1: initial spanning tree seed and
decoding a spanning tree of Gbar into an islanding.
"""

import random

import networkx as nx

from islanding.augmented import SOURCE, SINK


# The tree-initialisation seed, distinct from whatever seed a caller used to
# draw the sample sets. Named so a run script can RECORD which start it got
# without duplicating the literal: local_search calls
# initial_spanning_tree(...) with no seed argument, so every local-search
# result is a single-start result at this value unless a caller overrides it.
DEFAULT_TREE_SEED = 0


def initial_spanning_tree(gbar, seed=DEFAULT_TREE_SEED):
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


def _project_components(subgraph, original_nodes):
    result = []
    for component in nx.connected_components(subgraph):
        projected = frozenset(component & original_nodes)
        if projected:
            result.append(projected)
    result.sort(key=min)
    return result


def decode_islanding(tree, topology):
    """Decode a spanning tree of Gbar into stable and unstable islands.

    Removing the (SOURCE, SINK) edge splits the tree into a SOURCE-side
    component (stable nodes) and a SINK-side component (failed nodes).
    On each side, the supernode itself must be dropped before taking
    connected components: SOURCE (resp. SINK) is adjacent to every
    original node, so leaving it in would merge all islands on that side
    into one. What remains connects original nodes only through intact
    relay paths, so its connected components -- projected onto original
    nodes -- are exactly the stable (resp. unstable) islands.
    """
    split = tree.copy()
    split.remove_edge(SOURCE, SINK)
    components = list(nx.connected_components(split))
    sink_component = next(c for c in components if SINK in c)
    source_component = next(c for c in components if SOURCE in c)

    original_nodes = set(topology.nodes())

    stable_subgraph = split.subgraph(source_component - {SOURCE})
    stable_islands = _project_components(stable_subgraph, original_nodes)

    unstable_subgraph = split.subgraph(sink_component - {SINK})
    unstable_islands = _project_components(unstable_subgraph, original_nodes)

    return stable_islands, unstable_islands


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
    stable_islands, unstable_islands = decode_islanding(tree, topology)

    node_to_island = {}
    for index, island in enumerate(stable_islands):
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

    reconnected_edge_set = {(u, v) for u, v, _, _ in reconnected_edges}
    components = build_reconnected_components(
        stable_islands, reconnected_edge_set, topology
    )

    return components, unstable_islands, reconnected_edge_set


def build_reconnected_components(stable_islands, reconnected_edges, topology):
    """Group stable islands into reconnected components.

    Split out of decode_reconnection so the same construction serves
    callers that already hold a partition (an inner solver's InnerResult)
    and never had a spanning tree to decode -- see
    counterexample.find_counterexamples_for_partition. decode_reconnection
    itself calls this, so there is exactly one implementation.

    `reconnected_edges` is E^rec as canonical (min, max) tuples; both
    endpoints of every such edge must lie in some stable island, which is
    what E^rec's definition guarantees.

    Each component's edges come from two disjoint sources: the original
    edges induced WITHIN each constituent island (computed per island,
    never over the merged node set -- an edge whose endpoints land in the
    same merged component only because two OTHER islands got reconnected
    must not be readmitted), plus the reconnected edges joining islands
    within that group.
    """
    node_to_island = {}
    for index, island in enumerate(stable_islands):
        for node in island:
            node_to_island[node] = index

    grouped_edges = []
    merge_graph = nx.Graph()
    merge_graph.add_nodes_from(range(len(stable_islands)))
    for u, v in reconnected_edges:
        u, v = (u, v) if u < v else (v, u)
        idx_u, idx_v = node_to_island[u], node_to_island[v]
        merge_graph.add_edge(idx_u, idx_v)
        grouped_edges.append((u, v, idx_u, idx_v))

    components = []
    for group in nx.connected_components(merge_graph):
        constituent_islands = tuple(sorted((stable_islands[i] for i in group), key=min))

        component = nx.Graph()
        for island in constituent_islands:
            component.add_nodes_from(island)
            for u, v in topology.edges():
                u, v = (u, v) if u < v else (v, u)
                if u in island and v in island:
                    component.add_edge(u, v)
        for u, v, idx_u, idx_v in grouped_edges:
            if idx_u in group and idx_v in group:
                component.add_edge(u, v)

        component.graph["islands"] = constituent_islands
        components.append(component)

    components.sort(key=lambda component: min(component.nodes()))
    return components
