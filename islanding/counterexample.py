"""Counterexample witnesses for claimed-stable islands and components.

See docs/counterexample_spec.md for the full specification. The linear
method builds M = D + L for a unit and returns zero-padded, boundary-
scaled eigenvectors for every non-positive eigenvalue -- genuine
instabilities, as opposed to F5/F6's weaker k_1/k_2-decay-rate hinge.

Deliberately does not import islanding.cost (which imports this module to
wire counterexample=linear_counterexample(...) into linear_consensus_model)
-- canon() is duplicated locally to avoid a circular import.
"""

import numpy as np

from islanding.search import decode_islanding, decode_reconnection


def _canon(edge):
    i, j = edge
    return (i, j) if i < j else (j, i)


def linear_counterexample(node_weights, edge_weights):
    """SystemModel.counterexample for grounded linear consensus.

    M_u = D_u + L_u over a unit's node/edge set (size |V_u|, independent
    of d); every eigenvalue <= 0 (M_u symmetric since w_ij = w_ji) is a
    genuine instability. Because the dynamics decouple per coordinate
    (dot_x = -(M_u kron I_d) x), M_u's eigenvalues repeat d times but
    never change, so the stability criterion -- and the count of
    witnesses, one per non-positive eigenvalue -- is independent of d.
    Each witness places its eigenvector, zero-padded to node_order, into
    coordinate 0 of an (n, d) array (all other coordinates zero) and
    scales the whole array so its Frobenius norm equals threshold_radius.
    """

    def _find(unit_nodes, unit_edges, node_order, d, threshold_radius):
        unit_nodes = list(unit_nodes)
        index = {node: i for i, node in enumerate(unit_nodes)}
        size = len(unit_nodes)

        M = np.zeros((size, size))
        for node in unit_nodes:
            M[index[node], index[node]] += node_weights[node]

        for u, v in unit_edges:
            w = edge_weights[_canon((u, v))]
            i, j = index[u], index[v]
            M[i, j] -= w
            M[j, i] -= w
            M[i, i] += w
            M[j, j] += w

        eigenvalues, eigenvectors = np.linalg.eigh(M)

        n = len(node_order)
        node_position = {node: pos for pos, node in enumerate(node_order)}

        witnesses = []
        for k, lam in enumerate(eigenvalues):
            if lam > 0:
                continue

            padded = np.zeros((n, d))
            for node in unit_nodes:
                padded[node_position[node], 0] = eigenvectors[index[node], k]

            padded *= threshold_radius / np.linalg.norm(padded)
            witnesses.append(padded)

        return witnesses

    return _find


def find_counterexamples(A, problem):
    """Decode A once, check every stable island and reconnected component.

    Returns (island_witnesses, component_witnesses) -- islands checked
    against Omega's boundary (beta-scaled), components against Omega_hat's
    boundary (gamma-scaled), both inflated by the model's factor.
    """
    topology = problem.topology
    model = problem.model
    config = problem.config

    node_order = sorted(topology.nodes())
    factor = 1.0 + model.inflation(topology)
    d = problem.samples_1.shape[2]

    stable_islands, _ = decode_islanding(A, topology)
    components, _, _ = decode_reconnection(A, topology)

    island_radius = (2.0 * factor * config.beta) ** 0.5
    component_radius = (2.0 * factor * config.gamma) ** 0.5

    island_witnesses = []
    for island in stable_islands:
        island_edges = {
            _canon((u, v))
            for u, v in topology.edges()
            if u in island and v in island
        }
        island_witnesses.extend(
            model.counterexample(island, island_edges, node_order, d, island_radius)
        )

    component_witnesses = []
    for component in components:
        component_witnesses.extend(
            model.counterexample(
                component.nodes(), component.edges(), node_order, d, component_radius
            )
        )

    return island_witnesses, component_witnesses
