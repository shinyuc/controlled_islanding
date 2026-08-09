"""Generic example-builder logic, shared across examples.

`Setup` bundles everything that varies per problem instance (node/edge
weights, node/edge costs). The builders here depend only on a `Setup`
or on explicit weights/topology passed in -- never on a specific
example's topology -- so they're reusable across examples. Each example
module supplies its own topology-specific `build_topology()` and NODES/
EDGES, and passes the topology in wherever one is needed (`build_M`,
`build_problem`).

Importing this module defines builders only -- it never runs a solve or
any computation beyond definitions.
"""

from dataclasses import dataclass

import numpy as np

from islanding.cost import CostData, AlgorithmConfig, linear_consensus_model
from islanding.evaluate import Problem, draw_samples

D = 2  # state dimension x_i in R^D, shared by all current examples


def _canon(edge):
    i, j = edge
    return (i, j) if i < j else (j, i)


@dataclass(frozen=True)
class Setup:
    name: str
    node_weights: dict
    edge_weights: dict
    node_costs: dict
    edge_costs: dict


def build_config():
    return AlgorithmConfig(
        kappa_op=1.0, kappa_rec=0.7,
        alpha_1=1.0, beta=1.0, k_1=0.2, P_1=20,
        alpha_2=1.0, gamma=1.0, k_2=0.2, P_2=20,
        epsilon=0.01, max_iterations=100, swap_supernode_edges=True,
    )


def build_M(nodes, edges, node_weights, edge_weights):
    """Build M = D + L for a unit. `nodes` is the unit's node set, `edges`
    its edge set -- for a reconnected component, pass component.edges()
    (the E_s^rec set), not an induced subgraph.
    """
    nodes = list(nodes)
    index = {node: i for i, node in enumerate(nodes)}
    size = len(nodes)

    M = np.zeros((size, size))
    for node in nodes:
        M[index[node], index[node]] += node_weights[node]

    for u, v in edges:
        w = edge_weights[_canon((u, v))]
        i, j = index[u], index[v]
        M[i, j] -= w
        M[j, i] -= w
        M[i, i] += w
        M[j, j] += w

    return M


def build_model(setup):
    return linear_consensus_model(setup.node_weights, setup.edge_weights)


def build_cost_data(setup):
    return CostData(node_costs=setup.node_costs, edge_costs=setup.edge_costs)


def build_problem(setup, topology, seed=None):
    model = build_model(setup)
    cost_data = build_cost_data(setup)
    config = build_config()

    if seed is not None:
        np.random.seed(seed)

    n = len(list(topology.nodes()))
    samples_1, samples_2 = draw_samples(topology, model, config, n=n, d=D)

    return Problem(
        topology=topology, model=model, cost_data=cost_data, config=config,
        samples_1=samples_1, samples_2=samples_2,
    )


def faulty_nodes(setup):
    """Nodes with node_weights < 0 -- derived, not stored, so this can
    never drift from the weights themselves.
    """
    return sorted(node for node, w in setup.node_weights.items() if w < 0)


def faulty_edges(setup):
    """Edges with edge_weights < 0 -- derived, not stored."""
    return sorted(edge for edge, w in setup.edge_weights.items() if w < 0)
