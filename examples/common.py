"""Generic example-builder logic, shared across examples.

`Setup` bundles everything that varies per problem instance (node/edge
weights, node/edge costs). The builders here depend only on a `Setup`
or on explicit weights/topology passed in -- never on a specific
example's topology -- so they're reusable across examples. Each example
module supplies its own topology-specific `build_topology()` and NODES/
EDGES, and passes the topology in wherever one is needed (`build_M`,
`build_problem`).

Algorithm parameters are deliberately NOT here: `AlgorithmConfig` choices
(alpha, k, P, ...) differ substantially per case, so each case owns its own
`build_config()` in its own config module -- examples/linear_10node/config.py,
examples/linear_30node/config.py. `build_problem` takes the resulting config
as a required argument.

Importing this module defines builders only -- it never runs a solve or
any computation beyond definitions.
"""

from dataclasses import dataclass

import numpy as np

from islanding.cost import CostData, linear_consensus_model
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


def build_problem(setup, topology, config, seed=None):
    """Build a Problem for `setup` on `topology`, drawing both sample sets.

    `config` is required -- parameter choices differ substantially per case,
    so there is no shared baseline to fall back to. Pass the one from the
    case's own config module (examples/linear_10node/config.py,
    examples/linear_30node/config.py).

    It has to be supplied *here* rather than swapped onto the returned
    Problem afterwards, because the draw below reads beta/gamma/P_1/P_2 --
    a later swap would desync the sample arrays from P_1/P_2, and f5/f6
    raise on that mismatch.
    """
    model = build_model(setup)
    cost_data = build_cost_data(setup)

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
