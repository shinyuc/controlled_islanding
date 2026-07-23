"""10-node linear consensus example on one shared topology, supporting
multiple weight/cost setups.

A `Setup` bundles everything that varies per problem instance (node/edge
weights, node/edge costs); `build_topology()` and `build_config()` stay
shared and argument-free, since all setups study the same network under
the same algorithm parameters -- holding those fixed is what makes
setups comparable.

`EDGE_FAULTS` is the original setup: **edge faults only**. Two edges,
(1,2) and (7,10), carry strongly negative weights (-3 instead of the
default +1), so any island containing both endpoints of a faulty edge is
unstable. Both faults are removable by cutting -- no node needs shedding
-- which makes this a clean first end-to-end example: node costs (2)
exceed edge costs (1), so cutting is cheaper than shedding, matching the
fault model. Adding a new fault structure means adding another `Setup`
instance, not another file.

10 nodes, 17 edges, d=2 per-node state (the two coordinates decouple
since all weights are scalars, so stability is governed by the same
d=1 criterion; see docs/counterexample_spec.md section 5). See
docs/example_linear_10node_spec.md for the full derivation, including
why each EDGE_FAULTS fault is removable by cutting.

This module defines builder functions and setups only -- importing it
never runs a solve or any computation beyond definitions.
"""

from dataclasses import dataclass

import numpy as np

from islanding.topology import Topology
from islanding.cost import CostData, AlgorithmConfig, linear_consensus_model
from islanding.evaluate import Problem, draw_samples

NODES = list(range(1, 11))

EDGES = [
    (1, 2), (2, 3), (1, 4), (2, 5), (2, 6), (3, 7),
    (4, 5), (5, 6), (6, 7),
    (4, 8), (5, 8), (5, 9), (6, 9), (6, 10), (7, 10),
    (8, 9), (9, 10),
]


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


_FAULTY_WEIGHTS = {(1, 2): -3.0, (7, 10): -3.0}
_HEALTHY_NODE_WEIGHT = 1.0  # grounding strength w_i
_HEALTHY_EDGE_WEIGHT = 1.0  # coupling strength w_ij for non-faulty edges

EDGE_FAULTS = Setup(
    name="edge_faults",
    node_weights={i: _HEALTHY_NODE_WEIGHT for i in NODES},
    edge_weights={
        _canon(edge): _FAULTY_WEIGHTS.get(_canon(edge), _HEALTHY_EDGE_WEIGHT)
        for edge in EDGES
    },
    node_costs={i: 2.0 for i in NODES},
    edge_costs={_canon(edge): 1.0 for edge in EDGES},
)


def build_topology():
    return Topology(nodes=NODES, edges=EDGES)


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


def build_problem(setup, seed=None):
    topology = build_topology()
    model = build_model(setup)
    cost_data = build_cost_data(setup)
    config = build_config()

    if seed is not None:
        np.random.seed(seed)

    samples_1, samples_2 = draw_samples(topology, model, config, n=len(NODES), d=2)

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
