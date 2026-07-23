"""10-node linear consensus example with two faulty edges.

Fault model: **edge faults only**. Two edges, (1,2) and (7,10), carry
strongly negative weights (-3 instead of the default +1), so any island
containing both endpoints of a faulty edge is unstable. Both faults are
removable by cutting -- no node needs shedding -- which makes this a
clean first end-to-end example: node costs (2) exceed edge costs (1), so
cutting is cheaper than shedding, matching the fault model.

10 nodes, 17 edges, d=2 per-node state (the two coordinates decouple
since all weights are scalars, so stability is governed by the same
d=1 criterion; see docs/counterexample_spec.md section 5). See
docs/example_linear_10node_spec.md for the full derivation, including
why each fault is removable by cutting.

This module defines builder functions only -- importing it never runs a
solve or any computation beyond function definitions.
"""

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

FAULTY_WEIGHTS = {(1, 2): -3.0, (7, 10): -3.0}


def _canon(edge):
    i, j = edge
    return (i, j) if i < j else (j, i)


def build_topology():
    return Topology(nodes=NODES, edges=EDGES)


def build_model():
    node_weights = {i: 1.0 for i in NODES}
    edge_weights = {
        _canon(edge): FAULTY_WEIGHTS.get(_canon(edge), 1.0) for edge in EDGES
    }
    return linear_consensus_model(node_weights, edge_weights)


def build_cost_data():
    node_costs = {i: 2 for i in NODES}
    edge_costs = {_canon(edge): 1 for edge in EDGES}
    return CostData(node_costs=node_costs, edge_costs=edge_costs)


def build_config():
    return AlgorithmConfig(
        kappa_op=1.0, kappa_rec=0.7,
        alpha_1=1.0, beta=1.0, k_1=0.2, P_1=20,
        alpha_2=1.0, gamma=1.0, k_2=0.2, P_2=20,
        epsilon=0.01, max_iterations=100, swap_supernode_edges=True,
    )


def build_problem(seed=None):
    topology = build_topology()
    model = build_model()
    cost_data = build_cost_data()
    config = build_config()

    if seed is not None:
        np.random.seed(seed)

    samples_1, samples_2 = draw_samples(topology, model, config, n=len(NODES), d=2)

    return Problem(
        topology=topology, model=model, cost_data=cost_data, config=config,
        samples_1=samples_1, samples_2=samples_2,
    )
