"""10-node linear consensus example on one shared topology, supporting
multiple weight/cost setups.

The generic `Setup` dataclass and builders (`build_config`, `build_M`,
`build_model`, `build_cost_data`, `build_problem`, `faulty_nodes`,
`faulty_edges`) live in examples/common.py and are re-exported here;
this module keeps only what's 10-node-specific: NODES, EDGES,
`build_topology()`, and the Setup instances below. `build_topology()`
stays here (rather than in common.py) since it's specific to this
example's network -- all setups on this topology share it, which is
what makes them comparable.

`EDGE_FAULTS` is the original setup: **edge faults only**. Two edges,
(1,2) and (7,10), carry strongly negative weights (-3 instead of the
default +1), so any island containing both endpoints of a faulty edge is
unstable. Both faults are removable by cutting -- no node needs shedding
-- which makes this a clean first end-to-end example: node costs (2)
exceed edge costs (1), so cutting is cheaper than shedding, matching the
fault model. Adding a new fault structure means adding another `Setup`
instance, not another file.

`NODE_AND_EDGE_FAULTS` adds a genuinely unrescuable faulty node (10) on
top of the same two faulty edges -- see its definition below for why no
island containing node 10 can be stable, regardless of partition.

10 nodes, 17 edges, d=2 per-node state (the two coordinates decouple
since all weights are scalars, so stability is governed by the same
d=1 criterion; see docs/counterexample_spec.md section 5). See
docs/example_linear_10node_spec.md for the full derivation, including
why each EDGE_FAULTS fault is removable by cutting.

This module defines topology-specific builder functions and setups
only -- importing it never runs a solve or any computation beyond
definitions.
"""

from islanding.topology import Topology
from examples.common import (
    Setup, build_config, build_M, build_model, build_cost_data,
    build_problem, faulty_nodes, faulty_edges, _canon,
)

NODES = list(range(1, 11))

EDGES = [
    (1, 2), (2, 3), (1, 4), (2, 5), (2, 6), (3, 7),
    (4, 5), (5, 6), (6, 7),
    (4, 8), (5, 8), (5, 9), (6, 9), (6, 10), (7, 10),
    (8, 9), (9, 10),
]

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

# NODE_AND_EDGE_FAULTS: same two faulty edges as EDGE_FAULTS, plus node 10
# is itself faulty (w_10 = -20.0). Node 10's neighbours are 6, 7, 9 with
# couplings 1.0, -3.0, 1.0 (summing to -1.0), so even at full coupling its
# diagonal is w_10 + sum(couplings) = -20.0 + (-1.0) = -21.0 < 0 -- no
# partition can rescue it, so it must be shed into an unstable island.
NODE_AND_EDGE_FAULTS = Setup(
    name="node_and_edge_faults",
    node_weights={**{i: _HEALTHY_NODE_WEIGHT for i in NODES}, 10: -20.0},
    edge_weights={
        _canon(edge): _FAULTY_WEIGHTS.get(_canon(edge), _HEALTHY_EDGE_WEIGHT)
        for edge in EDGES
    },
    node_costs={i: 2.0 for i in NODES},
    edge_costs={_canon(edge): 1.0 for edge in EDGES},
)


def build_topology():
    return Topology(nodes=NODES, edges=EDGES)
