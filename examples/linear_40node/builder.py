"""Setup/builder module for the 40-node linear consensus example.

Reuses the shared, topology-agnostic builders from examples/common.py --
build_model, build_cost_data, build_problem, build_M, faulty_nodes, and
faulty_edges are re-exported here unchanged, not redefined. Only
`build_topology()` and the `LINEAR_40` Setup instance are specific to this
example, matching the pattern in
examples/linear_10node/linear_10node.py. This case's `AlgorithmConfig` lives
separately in examples/linear_40node/config.py.

Weights come from the generated, seeded topology artifact
(linear_40node_topology.py) -- see
examples/linear_40node/generate_linear_40node.py. This case has no spec
document of its own; it exists as a density-matched companion to the
20-, 30- and 50-node cases for runtime comparison, and it is the largest
case whose faults are placed explicitly against a measured bottleneck
rather than taken as "the first n nodes".

Importing this module defines builders and the setup only -- it never
runs a solve or any computation beyond definitions.
"""

from islanding.topology import Topology
from examples.common import (
    Setup, build_M, build_model, build_cost_data,
    build_problem, faulty_nodes, faulty_edges,
)
from examples.linear_40node.linear_40node_topology import (
    NODES, EDGES, NODE_WEIGHTS, EDGE_WEIGHTS, POSITIONS,
)

LINEAR_40 = Setup(
    name="linear_40",
    node_weights=NODE_WEIGHTS,
    edge_weights=EDGE_WEIGHTS,
    node_costs={node: 2.0 for node in NODES},
    edge_costs={edge: 1.0 for edge in EDGES},
)


def build_topology():
    return Topology(nodes=NODES, edges=EDGES)
