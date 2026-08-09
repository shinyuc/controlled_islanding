"""Setup/builder module for the 50-node linear consensus example.

Reuses the shared, topology-agnostic builders from examples/common.py --
build_model, build_cost_data, build_problem, build_config, build_M,
faulty_nodes, and faulty_edges are re-exported here unchanged, not
redefined. Only `build_topology()` and the `LINEAR_50` Setup instance are
specific to this example, matching the pattern in examples/linear_10node.py.

Weights come from the generated, seeded topology artifact
(linear_50node_topology.py) -- see docs/example_linear_50node_spec.md and
examples/linear_50node/generate_linear_50node.py.

Importing this module defines builders and the setup only -- it never
runs a solve or any computation beyond definitions.
"""

from islanding.topology import Topology
from examples.common import (
    Setup, build_config, build_M, build_model, build_cost_data,
    build_problem, faulty_nodes, faulty_edges,
)
from examples.linear_50node.linear_50node_topology import (
    NODES, EDGES, NODE_WEIGHTS, EDGE_WEIGHTS, POSITIONS,
)

LINEAR_50 = Setup(
    name="linear_50",
    node_weights=NODE_WEIGHTS,
    edge_weights=EDGE_WEIGHTS,
    node_costs={node: 2.0 for node in NODES},
    edge_costs={edge: 1.0 for edge in EDGES},
)


def build_topology():
    return Topology(nodes=NODES, edges=EDGES)
