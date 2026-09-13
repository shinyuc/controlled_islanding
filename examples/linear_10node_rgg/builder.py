"""Setup/builder module for the 10-node random-geometric-graph example.

Reuses the shared, topology-agnostic builders from examples/common.py --
build_model, build_cost_data, build_problem, build_M, faulty_nodes, and
faulty_edges are re-exported here unchanged, not redefined. Only
`build_topology()` and the `LINEAR_10_RGG` Setup instance are specific to
this example, matching examples/linear_20node/builder.py. This case's
`AlgorithmConfig` lives separately in
examples/linear_10node_rgg/config.py.

DISTINCT FROM examples/linear_10node/, which is the same SIZE but a
hand-specified topology. This one is drawn by the same random geometric
rule as the 20-, 30- and 40-node cases, which is the whole reason it
exists: a scaling table wants its smallest entry generated the same way
as the rest. Both are kept, and neither replaces the other in the tests
or specs written against them.

Weights come from the generated, seeded artefact
(linear_10node_rgg_topology.py) -- see
examples/linear_10node_rgg/generate_linear_10node_rgg.py.

Importing this module defines builders and the setup only -- it never
runs a solve or any computation beyond definitions.
"""

from islanding.topology import Topology
from examples.common import (
    Setup, build_M, build_model, build_cost_data,
    build_problem, faulty_nodes, faulty_edges,
)
from examples.linear_10node_rgg.linear_10node_rgg_topology import (
    NODES, EDGES, NODE_WEIGHTS, EDGE_WEIGHTS, POSITIONS,
)

LINEAR_10_RGG = Setup(
    name="linear_10_rgg",
    node_weights=NODE_WEIGHTS,
    edge_weights=EDGE_WEIGHTS,
    # The same unit costs as every other case (c_i = 2, c_ij = 1), so a
    # cross-case comparison is about topology rather than about pricing.
    node_costs={node: 2.0 for node in NODES},
    edge_costs={edge: 1.0 for edge in EDGES},
)


def build_topology():
    return Topology(nodes=NODES, edges=EDGES)
