"""Setup/builder module for the IEEE 30-bus case.

Reuses the shared, topology-agnostic builders from examples/common.py --
build_model, build_cost_data, build_problem, build_M, faulty_nodes, and
faulty_edges are re-exported here unchanged, not redefined. Only
`build_topology()` and the `IEEE_30` Setup instance are specific to this
example, matching examples/linear_30node/builder.py. This case's
`AlgorithmConfig` lives separately in examples/ieee_30_bus/config.py.

WHAT IS AND IS NOT FROM THE IEEE TEST CASE. The TOPOLOGY is: 30 buses and
41 branches, parsed from the IEEE Common Data Format into
ieee_30_bus_topology.py. Nothing else. The dynamics are grounded linear
consensus, the same model every other case uses, and the weights are drawn
from one seed by the same rules as the synthetic cases -- see
generate_ieee_30_bus.py. No electrical quantity from the CDF file (line
impedance, bus voltage, generation, load) reaches any number here. This
case is a real network's SHAPE under the repository's own dynamics, not a
power-flow study.

Hence the two-module import below: NODES/EDGES from the parser's output,
NODE_WEIGHTS/EDGE_WEIGHTS from our seeded generator. The synthetic cases
read all four from one artefact because one generator produced all four.

There are no POSITIONS. The IEEE case is a bus/branch list with no
coordinates, so anything geometric -- the edge-length invariant the
synthetic cases test, a plotted layout -- has no counterpart here.

Importing this module defines builders and the setup only -- it never
runs a solve or any computation beyond definitions.
"""

from islanding.topology import Topology
from examples.common import (
    Setup, build_M, build_model, build_cost_data,
    build_problem, faulty_nodes, faulty_edges,
)
from examples.ieee_30_bus.ieee_30_bus_topology import NODES, EDGES
from examples.ieee_30_bus.ieee_30_bus_weights import NODE_WEIGHTS, EDGE_WEIGHTS

IEEE_30 = Setup(
    name="ieee_30_bus",
    node_weights=NODE_WEIGHTS,
    edge_weights=EDGE_WEIGHTS,
    # The same unit costs as every other case (c_i = 2, c_ij = 1), so a
    # cross-case comparison is about topology rather than about pricing.
    node_costs={node: 2.0 for node in NODES},
    edge_costs={edge: 1.0 for edge in EDGES},
)


def build_topology():
    return Topology(nodes=NODES, edges=EDGES)
