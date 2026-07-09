"""Structural checks for LinearConsensus's signed Laplacian."""

import numpy as np

from islanding.topology import Topology
from islanding.dynamics import LinearConsensus


def test_laplacian_entries_hand_computed():
    # Triangle 0-1-2 with hand-picked signed weights.
    # Gamma_01 = 2.0, Gamma_12 = -3.0, Gamma_02 = 1.5
    #
    # L_ii = sum of incident Gamma_ij:
    #   L_00 = Gamma_01 + Gamma_02 = 2.0 + 1.5   = 3.5
    #   L_11 = Gamma_01 + Gamma_12 = 2.0 + (-3.0) = -1.0
    #   L_22 = Gamma_12 + Gamma_02 = -3.0 + 1.5   = -1.5
    #
    # L_ij = -Gamma_ij for i != j:
    #   L_01 = -2.0, L_02 = -1.5, L_12 = 3.0
    topology = Topology(nodes=[0, 1, 2], edges=[(0, 1), (1, 2), (0, 2)])
    weights = {(0, 1): 2.0, (1, 2): -3.0, (0, 2): 1.5}
    dynamics = LinearConsensus(topology=topology, weights=weights)

    expected = np.array([
        [3.5, -2.0, -1.5],
        [-2.0, -1.0, 3.0],
        [-1.5, 3.0, -1.5],
    ])

    np.testing.assert_allclose(dynamics.laplacian(), expected)


def test_laplacian_symmetric(toy10_dynamics):
    L = toy10_dynamics.laplacian()
    np.testing.assert_allclose(L, L.T)


def test_laplacian_not_psd(toy10_dynamics):
    L = toy10_dynamics.laplacian()
    smallest_eigenvalue = np.linalg.eigvalsh(L).min()
    assert smallest_eigenvalue < -1e-9
