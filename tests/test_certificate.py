"""Tests for LinearConsensus.is_island_stable (island stability certificate)."""

import numpy as np

from islanding.topology import Topology
from islanding.dynamics import LinearConsensus


def test_stable_triangle_all_positive_weights():
    # Triangle 0-1-2, all-positive weights -> standard graph Laplacian,
    # which is always PSD (row sums are zero, so the all-ones vector
    # gives eigenvalue exactly 0; all other eigenvalues are positive).
    #   L = [[ 2, -1, -1],
    #        [-1,  2, -1],
    #        [-1, -1,  2]]
    # Hand verdict: PSD -> stable.
    topology = Topology(nodes=[0, 1, 2], edges=[(0, 1), (1, 2), (0, 2)])
    weights = {(0, 1): 1.0, (1, 2): 1.0, (0, 2): 1.0}
    dynamics = LinearConsensus(topology=topology, weights=weights)

    assert dynamics.is_island_stable([0, 1, 2]) == True


def test_unstable_path_with_negative_weight():
    # Path 0-1-2 (no edge 0-2), weights w01 = 2.0, w12 = -1.0.
    #   L_00 = 2.0
    #   L_11 = 2.0 + (-1.0) = 1.0
    #   L_22 = -1.0
    #   L = [[ 2, -2,  0],
    #        [-2,  1,  1],
    #        [ 0,  1, -1]]
    # Hand verdict: L_22 = -1 < 0, so e_2^T L e_2 < 0 -> L cannot be PSD
    # regardless of off-diagonal entries -> not stable.
    topology = Topology(nodes=[0, 1, 2], edges=[(0, 1), (1, 2)])
    weights = {(0, 1): 2.0, (1, 2): -1.0}
    dynamics = LinearConsensus(topology=topology, weights=weights)

    assert dynamics.is_island_stable([0, 1, 2]) == False


def test_cut_edges_are_dropped_not_sliced():
    # Nodes 0, 1 form the island; node 2 is external. Edges: intra-island
    # (0,1) with weight 1.0, and a cut edge (0,2) with weight -3.0.
    #
    # Correct island Laplacian (intra-island edges only, degrees
    # recomputed from those edges alone):
    #   L_island = [[ 1, -1],
    #               [-1,  1]]
    #   eigenvalues 0, 2 -> PSD -> stable.
    #
    # Full-topology Laplacian (for contrast only):
    #   L_00 = w01 + w02 = 1.0 + (-3.0) = -2.0
    #   L_11 = w01 = 1.0
    #   L_22 = w02 = -3.0
    #   L_01 = -1.0, L_02 = 3.0, L_12 = 0.0
    #   L_full = [[-2, -1,  3],
    #             [-1,  1,  0],
    #             [ 3,  0, -3]]
    #
    # The WRONG construction (principal submatrix of L_full on {0,1}):
    #   L_wrong = [[-2, -1],
    #              [-1,  1]]
    # has trace -1 and determinant -2*1 - (-1*-1) = -3, so its
    # eigenvalues are (-1 +/- sqrt(13)) / 2, i.e. approx 1.30 and -2.30
    # -- indefinite, NOT PSD. This differs from the correct verdict, so
    # a test using the wrong construction would report unstable when the
    # island is actually stable.
    topology = Topology(nodes=[0, 1, 2], edges=[(0, 1), (0, 2)])
    weights = {(0, 1): 1.0, (0, 2): -3.0}
    dynamics = LinearConsensus(topology=topology, weights=weights)

    # Correct behavior: cut edge (0, 2) is dropped entirely.
    assert dynamics.is_island_stable([0, 1]) == True

    # Sanity-check that the wrong (principal-submatrix) construction
    # really does disagree, i.e. this test is capable of catching the bug.
    full_laplacian = dynamics.laplacian()
    wrong_submatrix = full_laplacian[np.ix_([0, 1], [0, 1])]
    assert np.linalg.eigvalsh(wrong_submatrix).min() < -1e-9
