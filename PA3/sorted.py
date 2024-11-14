import numpy as np
from scipy.spatial import KDTree
from distancecalc import distance_calculator_barycentric

def closest_point_sorted(meshFile, dk):
    """
    Finds the closest point on a given surface mesh using a KDTree for efficient searching.
    """
    # Read mesh data
    with open(meshFile, 'r') as fid:
        n_vert = int(fid.readline().strip())
        DV = np.array([list(map(float, fid.readline().strip().split())) for _ in range(n_vert)]).T
        n_tr = int(fid.readline().strip())
        triangles = np.array([list(map(int, fid.readline().strip().split()[:3])) for _ in range(n_tr)])

    # Construct KDTree
    triangle_centers = np.mean(DV[:, triangles], axis=1).T  # Calculate centroids
    kd_tree = KDTree(triangle_centers)

    # Initialize variables
    sk = dk
    n_frames = sk.shape[1]
    c = np.zeros((3, n_frames))
    d = np.zeros(n_frames)

    # For each frame, find the closest triangle and point
    for j in range(n_frames):
        _, idx = kd_tree.query(sk[:, j])  # Closest triangle center
        tri = triangles[idx]
        p, q, r = DV[:, tri[0]], DV[:, tri[1]], DV[:, tri[2]]
        d[j], c[:, j] = distance_calculator_barycentric(p, q, r, sk[:, j])

    return d, c
