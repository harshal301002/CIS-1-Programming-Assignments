import numpy as np
from distancecalc import distance_calculator_barycentric

def closest_point_simple(meshFile, dk):
    """
    Finds the closest point on a given surface mesh using a brute-force linear search.
    """
    # Read mesh data
    with open(meshFile, 'r') as fid:
        n_vert = int(fid.readline().strip())
        DV = np.array([list(map(float, fid.readline().strip().split())) for _ in range(n_vert)]).T
        n_tr = int(fid.readline().strip())
        triangles = np.array([list(map(int, fid.readline().strip().split()[:3])) for _ in range(n_tr)])

    # Initialize variables
    sk = dk
    n_frames = sk.shape[1]
    c = np.zeros((3, n_frames))
    d = np.zeros(n_frames)

    # For each frame, find the closest point
    for j in range(n_frames):
        min_dist = float('inf')
        min_c = None
        for tri in triangles:
            p, q, r = DV[:, tri[0]], DV[:, tri[1]], DV[:, tri[2]]
            dist, c_temp = distance_calculator_barycentric(p, q, r, sk[:, j])
            if dist < min_dist:
                min_dist = dist
                min_c = c_temp
        d[j] = min_dist
        c[:, j] = min_c

    return d, c
