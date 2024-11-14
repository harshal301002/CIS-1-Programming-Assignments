import numpy as np
from projectiononseg import projection_on_segment

def distance_calculator_barycentric(p, q, r, a):
    """
    Calculates the closest point in or on a triangle with vertices 'p', 'q', 'r',
    from a point in space 'a' using barycentric coordinates.

    :param p: Vertex 1 of the triangle (3,).
    :param q: Vertex 2 of the triangle (3,).
    :param r: Vertex 3 of the triangle (3,).
    :param a: Query point (3,).
    :return: Tuple (distance, closest_point).
    """
    # Triangle edge vectors
    pq = q - p
    pr = r - p
    pa = a - p

    # Compute dot products
    d00 = np.dot(pq, pq)
    d01 = np.dot(pq, pr)
    d11 = np.dot(pr, pr)
    d20 = np.dot(pa, pq)
    d21 = np.dot(pa, pr)

    # Compute barycentric coordinates
    denom = d00 * d11 - d01 * d01
    if denom == 0:
        # Degenerate triangle case, return one of the vertices
        closest_point = p
        distance = np.linalg.norm(a - p)
        return distance, closest_point

    u = (d11 * d20 - d01 * d21) / denom
    v = (d00 * d21 - d01 * d20) / denom
    w = 1 - u - v

    # Closest point inside the triangle
    if u >= 0 and v >= 0 and w >= 0:
        closest_point = u * q + v * r + w * p
    else:
        # Closest point on an edge or vertex
        edge_candidates = [
            projection_on_segment(a, p, q),
            projection_on_segment(a, q, r),
            projection_on_segment(a, r, p)
        ]
        distances = [np.linalg.norm(a - c) for c in edge_candidates]
        closest_point = edge_candidates[np.argmin(distances)]

    # Compute the distance
    distance = np.linalg.norm(a - closest_point)

    return distance, closest_point
