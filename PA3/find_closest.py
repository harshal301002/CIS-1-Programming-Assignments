import numpy as np

def find_closest_point_on_triangle(p, q, r, a):
    """
    Computes the closest point on the triangle defined by vertices p, q, r to point a.
    """
    # Compute vectors
    ab = q - p
    ac = r - p
    ap = a - p

    # Compute dot products
    d1 = np.dot(ab, ap)
    d2 = np.dot(ac, ap)
    d3 = np.dot(ab, ab)
    d4 = np.dot(ab, ac)
    d5 = np.dot(ac, ac)

    # Compute barycentric coordinates
    denom = d3 * d5 - d4 * d4
    if denom == 0:
        # Degenerate triangle
        # Return the closest vertex
        distances = [np.linalg.norm(a - p), np.linalg.norm(a - q), np.linalg.norm(a - r)]
        min_index = np.argmin(distances)
        return [p, q, r][min_index]

    v = (d5 * d1 - d4 * d2) / denom
    w = (d3 * d2 - d4 * d1) / denom
    u = 1 - v - w

    # Check if point is inside the triangle
    if u >= 0 and v >= 0 and w >= 0:
        # Closest point is inside the triangle
        return p + v * ab + w * ac
    else:
        # Closest point is on the triangle's edge or vertex
        return closest_point_on_triangle_edges(p, q, r, a)

def closest_point_on_triangle_edges(p, q, r, a):
    """
    Computes the closest point on the edges of the triangle to point a.
    """
    # Closest point on edge pq
    cp_pq = closest_point_on_segment(p, q, a)
    # Closest point on edge qr
    cp_qr = closest_point_on_segment(q, r, a)
    # Closest point on edge rp
    cp_rp = closest_point_on_segment(r, p, a)

    # Compute distances
    dist_pq = np.linalg.norm(a - cp_pq)
    dist_qr = np.linalg.norm(a - cp_qr)
    dist_rp = np.linalg.norm(a - cp_rp)

    # Find the closest point
    min_dist = min(dist_pq, dist_qr, dist_rp)
    if min_dist == dist_pq:
        return cp_pq
    elif min_dist == dist_qr:
        return cp_qr
    else:
        return cp_rp

def closest_point_on_segment(p1, p2, a):
    """
    Computes the closest point on the line segment between p1 and p2 to point a.
    """
    d = p2 - p1
    l2 = np.dot(d, d)
    if l2 == 0:
        # p1 and p2 are the same point
        return p1
    t = np.dot(a - p1, d) / l2
    t = np.clip(t, 0, 1)
    return p1 + t * d
