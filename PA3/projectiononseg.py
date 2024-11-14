import numpy as np

def projection_on_segment(c, p, q):
    """
    Finds the closest point on a line segment defined by points 'p' and 'q' to a point 'c'.
    """
    l = np.dot(c - p, q - p) / np.dot(q - p, q - p)
    l_star = max(0, min(l, 1))
    c_star = p + l_star * (q - p)
    return c_star
