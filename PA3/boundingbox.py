import numpy as np
def compute_bounding_boxes(DV, index):
    """
    Computes the axis-aligned bounding boxes for each triangle.
    """
    n_tr = index.shape[1]
    bounding_boxes = []
    for i in range(n_tr):
        p = DV[:, index[0, i]]
        q = DV[:, index[1, i]]
        r = DV[:, index[2, i]]
        min_coords = np.minimum(np.minimum(p, q), r)
        max_coords = np.maximum(np.maximum(p, q), r)
        bounding_boxes.append((min_coords, max_coords))
    return bounding_boxes
