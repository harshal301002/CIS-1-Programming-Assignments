import numpy as np
import PointCloud as pc

def pivot(point_groups, frame_idx, debug=False):
    """
    Calculates the pivot calibration, determining the pointer tip's position in both pointer and EM tracker coordinates.
    
    :param point_groups: Nested list of PointCloud instances, each representing calibration data from different frames.
    :param frame_idx: Index to select the EM tracking data from each list.
    :param debug: If True, additional frame transformations are returned.

    :return: Tuple with positions of the pointer tip in pointer and EM tracker coordinates, and optionally frame transformations.
    """
    # Extract initial points from the selected frame and compute the centroid.
    initial_points = point_groups[0][frame_idx].data
    centroid = np.mean(initial_points, axis=0)
    adjusted_points = initial_points - centroid

    num_frames = len(point_groups)
    assembly_matrix = np.zeros((3 * num_frames, 6))
    solution_vector = np.zeros(3 * num_frames)

    frame_transformations = []

    # Construct the assembly matrix for the least squares solution.
    for i in range(num_frames):
        transformation = pc.PointCloud(adjusted_points).register(point_groups[i][frame_idx])
        if debug:
            frame_transformations.append(transformation)
        rotation_matrix, position_vector = transformation.rotation, transformation.translation
        for j in range(3):
            assembly_matrix[3 * i + j, :3] = rotation_matrix[j]
            assembly_matrix[3 * i + j, 3 + j] = -1
            solution_vector[3 * i + j] = -position_vector[j]

    # Solve the least squares problem to find the calibration points.
    calibration_solution = np.linalg.lstsq(assembly_matrix, solution_vector, rcond=None)
    pointer_tip = calibration_solution[0][:3]
    tracker_tip = calibration_solution[0][3:6]

    if debug:
        return pointer_tip, tracker_tip, frame_transformations
    return pointer_tip, tracker_tip
