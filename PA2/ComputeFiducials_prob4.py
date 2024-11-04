import numpy as np
import PointCloud as pc
import Distortion_correction as d

def tip_in_EM(empivot, emfiducialss, pointer_tip, coeffs, q_min, q_max, q_star_min, q_star_max):
    """
    Determines the position of the pointer tip in EM coordinates for each tracker data frame when the tip is on a fiducial pin.

    :param empivot: Path to the file with EM tracking data during calibration.
    :param emfiducialss: Path to the file with marker positions when the pointer is on the fiducials, relative to the EM tracker.
    :param pointer_tip: Coordinates of the pointer's tip relative to the pointer coordinate system (Output of pivot_cal.pivot).
    :param coeffs: Coefficient matrix for dewarping (Output of distortion.calculate_distortion).
    :param q_min: Vector of input minima for initial correction matrix creation (Output of distortion.calculate_distortion).
    :param q_max: Vector of input maxima for initial correction matrix creation (Output of distortion.calculate_distortion).
    :param q_star_min: Vector of output minima for initial correction matrix creation (Output of distortion.calculate_distortion).
    :param q_star_max: Vector of output maxima for initial correction matrix creation (Output of distortion.calculate_distortion).

    :type empivot: str
    :type emfiducialss: str
    :type pointer_tip: numpy.ndarray of shape (3,)
    :type coeffs: numpy.ndarray of shape (degree**3, 3)
    :type q_min: numpy.ndarray shape (3,)
    :type q_max: numpy.ndarray shape (3,)
    :type q_star_min: numpy.ndarray shape (3,)
    :type q_star_max: numpy.ndarray shape (3,)

    :return: A PointCloud representing the location of the pointer tip in EM tracker coordinates for each observation set.
    :rtype: pc.PointCloud
    """

    G_corrected = d.apply_correction(emfiducialss, coeffs, q_min, q_max, q_star_min, q_star_max)
    G_original = d.apply_correction(empivot, coeffs, q_min, q_max, q_star_min, q_star_max)
    G_0 = np.mean(G_original[0][0].data, axis=1, keepdims=True)

    # Centered data around the mean
    G_j = G_original[0][0].data - G_0

    # Initialize an empty PointCloud to collect the results
    tip_positions = pc.PointCloud()

    for frame in G_corrected:
        # Perform registration and transform the pointer tip position
        frame_transform = pc.PointCloud(G_j).register(frame[0])
        transformed_tip = pc.PointCloud(pointer_tip.reshape((3, 1))).transform(frame_transform)
        tip_positions = tip_positions.add(transformed_tip)

    return tip_positions
