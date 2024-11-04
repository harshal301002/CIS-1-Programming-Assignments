import numpy as np
import PointCloud as pc
import Distortion_correction as d

def tip_in_CT_coordinates(empivot, emnav, pointer_tip, registration_transform, coeffs, q_min, q_max, q_star_min, q_star_max):
    """
    Calculates the position of the pointer tip in CT coordinates for each tracker data frame.

    :param empivot: Path to the file with EM tracking data during calibration.
    :param emnav: Path to the file with marker positions when the pointer is in an arbitrary position, relative to the EM tracker.
    :param pointer_tip: Coordinates of the pointer's tip relative to the pointer coordinate system (Output of pivot_cal.pivot).
    :param registration_transform: Transformation frame from tracker to CT coordinates (Output of find_registration_transform).
    :param coeffs: Coefficient matrix for dewarping (Output of distortion.calculate_distortion).
    :param q_min: Vector of input minima for initial correction matrix creation (Output of distortion.calculate_distortion).
    :param q_max: Vector of input maxima for initial correction matrix creation (Output of distortion.calculate_distortion).
    :param q_star_min: Vector of output minima for initial correction matrix creation (Output of distortion.calculate_distortion).
    :param q_star_max: Vector of output maxima for initial correction matrix creation (Output of distortion.calculate_distortion).

    :type empivot: str
    :type emnav: str
    :type pointer_tip: numpy.ndarray shape (3,)
    :type registration_transform: TransformationFrame.TransformationFrame
    :type coeffs: numpy.ndarray with shape (degree**3, 3)
    :type q_min: numpy.ndarray shape (3,)
    :type q_max: numpy.ndarray shape (3,)
    :type q_star_min: numpy.ndarray shape (3,)
    :type q_star_max: numpy.ndarray shape (3,)

    :return: PointCloud containing the position of the pointer tip in CT coordinates for each frame.
    :rtype: pc.PointCloud
    """

    # Correct positions in EM tracker coordinates
    corrected_nav = d.apply_correction(emnav, coeffs, q_min, q_max, q_star_min, q_star_max)
    corrected_pivot = d.apply_correction(empivot, coeffs, q_min, q_max, q_star_min, q_star_max)
    pivot_mean = np.mean(corrected_pivot[0][0].data, axis=1, keepdims=True)

    # Center the corrected pivot data around its mean
    centered_pivot_data = corrected_pivot[0][0].data - pivot_mean

    # Initialize a PointCloud for storing CT coordinates
    ct_coordinates = pc.PointCloud()

    for frame in corrected_nav:
        # Register and transform the pointer tip into CT coordinates
        frame_transform = pc.PointCloud(centered_pivot_data).register(frame[0])
        transformed_tip = pc.PointCloud(pointer_tip.reshape((3, 1))).transform(frame_transform).transform(registration_transform)
        ct_coordinates = ct_coordinates.add(transformed_tip)

    return ct_coordinates
