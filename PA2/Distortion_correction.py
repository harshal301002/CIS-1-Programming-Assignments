import PointCloud as pc
import pivot_cal as piv
import PA2_calcExpected_prob1 as p1
import numpy as np
import scipy.misc as spmisc
import math

def calculate_distortion(calbody_file, calreadings_file, empivot_file):
    """
    Computes the coefficient matrix for distortion correction and applies it to a set of EM pivot calibration frames
    to yield the corrected pivot calibration.

    :param calbody_file: Path for the calibration object data file.
    :param calreadings_file: Path for the tracker readings file.
    :param empivot_file: Path for the EM pivot calibration poses.

    :type calbody_file: str
    :type calreadings_file: str
    :type empivot_file: str

    :return: corrected_pivot: Corrected pivot calibration data.
    :return: coeff_matrix: Matrix of coefficients for distortion correction.
    :return: min_vector: Minimum coordinate values in experimental data.
    :return: max_vector: Maximum coordinate values in experimental data.
    :return: expected_min: Minimum coordinate values in expected data.
    :return: expected_max: Maximum coordinate values in expected data.

    :rtype coeff_matrix: numpy.ndarray with shape (degree**3, 3)
    :rtype min_vector: numpy.ndarray shape (3,) or (, 3)
    :rtype max_vector: numpy.ndarray shape (3,) or (, 3)
    :rtype expected_min: numpy.ndarray shape (3,) or (, 3)
    :rtype expected_max: numpy.ndarray shape (3,) or (, 3)
    """

    tracker_frames = pc.fromfile(calreadings_file)

    measured_points = [frame[2] for frame in tracker_frames]
    expected_points = p1.compute_expected_marker_positions(calbody_file, calreadings_file)

    points_per_frame = np.shape(measured_points[0].data)[1]
    total_frames = len(measured_points)

    combined_measured = measured_points[0].data
    combined_expected = expected_points[0].data
    for i in range(1, total_frames):
        combined_measured = np.concatenate((combined_measured, measured_points[i].data), axis=1)
        combined_expected = np.concatenate((combined_expected, expected_points[i].data), axis=1)

    min_vector, max_vector, expected_min, expected_max = calculate_bounds(combined_measured, combined_expected)
    normalized_expected = normalize_data(points_per_frame * total_frames, combined_expected, expected_min, expected_max)
    normalized_measured = normalize_data(points_per_frame * total_frames, combined_measured, min_vector, max_vector)

    bernstein_matrix = compute_bernstein_matrix(normalized_measured, 5)
    coeff_matrix = solve_least_squares(bernstein_matrix, normalized_expected)

    corrected_cloud = apply_correction(empivot_file, coeff_matrix, min_vector, max_vector, expected_min, expected_max)
    corrected_pivot = piv.calculate_pivot(corrected_cloud, 0)

    return corrected_pivot, coeff_matrix, min_vector, max_vector, expected_min, expected_max

def apply_correction(input_file, coefficients, min_vector, max_vector, expected_min, expected_max):
    """
    Applies distortion correction to PointClouds loaded from a file.

    :param input_file: File path with point clouds to correct.
    :param coefficients: Matrix of coefficients for distortion correction.
    :param min_vector: Minimum coordinate values in experimental data.
    :param max_vector: Maximum coordinate values in experimental data.
    :param expected_min: Minimum coordinate values in expected data.
    :param expected_max: Maximum coordinate values in expected data.

    :type input_file: str
    :type coefficients: numpy.ndarray with shape (degree**3, 3)
    :type min_vector: numpy.ndarray shape (3,) or (, 3)
    :type max_vector: numpy.ndarray shape (3,) or (, 3)
    :type expected_min: numpy.ndarray shape (3,) or (, 3)
    :type expected_max: numpy.ndarray shape (3,) or (, 3)

    :return: corrected_clouds: Corrected point clouds.
    :rtype: list[PointCloud.PointCloud]
    """
    input_clouds = pc.fromfile(input_file)
    corrected_clouds = [cloud for cloud in input_clouds]

    points_per_cloud = np.shape(input_clouds[0][0].data)[1]

    for cloud_idx in range(len(input_clouds)):
        corrected_clouds[cloud_idx][0].data = compute_bernstein_matrix(
            normalize_data(points_per_cloud, input_clouds[cloud_idx][0].data, min_vector, max_vector), 5
        ).dot(coefficients)
        for i in range(points_per_cloud):
            for j in range(3):
                corrected_clouds[cloud_idx][0].data[i, j] = (corrected_clouds[cloud_idx][0].data[i, j]) * \
                                                           (expected_max[j] - expected_min[j]) + expected_min[j]
        corrected_clouds[cloud_idx][0].data = corrected_clouds[cloud_idx][0].data.T

    return corrected_clouds

def normalize_data(num_points, data, min_vector, max_vector):
    """
    Normalizes data within the range [0, 1].

    :param num_points: Number of points in each frame.
    :param data: Data array to be normalized.
    :param min_vector: Minimum values for each coordinate.
    :param max_vector: Maximum values for each coordinate.

    :type num_points: int
    :type data: numpy.ndarray shape (3, num_points)
    :type min_vector: numpy.ndarray shape (3,) or (, 3)
    :type max_vector: numpy.ndarray shape (3,) or (, 3)

    :return: Normalized data array.
    :rtype: numpy.ndarray shape (num_points, 3)
    """
    normalized_data = np.zeros([num_points, 3])
    for point in range(num_points):
        for coord in range(3):
            normalized_data[point][coord] = (data[coord][point] - min_vector[coord]) / (max_vector[coord] - min_vector[coord])
    return normalized_data

def solve_least_squares(F_matrix, U_matrix):
    """
    Solves for distortion coefficients using the least squares method.

    :param F_matrix: Matrix of Bernstein polynomials.
    :param U_matrix: Matrix of known corrected data values.

    :type F_matrix: numpy.ndarray shape (totalPoints, 216)
    :type U_matrix: numpy.ndarray shape (totalPoints, 3)

    :return: Distortion coefficient matrix.
    :rtype: numpy.ndarray shape (216, 3)
    """
    coefficients, _, _, _ = np.linalg.lstsq(F_matrix, U_matrix, rcond=None)
    return coefficients

def calculate_bounds(experimental_data, expected_data):
    """
    Computes minimum and maximum values for each coordinate in two datasets.

    :param experimental_data: Points from experimental data.
    :param expected_data: Points from expected (ground truth) data.

    :type experimental_data: numpy.ndarray shape (totalPoints, 3)
    :type expected_data: numpy.ndarray shape (totalPoints, 3)

    :return: Minimum and maximum vectors for both experimental and expected data.
    :rtype: tuple of numpy.ndarrays, each with shape (3,) or (, 3)
    """
    min_vector = np.min(experimental_data, axis=1)
    max_vector = np.max(experimental_data, axis=1)
    expected_min = np.min(expected_data, axis=1)
    expected_max = np.max(expected_data, axis=1)

    return min_vector, max_vector, expected_min, expected_max

def bernstein_polynomial(N, k, value):
    """
    Computes a Bernstein polynomial.

    :param N: Degree of the polynomial.
    :param k: Success count.
    :param value: Normalized data value.

    :type N: int
    :type k: int
    :type value: float

    :return: Value of the Bernstein polynomial.
    :rtype: float
    """
    return spmisc.comb(N, k, exact=True) * math.pow(1 - value, N - k) * math.pow(value, k)

def compute_bernstein_matrix(data, degree):
    """
    Generates the Bernstein polynomial matrix (F matrix) from normalized experimental data.

    :param data: Array of normalized experimental data points.
    :param degree: Degree of the Bernstein polynomial, typically 5.

    :type data: numpy.ndarray shape (totalPoints, 3)
    :type degree: int

    :return: Matrix of Bernstein polynomials.
    :rtype: numpy.ndarray shape (totalPoints, 216)
    """
    num_points = np.shape(data)[0]
    f_matrix = np.zeros([num_points, int(math.pow(degree + 1, 3))])

    for n in range(num_points):
        count = 0
        for i in range(degree + 1):
            for j in range(degree + 1):
                for k in range(degree + 1):
                    f_matrix[n][count] = bernstein_polynomial(degree, i, data[n][0]) * \
                                         bernstein_polynomial(degree, j, data[n][1]) * \
                                         bernstein_polynomial(degree, k, data[n][2])
                    count += 1

    return f_matrix
