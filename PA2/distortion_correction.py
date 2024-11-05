import PointCloud as pc
import pivot_calibration as piv
import calc_expected_Ci as p1
import numpy as np
from scipy.special import comb
import math

def distortion_calculation(calbody, calreading, empivot):
    """
    Generates and applies a correctionion matrix for distortion in EM pivot calibration data.

    :param calbody: Path to the file containing calibration object details
    :param calreading: Path to the file with tracker data
    :param empivot: Path to the file with EM pivot pose data

    :return: pivotanswer: Adjusted pivot calibration
    :return: coeff_mat: Distortion correctionion coefficients matrix
    :return: q_min, q_max: Min and max values per axis in experimental data
    :return: q_star_min, q_star_max: Min and max values per axis in reference data
    """
    
    # Load tracker frames from the readings file
    tracker_frames = pc.inp_file(calreading)
    
    # Extract third element of each tracker frame for calibration correctionion
    c = [frame[2] for frame in tracker_frames]
    
    # Get calculated expected calibration points
    c_exp = p1.c_expected(calbody, calreading)
    
    # Define points per frame and total frame count
    pPerFrame = np.shape(c[0].data)[1]    
    nFrames = len(c)
    
    # Initialize concatenated data arrays for both current and expected calibration data
    concatc = c[0].data
    concatc_exp = c_exp[0].data
    for i in range(1, nFrames):
        # Concatenate data across all frames for streamlined calculations
        concatc = np.concatenate((concatc, c[i].data), axis=1)
        concatc_exp = np.concatenate((concatc_exp, c_exp[i].data), axis=1)
    
    # Identify min and max ranges for coordinates in experimental and expected datasets
    q_min, q_max, q_star_min, q_star_max = calc_q(concatc, concatc_exp)
    
    # normalization data points to fall within 0-1 range, based on reference dataset bounds
    u_s_star = normalization(pPerFrame * nFrames, concatc_exp, q_star_min, q_star_max)
    u_s = normalization(pPerFrame * nFrames, concatc, q_min, q_max)
    
    # Construct the matrix of calc_berstein polynomials for scaled experimental data
    F_mat = normalized_matrix(u_s, 5)
    
    # Solve for the distortion correctionion coefficients using least squares
    coeff_mat = solve_linear_sys(F_mat, u_s_star)
    
    # Apply correctionion to the EM pivot positions, using the calculated coefficients
    EMcorrection = correction(empivot, coeff_mat, q_min, q_max, q_star_min, q_star_max)
    
    # Generate the final correctioned pivot calibration
    pivotanswer = piv.pivot(EMcorrection, 0)
    
    return pivotanswer, coeff_mat, q_min, q_max, q_star_min, q_star_max


def correction(inputs, coeffs, q_min, q_max, q_star_min, q_star_max):
    """
    Applies distortion correctionion to a set of input point clouds.

    :param inputs: Filename for the input point cloud data
    :param coeffs: Matrix containing distortion correctionion coefficients
    :param q_min: Minimum coordinate values in the original data
    :param q_max: Maximum coordinate values in the original data
    :param q_star_min: Minimum coordinate values in the reference data
    :param q_star_max: Maximum coordinate values in the reference data

    :return: outputcloud: Adjusted point clouds after distortion correctionion
    """
    
    # Retrieve input point cloud data
    inputcloud = pc.inp_file(inputs)
    outputcloud = []

    # Copy original input clouds to output list
    for p in range(len(inputcloud)):
        outputcloud.append(inputcloud[p])
    
    # Get number of points per frame
    points = np.shape(inputcloud[0][0].data)[1]
    
    # Adjust each input point cloud using the distortion correctionion matrix
    for k in range(len(inputcloud)):
        outputcloud[k][0].data = normalized_matrix(normalization(points, inputcloud[k][0].data, q_min, q_max), 5).dot(coeffs)
        for i in range(points):
            for j in range(3):
                # Scale correctioned points back to original range based on reference dataset bounds
                outputcloud[k][0].data[i, j] = (outputcloud[k][0].data[i, j]) * (q_star_max[j] - q_star_min[j]) + q_star_min[j]

        # Transpose data for consistency in format
        outputcloud[k][0].data = outputcloud[k][0].data.T

    return outputcloud


def normalization(pPerFrame, c, q_min, q_max):
    """
    Scales data to a normalizationd range between 0 and 1.

    :param pPerFrame: Number of points per data frame
    :param c: Data matrix to be scaled
    :param q_min: Minimum values per axis in the original dataset
    :param q_max: Maximum values per axis in the original dataset

    :return: u_s: normalizationd data matrix
    """
    
    # Initialize normalizationd data array
    u_s = np.zeros([pPerFrame, 3])

    # Scale each data point between q_min and q_max for uniformity
    for k in range(pPerFrame):
        for i in range(3):
            u_s[k][i] = (c[i][k] - q_min[i]) / (q_max[i] - q_min[i])

    return u_s


def solve_linear_sys(F, U):
    """
    Computes correctionion coefficients by solving an overdetermined linear system.

    :param F: Matrix of calc_berstein polynomial values
    :param U: Known correctioned data values

    :return: C: Matrix of computed distortion coefficients
    """
    
    # Solve using least-squares to estimate the coefficient matrix for data correctionion
    C = np.linalg.lstsq(F, U)

    return C[0]


def calc_q(c, c_exp):
    """
    Determines the range (min and max) of each axis in experimental and reference datasets.

    :param c: Experimental data points array
    :param c_exp: Reference ("ground truth") data points array

    :return: q_min, q_max: Min and max coordinates in experimental data
    :return: q_star_min, q_star_max: Min and max coordinates in reference data
    """
    
    # Initialize min and max vectors for both datasets
    q_min = np.zeros(3)
    q_max = np.zeros(3)
    q_star_min = np.zeros(3)
    q_star_max = np.zeros(3)

    # Calculate bounds for each coordinate axis
    for i in range(3):
        q_min[i] = min(c[i])
        q_max[i] = max(c[i])
        q_star_min[i] = min(c_exp[i])
        q_star_max[i] = max(c_exp[i])

    return q_min, q_max, q_star_min, q_star_max


def calc_berstein(N, k, u):
    """
    Computes the value of a calc_berstein polynomial at a given point.

    :param N: Polynomial degree
    :param k: "Successes" or coefficient order
    :param u: normalizationd data point

    :return: B: calc_berstein polynomial evaluated at u
    """
    B = comb(N, k, exact=True) * math.pow(1 - u, N - k) * math.pow(u, k)
    return B


def f_ijk(N, i, j, k, u_x, u_y, u_z):
    """
    Computes a matrix entry based on calc_berstein polynomials in three dimensions.

    :param N: Polynomial degree
    :param i, j, k: Successes for x, y, and z coordinates, respectively
    :param u_x, u_y, u_z: normalizationd coordinates

    :return: calc_berstein product for the matrix entry
    """
    return calc_berstein(N, i, u_x) * calc_berstein(N, j, u_y) * calc_berstein(N, k, u_z)


def normalized_matrix(u, deg):
    """
    Generates the matrix of calc_berstein polynomial terms for normalizationd data.

    :param u: normalizationd experimental data points
    :param deg: Polynomial degree

    :return: f_mat: Matrix of polynomial values for data distortion correctionion
    """
    
    # Total points in normalizationd dataset
    nPoints = np.shape(u)[0]

    # Initialize matrix to hold polynomial values
    f_mat = np.zeros([nPoints, int(math.pow(deg + 1, 3))])

    # Populate the matrix with polynomial terms for each data point
    for n in range(nPoints):
        c = 0
        for i in range(deg + 1):
            for j in range(deg + 1):
                for k in range(deg + 1):
                    f_mat[n][c] = f_ijk(deg, i, j, k, u[n][0], u[n][1], u[n][2]) ; c += 1

    return f_mat
