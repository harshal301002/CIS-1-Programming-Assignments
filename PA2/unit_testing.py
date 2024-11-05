import numpy as np
import scipy.linalg as lin_alg
import PointCloud as pc
import pivot_cal as pivot
import distortion_correction as distort


def test_register(tolerance=1e-4):
    """
    Tests the registration of a point cloud using random rotation and translation.
    Registration is verified if the calculated transformation aligns within the specified tolerance.

    :param tolerance: Allowed deviation between generated and calculated transformations
    :type tolerance: float

    :return: None
    """
    print('Running point cloud registration test...')
    angles = np.random.uniform(0, 2 * np.pi, 3)
    original_points = np.random.uniform(0, 10, (3, 10))
    print('\nInitial points:\n', original_points)

    rotation_matrix = generate_rotation_matrix(angles)
    print('\nRotation matrix:\n', rotation_matrix)

    translation_vector = np.random.uniform(0, 10, (3, 1))
    print('\nTranslation vector:\n', translation_vector)

    print('\nTolerance level:', tolerance)
    transformed_points = rotation_matrix.dot(original_points) + translation_vector
    print('\nTransformed points:\n', transformed_points)

    print('\nComputing transformation matrix...')
    transformation = pc.PointCloud(original_points).register(pc.PointCloud(transformed_points))
    print('Transformation complete!')

    print('\nComputed rotation:\n', transformation.r)
    print('\nComputed translation:\n', transformation.p)

    print('\nVerifying rotation accuracy...')
    assert np.all(np.abs(rotation_matrix - transformation.r) <= tolerance)

    print('\nChecking rotation determinant (should be 1)...')
    assert np.abs(lin_alg.det(transformation.r) - lin_alg.det(rotation_matrix)) <= tolerance

    print('\nVerifying translation accuracy...')
    assert np.all(np.abs(translation_vector - transformation.p) <= tolerance)

    print('\nRegistration test passed!')


def test_pivot_cal(empivot, tolerance=1e-2):
    """
    Validates pivot calibration by verifying transformation consistency.

    :param empivot: Path to the empivot.txt file with EM tracker data
    :param tolerance: Allowed error margin between expected and calculated values
    :type empivot: str
    :type tolerance: float

    :return: None
    """
    print('\nRunning pivot calibration test...')
    marker_frames = pc.fromfile(empivot)

    print('\nExtracted marker points:')
    for i, frame in enumerate(marker_frames):
        print(f'\nFrame {i + 1}:\n', frame[0].data)

    calibration_results = pivot.pivot(marker_frames, 0, True)
    print('\nCalculated tip position:\n', calibration_results[0])
    print('\nCalculated dimple position:\n', calibration_results[1])

    print('\nVerifying frame transformations for pivot calibration...')
    for i, frame in enumerate(marker_frames):
        tip_dimple_aligned = np.all(np.abs(
            calibration_results[1].reshape((3, 1)) - 
            pc.PointCloud(calibration_results[0].reshape((3, 1))).transform(calibration_results[2][i]).data
        ) <= tolerance)
        assert tip_dimple_aligned
        print(f'\nFrame {i + 1}/{len(marker_frames)}: Alignment Verified')

    print('\nPivot calibration test passed!')


def test_normalize():
    """
    Validates normalization of random data between 0 and 1.
    """
    print('\nRunning normalization test with random data...')
    random_data = np.random.uniform(-100, 100, (3, 10))
    print('\nRandom data:\n', random_data)

    calculated_qs = distort.calc_q(random_data, random_data)
    normalized_data = distort.normalize(10, random_data, calculated_qs[0], calculated_qs[1])

    print('\nNormalized data:\n', normalized_data)
    within_bounds = np.all(normalized_data >= 0) and np.all(normalized_data <= 1)
    assert within_bounds
    print('\nNormalization test passed!')


def test_f():
    """
    Validates the Bernstein polynomial matrix calculation for distortion correction.
    """
    print('Running Bernstein polynomial matrix test for distortion correction...')
    ones_matrix = np.ones((3, 10))
    print('\nInput matrix (ones):\n', ones_matrix)

    poly_matrix = distort.f_matrix(ones_matrix, 5)
    expected_shape = (3, 6 ** 3)
    print('\nExpected shape:', expected_shape)
    print('Computed shape:', poly_matrix.shape)
    assert expected_shape == poly_matrix.shape

    print('\nChecking last column values (should be 1s):')
    assert np.all(poly_matrix[:, -1] == 1)

    print('\nChecking other columns (should be 0s):')
    assert np.all(poly_matrix[:, :-1] == 0)

    print('\nBernstein polynomial matrix test passed!')


def test_solve_fcu(tolerance=1e-4):
    """
    Tests distortion correction using random data.

    :param tolerance: Allowed error tolerance for corrected data
    :type tolerance: float

    :return: None
    """
    print('\nRunning distortion correction test...')
    observed_data = np.random.uniform(-100, 100, (3, 10))
    true_data = observed_data + np.random.uniform(-2, 2, (3, 10))

    observed_normalized, true_normalized = normalize_data(observed_data, true_data)

    print('\nGenerating Bernstein polynomial matrix...')
    polynomial_matrix = distort.f_matrix(observed_normalized, 5)

    print('\nSolving for coefficient matrix...')
    coefficients = distort.solve_fcu(polynomial_matrix, true_normalized)
    distortion_corrected = np.all(np.abs(polynomial_matrix.dot(coefficients) - true_normalized) <= tolerance)
    assert distortion_corrected
    print('\nDistortion correction test passed!')


def generate_rotation_matrix(angles):
    """
    Helper function to generate a 3D rotation matrix.

    :param angles: Rotation angles for x, y, and z axes
    :type angles: numpy.array
    :return: 3x3 rotation matrix
    """
    theta, phi, gamma = angles
    rx = np.array([[1, 0, 0], [0, np.cos(theta), -np.sin(theta)], [0, np.sin(theta), np.cos(theta)]])
    ry = np.array([[np.cos(phi), 0, np.sin(phi)], [0, 1, 0], [-np.sin(phi), 0, np.cos(phi)]])
    rz = np.array([[np.cos(gamma), -np.sin(gamma), 0], [np.sin(gamma), np.cos(gamma), 0], [0, 0, 1]])

    return rx @ ry @ rz


def normalize_data(observed, ground_truth):
    """
    Normalizes observed and ground truth data using calculated `q` values.

    :param observed: Observed data
    :param ground_truth: Ground truth data
    :return: Normalized observed and ground truth data
    """
    q_obs, q_true = distort.calc_q(observed, ground_truth)
    return distort.normalize(10, observed, q_obs[0], q_obs[1]), distort.normalize(10, ground_truth, q_true[2], q_true[3])
