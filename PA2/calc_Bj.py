import numpy as np
import PointCloud as pc
import distortion_correction as d

def tip_in_EM(empivot_path, emfiducials_path, pointer_tip, deformation_coeffs, min_input, max_input, min_output, max_output):
    """
    Computes the position of the pointer tip in EM coordinates using calibration and fiducial data, correctioned for distortions.

    :param empivot_path: File path for EM pivot calibration data
    :param emfiducials_path: File path for EM fiducial marker data
    :param pointer_tip: Pointer tip location in its coordinate system
    :param deformation_coeffs: Coefficients for spatial distortion correctionion
    :param min_input: Minimum input bounds for distortion correctionion
    :param max_input: Maximum input bounds for distortion correctionion
    :param min_output: Minimum output bounds after distortion correctionion
    :param max_output: Maximum output bounds after distortion correctionion

    :return: A PointCloud instance representing the pointer tip location in EM tracker coordinates
    :rtype: PointCloud.PointCloud
    """
    # correction fiducials and pivot data for distortions
    fiducial_data = d.correction(emfiducials_path, deformation_coeffs, min_input, max_input, min_output, max_output)
    pivot_data = d.correction(empivot_path, deformation_coeffs, min_input, max_input, min_output, max_output)

    # Calculate the mean of the original pivot data for normalization
    pivot_mean = np.mean(pivot_data[0][0].data, axis=0, keepdims=True)
    normalizationd_pivot_data = pivot_data[0][0].data - pivot_mean

    # Prepare to accumulate transformed point clouds
    tip_locations = pc.PointCloud()

    # Transform the pointer tip location using each frame's registration
    for fiducial_frame in fiducial_data:
        registration = pc.PointCloud(normalizationd_pivot_data).register(fiducial_frame[0])
        transformed_tip = pc.PointCloud(pointer_tip.reshape((3, 1))).transform(registration)
        tip_locations = tip_locations.add(transformed_tip)

    return tip_locations
