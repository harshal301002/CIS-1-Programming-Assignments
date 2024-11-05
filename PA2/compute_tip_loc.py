import numpy as np
import PointCloud as pc
import distortion_correction as d

def tip_pointer(empivot, emnav, ptip, F_reg, coeffs, q_min, q_max, q_star_min, q_star_max):
    """
    Returns the position of the pointer tip in CT coordinates for tracker data frames.

    :param empivot: The file name/path containing EM tracking data during calibration
    :param emnav: The file name/path of the file with marker positions when the pointer is in an arbitrary position,
                  relative to the EM tracker.
    :param ptip: The coordinates of the tip of the pointer relative to the pointer coordinate system (Output of
                 pivot_cal.pivot)
    :param coeffs: Coefficient matrix for dewarping (Output of distortion.distortion_calculation)
    :param q_min: Vector of input minima for the initial correctionion matrix creation (Output of distortion.distortion_calculation)
    :param q_max: Vector of input maxima for the initial correctionion matrix creation (Output of distortion.distortion_calculation)
    :param q_star_min: Vector of output minima for the initial correctionion matrix creation (Output of distortion.distortion_calculation)
    :param q_star_max: Vector of output maxima for the initial correctionion matrix creation (Output of distortion.distortion_calculation)

    :return: A PointCloud instance with the transformed positions of the pointer tip for each frame of EM data
    """

    # correction the navigation and pivot datasets using the provided distortion coefficients
    correctioned_nav_data = d.correction(emnav, coeffs, q_min, q_max, q_star_min, q_star_max)
    correctioned_pivot_data = d.correction(empivot, coeffs, q_min, q_max, q_star_min, q_star_max)
    
    # Compute the centroid of the correctioned pivot data to use as a reference point
    reference_point = np.mean(correctioned_pivot_data[0][0].data, axis=0, keepdims=True)
    
    # normalization the pivot data by subtracting the centroid
    normalizationd_pivot = correctioned_pivot_data[0][0].data - reference_point

    # Initialize a PointCloud object to accumulate results
    accumulated_pointcloud = pc.PointCloud()

    # Process each frame of correctioned navigation data
    for nav_frame in correctioned_nav_data:
        # Register the normalizationd pivot data to the current navigation frame
        transformation = pc.PointCloud(normalizationd_pivot).register(nav_frame[0])
        
        # Transform the pointer tip using the computed transformation and additional registration
        transformed_tip = pc.PointCloud(ptip.reshape((3, 1))).transform(transformation).transform(F_reg)
        
        # Add the transformed tip to the accumulated results
        accumulated_pointcloud = accumulated_pointcloud.add(transformed_tip)

    return accumulated_pointcloud
