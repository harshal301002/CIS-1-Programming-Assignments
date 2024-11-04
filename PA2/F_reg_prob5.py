import PointCloud as pc

def find_registration_transform(ctfiducials, computed_points):
    """
    Computes the transformation from EM tracker coordinates to CT frame coordinates.

    :param ctfiducials: Path to the file with fiducial pin positions in the CT frame.
    :param computed_points: A PointCloud containing the calculated positions of the pointer tip relative to the EM tracker frame for each data frame.

    :type ctfiducials: str
    :type computed_points: pc.PointCloud

    :return: Transformation frame from tracker to CT coordinates.
    :rtype: TransformationFrame.TransformationFrame
    """

    # Load CT fiducial points
    ct_points = pc.fromfile(ctfiducials)

    # Register the computed points with the CT points
    registration_transform = computed_points.register(ct_points[0][0])

    return registration_transform
