import PointCloud as pc

def compute_expected_marker_positions(calbody_file, calreadings_file):
    """
    Calculates the anticipated EM marker locations on a calibration object, using measured positions from trackers.

    :param calbody_file: Path to the calibration object data file.
    :param calreadings_file: Path to the file with tracker readings.

    :type calbody_file: str
    :type calreadings_file: str

    :return: A list containing expected EM marker positions on the calibration object for each frame.
    :rtype: list[pc.PointCloud]
    """
    # Load tracker frames and calibration object frame data
    tracker_frames = pc.fromfile(calreadings_file)
    calibration_frame = pc.fromfile(calbody_file)

    expected_positions = []
    for frame in tracker_frames:
        # Register the calibration object’s initial markers to those in the current tracker frame
        frame_transform_d = calibration_frame[0][0].register(frame[0])
        frame_transform_a = calibration_frame[0][1].register(frame[1])

        # Apply the composed transformation to estimate marker positions
        expected_positions.append(calibration_frame[0][2].transform(frame_transform_d.inverse.compose(frame_transform_a)))

    return expected_positions
