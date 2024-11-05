import PointCloud as pc

def c_expected(calbody, calreading):
    """
    Computes expected EM marker positions on a calibration object, given measured positions of markers on the object
    and data from each tracker.

    :param calbody: Path to the calibration object data file
    :param calreading: Path to the tracker readings file

    :return: List of expected positions of EM tracking markers on the calibration object for each frame
    :rtype: list of PointCloud.PointCloud
    """
    # Load the tracker and object data from files
    tracker_data = pc.inp_file(calreading)
    object_data = pc.inp_file(calbody)

    # List to store the computed expected marker positions
    expected_positions = []

    # Iterate over each frame of tracker data
    for tracker_frame in tracker_data:
        # Register frames from object data to corresponding tracker frames
        frame_registration_d = object_data[0][0].register(tracker_frame[0])
        frame_registration_a = object_data[0][1].register(tracker_frame[1])

        # Compute the transformation required to map object data to tracker frame
        transformation = frame_registration_d.inv.compose(frame_registration_a)
        expected_position = object_data[0][2].transform(transformation)

        # Append the calculated position to the results list
        expected_positions.append(expected_position)

    return expected_positions
