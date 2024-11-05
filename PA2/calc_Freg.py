import PointCloud as pc


def find_freg(ctfiducials, Cs):
    """
    Calculates the transformation from tracker coordinates to CT coordinates.

    :param ctfiducials: Path to the file containing the positions of each fiducial pin in the CT coordinate system
    :param Cs: A PointCloud instance representing the computed positions of the pointer tip relative to the EM tracker frame for each data frame
    
    :type ctfiducials: str
    :type Cs: PointCloud.PointCloud
    
    :return: A frame transformation from tracker coordinates to CT coordinates
    :rtype: Frame.Frame

    """

    load_fid = pc.inp_file(ctfiducials)
    registration = Cs.register(load_fid[0][0])

    return registration
