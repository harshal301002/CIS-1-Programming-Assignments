# compute_dk.py
import numpy as np
import re
from pointcloud import PointCloud
from frame import Frame

def compute_dk(bodyA, bodyB, sampleReadings):
    """
    Calculates the pointer tip coordinates with respect to calibration body 'B' across different frames.
    """
    # Read data from bodyA
    with open(bodyA, 'r') as fid:
        x = fid.readline()
        # Extract integer from the first line
        na_match = re.match(r'(\d+)', x.strip())
        if na_match:
            na = int(na_match.group(1))
        else:
            raise ValueError(f"Cannot parse number of markers from line: {x}")
        # Read 'na' lines of marker data
        DA_data = np.array([np.fromstring(fid.readline().strip(), sep=' ') for _ in range(na)]).T
        DA = PointCloud(DA_data)
        # Read the next line for Pa
        Pa_line = fid.readline()
        while Pa_line.strip() == '':
            Pa_line = fid.readline()
        Pa = np.fromstring(Pa_line.strip(), sep=' ')

    # Read data from bodyB
    with open(bodyB, 'r') as fid:
        x = fid.readline()
        # Extract integer from the first line
        nb_match = re.match(r'(\d+)', x.strip())
        if nb_match:
            nb = int(nb_match.group(1))
        else:
            raise ValueError(f"Cannot parse number of markers from line: {x}")
        # Read 'nb' lines of marker data
        DB_data = np.array([np.fromstring(fid.readline().strip(), sep=' ') for _ in range(nb)]).T
        DB = PointCloud(DB_data)
        # Read the next line for Pb (not used here)
        Pb_line = fid.readline()
        while Pb_line.strip() == '':
            Pb_line = fid.readline()
        Pb = np.fromstring(Pb_line.strip(), sep=' ')

    # Read sample readings
    with open(sampleReadings, 'r') as fid:
        x = fid.readline()
        # Extract ns and nf from the first line
        ns_nf = re.findall(r'\d+', x)
        if len(ns_nf) >= 2:
            ns, nf = map(int, ns_nf[:2])
        else:
            raise ValueError(f"Cannot parse ns and nf from line: {x}")
        nd = ns - na - nb
        da = np.zeros((3, na, nf))
        db = np.zeros((3, nb, nf))
        dummy = np.zeros((3, nd, nf))

        for j in range(nf):
            for i in range(na):
                da_line = fid.readline()
                da[:, i, j] = np.fromstring(da_line.strip(), sep=' ')
            for i in range(nb):
                db_line = fid.readline()
                db[:, i, j] = np.fromstring(db_line.strip(), sep=' ')
            for i in range(nd):
                dummy_line = fid.readline()
                dummy[:, i, j] = np.fromstring(dummy_line.strip(), sep=' ')

    # Initialize dk
    dk = np.zeros((3, nf))

    # Compute transformations and dk
    # for i in range(nf):
    #     da_i = PointCloud(da[:, :, i])
    #     db_i = PointCloud(db[:, :, i])

    #     # Register da_i to DA to get F_A
    #     F_A = da_i.register(DA)

    #     # Register db_i to DB to get F_B
    #     F_B = db_i.register(DB)

    #     # Compute the inverse of F_B
    #     F_B_inv = F_B.inv

    #     # Compute the transformation from Body A to Body B
    #     F = F_B_inv.compose(F_A)

    #     # Apply the transformation to Pa
    #     Pa_transformed = F.rotation @ Pa + F.translation
    #     dk[:, i] = np.round(Pa_transformed, 2)

        # Compute transformations and dk
    for i in range(nf):
        # Create PointCloud instances
        da_i = PointCloud(da[:, :, i])
        db_i = PointCloud(db[:, :, i])

        # Register to find transformations
        F_A = da_i.register(DA)
        F_B = db_i.register(DB)

        # Invert F_B
        F_B_inv = F_B.inv

        # Compute F_AB
        F_AB = F_B_inv.compose(F_A)

        # Transform Pa
        dk[:, i] = F_AB.transform_point(Pa)

    return dk
