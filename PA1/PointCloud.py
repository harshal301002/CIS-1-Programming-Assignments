import numpy as np
import scipy.linalg as scialg
from FrameOperations import Frame, skew

class PointCloud:
    def __init__(self, points):
        """
        Class for representing a set of coordinate locations within a frame
        :param points: Numpy array of column vectors, with each column representing each point in the point cloud 
        """
        self.points = points


    def register(self, bcloud):
        """"Return the frame for a rigid point cloud to point cloud transformation"""
        
        # Ensure that both clouds have the same number of points
        assert np.shape(self.points) == np.shape(bcloud.points), "Point clouds must be of equal size"
        
        
        # Find the centroids of each point cloud
        a_bar = np.mean(self.points, axis=1, keepdims=True)
        b_bar = np.mean(bcloud.points, axis=1, keepdims=True)

        # Get points in terms of the cloud centroid
        source = self.points.T - a_bar.T
        target = bcloud.points.T - b_bar.T

        # Construct an M matrix for each correponding set of points, where M(a,b)q = 0
        # Note that q is the unit quaternion corresponding to rotation matrix R
        M = None
        for (source, target) in zip(source, target):
            a = np.array([source]).T
            b = np.array([target]).T

            quad1 = np.array([[0]])
            quad2 = (b - a).T
            quad3 = (b - a)
            quad4 = skew(b + a)
            top = np.concatenate((quad1, quad2), axis=1)
            bot = np.concatenate((quad3, quad4), axis=1)
            m = np.concatenate((top, bot), axis=0)
            if type(M) == type(None):
                M = m
            else:
                M = np.vstack((M,m))
            
        # Derive q from the SVD of M
        [U, S, Vt] = scialg.svd(M)
        V = Vt.T
        q = V[:,3]
        [q0, q1, q2, q3] = q.flat
        [q02, q12, q22, q32] = [q0**2, q1**2, q2**2, q3**2]

        # Derive F = (R,p) from q
        R = np.round(np.array([
            [q02+q12-q22-q32, 2*(q1*q2-q0*q3), 2*(q1*q3+q0*q2)],
            [2*(q1*q2+q0*q3), q02-q12+q22-q32, 2*(q2*q3-q0*q1)],
            [2*(q1*q3-q0*q2), 2*(q2*q3+q0*q1), q02-q12-q22+q32]
        ]), 5)
        p = b_bar - R.dot(a_bar)

        return Frame(R,p)


    def pivot(self, bclouds):
        """
        Perform pivot calibration using a set of point clouds captured at different probe poses.
        
        :param bclouds: A list of PointCloud objects, each representing different poses of the probe.
        :return: The calibration offset and the computed pivot point.
        """
        # Compute centroid of the reference cloud (self)
        centroid = np.mean(self.points, axis=1, keepdims=True)
        R = None
        p = None

        # Construct a least squares matrix
        for cloud in bclouds:
            gj = PointCloud(cloud.points - centroid)
            Fg = gj.register(cloud)

            Rj = np.hstack((Fg.r, -np.eye(3)))
            pj = -Fg.p

            if type(R) == type(None):
                R, p = Rj, pj
            else:
                R = np.vstack((R,Rj))
                p = np.vstack((p,pj))
        
        # Solve the least squares problem
        p_soln = np.linalg.lstsq(R, p)[0]
        
        # Extract the calibration offset and the pivot point from the solution
        p_cal = np.array(p_soln[3:6])  # Calibration offset
        p_piv = np.array(p_soln[0:3])  # Pivot point

        return p_cal, p_piv

    
    def transform(self, f):
        """
        Evaluate a frame transformation applied to the current point cloud.
        :param f: The Frame transformation f = [r, p] to transform with
        :type f: Frame.Frame
        :return: The resulting point cloud
        :rtype: PointCloud
        """
        return PointCloud(f.r.dot(self.data) + f.p)