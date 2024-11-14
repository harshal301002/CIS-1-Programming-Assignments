import numpy as np
from frame import Frame

class PointCloud:
    """
    Class representing a point cloud. Consists of a numpy array of column vectors representing each point in the
    cloud, and pertinent methods for frame transformations, file IO, etc.
    """
    def __init__(self, data=None):
        """
        Initializes the point cloud, either empty or with the data provided
        :param data: Numpy array of column vectors, with each column representing each point in the point cloud
        :type data: numpy.array([numpy.float64][]), M x N (usually 3 x N)
        """
        self.data = data

    def register(self, target_cloud):
        """
        Performs rigid-body registration with respect to another point cloud target_cloud using the quaternion-based method,
        and returns the corresponding frame transformation.
        :param target_cloud: The point cloud being mapped to
        :type target_cloud: PointCloud

        :return: The Frame transformation F = [rot_matrix, trans_vector] from current frame to target_cloud
        :rtype: Frame
        """
        # Calculate centroids of self and target clouds
        centroid_self = np.mean(self.data, axis=1).reshape(-1, 1)
        centroid_target = np.mean(target_cloud.data, axis=1).reshape(-1, 1)

        # Demean the point clouds
        demeaned_self = self.data - centroid_self
        demeaned_target = target_cloud.data - centroid_target

        # Compute the covariance matrix
        H = demeaned_self @ demeaned_target.T

        # Build the symmetric 4x4 matrix
        delta = np.array([
            [H[0, 0] + H[1, 1] + H[2, 2], H[1, 2] - H[2, 1], H[2, 0] - H[0, 2], H[0, 1] - H[1, 0]],
            [H[1, 2] - H[2, 1], H[0, 0] - H[1, 1] - H[2, 2], H[0, 1] + H[1, 0], H[2, 0] + H[0, 2]],
            [H[2, 0] - H[0, 2], H[0, 1] + H[1, 0], -H[0, 0] + H[1, 1] - H[2, 2], H[1, 2] + H[2, 1]],
            [H[0, 1] - H[1, 0], H[2, 0] + H[0, 2], H[1, 2] + H[2, 1], -H[0, 0] - H[1, 1] + H[2, 2]]
        ])

        # Find the eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eig(delta)
        max_index = np.argmax(eigenvalues)
        q = eigenvectors[:, max_index]

        # Convert quaternion to rotation matrix
        rot_matrix = self.quaternion_to_rotation_matrix(q)

        # Compute translation
        trans_vector = centroid_target.flatten() - rot_matrix @ centroid_self.flatten()

        return Frame(rot_matrix, trans_vector)



    def transform(self, frame_transformation):
        """
        Evaluate a frame transformation applied to the current point cloud.
        :param frame_transformation: The Frame transformation frame_transformation = [rot_matrix, trans_vector] to transform with
        :type frame_transformation: Frame

        :return: The resulting point cloud
        :rtype: PointCloud
        """
        transformed_points = frame_transformation.rotation @ self.data + frame_transformation.translation.reshape(-1, 1)
        return PointCloud(transformed_points)

    def add(self, new_cloud):
        """
        Adds the data in the PointCloud new_cloud to the current PointCloud, and outputs their union.

        :param new_cloud: The PointCloud to add
        :type new_cloud: PointCloud

        :return: The union of the two PointClouds, with the data from new_cloud after the data of the original
        :rtype: PointCloud
        """
        if self.data is None:
            combined_data = new_cloud.data
        else:
            combined_data = np.concatenate((self.data, new_cloud.data), axis=1)
        return PointCloud(combined_data)

    @staticmethod
    def quaternion_to_rotation_matrix(q):
        """
        Converts a quaternion into a rotation matrix.

        :param q: Quaternion as a numpy array [q0, q1, q2, q3]
        :type q: numpy.array of shape (4,)

        :return: Rotation matrix corresponding to the quaternion
        :rtype: numpy.array of shape (3, 3)
        """
        q0, q1, q2, q3 = q
        R = np.array([
            [q0**2 + q1**2 - q2**2 - q3**2, 2*(q1*q2 - q0*q3),           2*(q1*q3 + q0*q2)],
            [2*(q1*q2 + q0*q3),           q0**2 - q1**2 + q2**2 - q3**2, 2*(q2*q3 - q0*q1)],
            [2*(q1*q3 - q0*q2),           2*(q2*q3 + q0*q1),           q0**2 - q1**2 - q2**2 + q3**2]
        ])
        return R
