import numpy as np
import scipy.linalg as linalg
import pandas as pd
import TransformationFrame

class PointCloud:
    """
    Represents a collection of points in 3D space. Each point is a column vector, and the class
    includes methods for transformations, data operations, and file input/output.
    """
    def __init__(self, data=None):
        """
        Initializes the point cloud with optional data.

        :param data: Numpy array where each column represents a point in the cloud.
        :type data: numpy.ndarray, typically M x N (e.g., 3 x N)
        """
        self.data = data

    def register(self, target_cloud):
        """
        Computes a transformation to register this cloud with another point cloud.

        :param target_cloud: The target PointCloud for alignment.
        :type target_cloud: PointCloud

        :return: A TransformationFrame with rotation and translation for aligning this cloud to target_cloud.
        :rtype: TransformationFrame.TransformationFrame
        """
        # Calculate centroids of both clouds
        source_centroid = np.mean(self.data, axis=1, keepdims=True)
        target_centroid = np.mean(target_cloud.data, axis=1, keepdims=True)

        # Center each cloud around its respective centroid
        source_centered = self.data - source_centroid
        target_centered = target_cloud.data - target_centroid

        # Use SVD to compute the rotation matrix
        covariance_matrix = source_centered.dot(target_centered.T)
        u, s, v_t = linalg.svd(covariance_matrix)

        # Ensure a proper rotation by adjusting for any reflection
        correction_matrix = np.identity(v_t.shape[1])
        correction_matrix[-1, -1] = linalg.det(v_t.dot(u.T))

        rotation_matrix = v_t.dot(correction_matrix.dot(u.T))
        translation_vector = target_centroid - rotation_matrix.dot(source_centroid)

        return TransformationFrame.TransformationFrame(rotation_matrix, translation_vector)

    def transform(self, frame_transform):
        """
        Applies a frame transformation to this point cloud.

        :param frame_transform: The TransformationFrame object with rotation and translation to apply.
        :type frame_transform: TransformationFrame.TransformationFrame

        :return: A new PointCloud with the transformed points.
        :rtype: PointCloud
        """
        transformed_data = frame_transform.rotation_matrix.dot(self.data) + frame_transform.translation_vector
        return PointCloud(transformed_data)

    def add(self, other_cloud):
        """
        Combines this point cloud with another and returns the merged result.

        :param other_cloud: The PointCloud to merge with.
        :type other_cloud: PointCloud

        :return: A new PointCloud containing points from both clouds.
        :rtype: PointCloud
        """
        if self.data is None:
            return PointCloud(other_cloud.data)
        combined_data = np.concatenate((self.data, other_cloud.data), axis=1)
        return PointCloud(combined_data)

def fromfile(filepath):
    """
    Loads a series of PointClouds from a file.

    :param filepath: Path to the input file.
    :type filepath: str

    :return: A list of PointClouds for each frame, representing different sets of known points.
    :rtype: list[list[PointCloud]]
    """
    header_info = pd.read_csv(filepath, header=None, nrows=1)
    dataset_name = header_info.values[0, -1].split('.')[0].split('-')[-1]

    frame_counts = {
        'calbody': 1, 'calreadings': header_info.values[0, -2], 'empivot': header_info.values[0, -2],
        'optpivot': header_info.values[0, -2], 'output1': header_info.values[0, -2], 'fiducials': 1,
        'fiducialss': header_info.values[0, -2], 'nav': header_info.values[0, -2], 'output2': header_info.values[0, -2]
    }

    data_frame = pd.read_csv(filepath, header=None, names=['x', 'y', 'z'], skiprows=1)

    frame_data = []
    start_index = True
    point_indices = None
    points_per_frame = None

    for frame in range(frame_counts[dataset_name]):
        if start_index:
            start_index = False
            point_indices = [0]
            offset = 0 if frame_counts[dataset_name] == 1 else 1
            for idx in range(header_info.shape[1] - 1 - offset):
                point_indices.append(point_indices[idx] + header_info.values[0, idx])

            points_per_frame = point_indices[-1]

        clouds = []
        for i in range(len(point_indices) - 1):
            points = data_frame.values[point_indices[i] + points_per_frame * frame : point_indices[i + 1] + points_per_frame * frame, :].T
            clouds.append(PointCloud(points))

        frame_data.append(clouds)

    return frame_data
