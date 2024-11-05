import numpy as np
import scipy.linalg as scialg
import pandas as pd
import Frame_Transformation


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
        Performs rigid-body registration with respect to another point cloud target_cloud, and returns the corresponding frame
        transformation.
        :param target_cloud: The point cloud being mapped to
        :type target_cloud: PointCloud

        :return: The Frame transformation F = [rot_matrix, trans_vector] from current frame to target_cloud
        :rtype: Frame.Frame
        """
        # Calculate centroids of self and target clouds
        centroid_self = np.mean(self.data, axis=1, keepdims=True)
        centroid_target = np.mean(target_cloud.data, axis=1, keepdims=True)

        demeaned_self = self.data - centroid_self
        demeaned_target = target_cloud.data - centroid_target

        # Compute cross-covariance matrix and solve for rotation using SVD
        cross_covariance = demeaned_self.dot(demeaned_target.T)
        u_mat, singular_vals, v_t_mat = scialg.svd(cross_covariance)

        u_mat = u_mat.T
        v_t_mat = v_t_mat.T

        reflection_adjustment = np.identity(v_t_mat.shape[1])
        reflection_adjustment[-1, -1] = scialg.det(v_t_mat.dot(u_mat))

        rot_matrix = v_t_mat.dot(reflection_adjustment.dot(u_mat))
        
        trans_vector = centroid_target - rot_matrix.dot(centroid_self)

        return Frame_Transformation.Frame(rot_matrix, trans_vector)

    def transform(self, frame_transformation):
        """
        Evaluate a frame transformation applied to the current point cloud.
        :param frame_transformation: The Frame transformation frame_transformation = [rot_matrix, trans_vector] to transform with
        :type frame_transformation: Frame.Frame

        :return: The resulting point cloud
        :rtype: PointCloud
        """
        transformed_points = frame_transformation.rotation.dot(self.data) + frame_transformation.translation
        return PointCloud(transformed_points)

    def add(self, new_cloud):
        """
        Adds the data in the PointCloud new_cloud to the current PointCloud, and outputs their union.

        :param new_cloud: The PointCloud to add
        :type new_cloud: PointCloud

        :return: The union of the two PointClouds, with the data from new_cloud after the data of the original
        :rtype: PointCloud
        """
        combined_cloud = PointCloud()
        if self.data is None:
            combined_cloud.data = new_cloud.data
        else:
            combined_cloud = PointCloud(np.concatenate((self.data, new_cloud.data), axis=1))
        return combined_cloud


def inp_file(filepath):
    """
    Extract a list of PointClouds from a file.
    :param filepath: The file path to the input data.
    :type filepath: str

    :return: A list of lists of PointClouds. Each internal list represents a frame, with the clouds therein
             representing different sets of known points.

    :rtype: [PointCloud][]
    """

    header_data = pd.read_csv(filepath, header=None, nrows=1)
    file_key = header_data.values[0, header_data.shape[1] - 1].split('.')[0].split('-')[-1]

    frame_counts = {'calbody': 1, 'calreadings': header_data.values[0, -2], 'empivot': header_data.values[0, -2],
                    'optpivot': header_data.values[0, -2], 'output1': header_data.values[0, -2], 'fiducials': 1,
                    'fiducialss': header_data.values[0, -2], 'nav': header_data.values[0, -2], 'output2': header_data.values[0, -2]}

    point_data = pd.read_csv(filepath, header=None, names=['x', 'y', 'z'], skiprows=1)

    initialize = True
    all_frames = []
    column_indices = None
    single_frame_size = None

    for frame_index in range(frame_counts[file_key]):
        if initialize:
            initialize = False
            column_indices = [0]
            offset = 0 if frame_counts[file_key] == 1 else 1
            for col_index in range(header_data.shape[1] - 1 - offset):
                column_indices.append(column_indices[col_index] + header_data.values[0, col_index])

            single_frame_size = column_indices[-1]

        frame_cloud_list = []
        for cloud_index in range(len(column_indices) - 1):
            frame_cloud_list.append(PointCloud(
                point_data.values[column_indices[cloud_index] + single_frame_size * frame_index:
                                  column_indices[cloud_index + 1] + single_frame_size * frame_index, :].T))

        all_frames.append(frame_cloud_list)

    return all_frames
