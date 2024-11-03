import unittest
import numpy as np
from PointCloud import PointCloud, pivot

class TestPivotCalibration(unittest.TestCase):

    def setUp(self):
        """
        Setup the test cases with predefined point cloud data.
        """
        self.point_clouds = []

        # Create synthetic point cloud data for different poses
        points_1 = np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3]])
        points_2 = np.array([[2, 1, 1], [3, 2, 2], [4, 3, 3]])
        points_3 = np.array([[1, 2, 1], [2, 3, 2], [3, 4, 3]])

        # Store point clouds with slightly different poses
        self.point_clouds.append([PointCloud(points_1)])
        self.point_clouds.append([PointCloud(points_2)])
        self.point_clouds.append([PointCloud(points_3)])

        # Ground truth calibration offset and pivot point
        self.calibration_offset = np.array([0.5, 0.5, 0.5])  # More realistic offset values
        self.pivot_point = np.array([0, 0, 0])

    def test_pivot_stationary(self):
        """
        Test the pivot calibration with stationary probe data.
        """
        calibration_offset, pivot_point = pivot(self.point_clouds, 0)

        # Check if the calculated calibration offset and pivot point match the ground truth
        np.testing.assert_allclose(calibration_offset, self.calibration_offset, atol=1e-3, err_msg="Calibration offset is incorrect")
        np.testing.assert_allclose(pivot_point, self.pivot_point, atol=1e-3, err_msg="Pivot point is incorrect")

    def test_pivot_movement(self):
        """
        Test the pivot calibration with a moving probe in a straight line.
        """
        # Create linear movement points
        points_1 = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]])
        points_2 = np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3]])
        points_3 = np.array([[2, 2, 2], [3, 3, 3], [4, 4, 4]])

        self.point_clouds = []
        self.point_clouds.append([PointCloud(points_1)])
        self.point_clouds.append([PointCloud(points_2)])
        self.point_clouds.append([PointCloud(points_3)])

        calibration_offset, pivot_point = pivot(self.point_clouds, 0)

        # Check if the calculated calibration offset and pivot point match the expected values
        np.testing.assert_allclose(calibration_offset, np.array([0, 0, 0]), atol=1e-3, err_msg="Calibration offset is incorrect for movement")
        np.testing.assert_allclose(pivot_point, np.array([1, 1, 1]), atol=1e-3, err_msg="Pivot point is incorrect for movement")

    def test_single_frame(self):
        """
        Test pivot calibration with only a single frame.
        """
        single_frame_clouds = [[PointCloud(np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3]]))]]

        calibration_offset, pivot_point = pivot(single_frame_clouds, 0)

        # For a single frame, the pivot point and offset should be zero
        np.testing.assert_allclose(calibration_offset, np.zeros(3), atol=1e-3, err_msg="Calibration offset should be zero for a single frame")
        np.testing.assert_allclose(pivot_point, np.zeros(3), atol=1e-3, err_msg="Pivot point should be zero for a single frame")

    def test_random_movement(self):
        """
        Test pivot calibration with random probe movements.
        """
        # Generate random point clouds
        np.random.seed(42)
        points_1 = np.random.rand(3, 3) * 10
        points_2 = np.random.rand(3, 3) * 10
        points_3 = np.random.rand(3, 3) * 10

        self.point_clouds = []
        self.point_clouds.append([PointCloud(points_1)])
        self.point_clouds.append([PointCloud(points_2)])
        self.point_clouds.append([PointCloud(points_3)])

        calibration_offset, pivot_point = pivot(self.point_clouds, 0)

        # Check if the calculated calibration offset and pivot point do not return nonsensical values
        np.testing.assert_array_less(np.abs(calibration_offset), 100, err_msg="Calibration offset should be within reasonable bounds")
        np.testing.assert_array_less(np.abs(pivot_point), 100, err_msg="Pivot point should be within reasonable bounds")

if __name__ == '__main__':
    unittest.main()
