import unittest
import numpy as np
from FrameOperations import Frame, PointCloud

def generate_random_rotation():
    """ Generate a random 3x3 rotation matrix """
    angles = np.random.uniform(0, 2 * np.pi, (3,))
    rx = np.array([[1, 0, 0],
                   [0, np.cos(angles[0]), -np.sin(angles[0])],
                   [0, np.sin(angles[0]), np.cos(angles[0])]])

    ry = np.array([[np.cos(angles[1]), 0, np.sin(angles[1])],
                   [0, 1, 0],
                   [-np.sin(angles[1]), 0, np.cos(angles[1])]])

    rz = np.array([[np.cos(angles[2]), -np.sin(angles[2]), 0],
                   [np.sin(angles[2]), np.cos(angles[2]), 0],
                   [0, 0, 1]])

    return rz.dot(ry).dot(rx)


class TestRegistration(unittest.TestCase):

    def assert_matrices_close(self, mat1, mat2, atol=1e-4):
        """ Helper function to assert two matrices are approximately equal """
        self.assertTrue(np.allclose(mat1, mat2, atol=atol), 
                        msg=f"Matrix mismatch:\n{mat1}\n!=\n{mat2}")


    def assert_vectors_close(self, vec1, vec2, atol=1e-4):
        """ Helper function to assert two vectors are approximately equal """
        self.assertTrue(np.allclose(vec1, vec2, atol=atol), 
                        msg=f"Vector mismatch:\n{vec1}\n!=\n{vec2}")


    def test_case_1_arbitrary_rotation(self):
        """ Test a 3D rotation around an arbitrary axis and translation """
        source_points = np.array([[2, 5, 1], [1, 3, 2], [7, 4, 8]]) 
        rotation_matrix = generate_random_rotation()
        translation_vector = np.array([[3], [4], [5]])
        target_points = rotation_matrix.dot(source_points) + translation_vector

        source_cloud = PointCloud(source_points)
        target_cloud = PointCloud(target_points)
        recovered_frame = source_cloud.register(target_cloud)

        self.assert_matrices_close(recovered_frame.r, rotation_matrix)
        self.assert_vectors_close(recovered_frame.p, translation_vector)


    def test_case_2_large_translation(self):
        """ Test a large translation with minimal rotation """
        source_points = np.array([[2, 3, 1], [1, 4, 2], [0, 2, 3]])  
        rotation_matrix = np.eye(3)
        translation_vector = np.array([[100], [200], [300]])
        target_points = rotation_matrix.dot(source_points) + translation_vector

        source_cloud = PointCloud(source_points)
        target_cloud = PointCloud(target_points)
        recovered_frame = source_cloud.register(target_cloud)

        self.assert_matrices_close(recovered_frame.r, rotation_matrix)
        self.assert_vectors_close(recovered_frame.p, translation_vector)


    def test_case_3_small_rotation_large_translation(self):
        """ Test a small rotation combined with large translation """
        source_points = np.array([[1, 2, 3], [4, 5, 6], [6, 7, 8]]) 
        rotation_matrix = np.array([[0.9998477, -0.0174524, 0], [0.0174524, 0.9998477, 0], [0, 0, 1]]) 
        translation_vector = np.array([[100], [200], [300]])  
        target_points = rotation_matrix.dot(source_points) + translation_vector

        source_cloud = PointCloud(source_points)
        target_cloud = PointCloud(target_points)
        recovered_frame = source_cloud.register(target_cloud)

        self.assert_matrices_close(recovered_frame.r, rotation_matrix)
        self.assert_vectors_close(recovered_frame.p, translation_vector)


    def test_case_4_identity_transformation(self):
        """ Test with identity transformation (no rotation, no translation) """
        source_points = np.array([[1, 2, 3], [3, 4, 5], [5, 6, 7]]) 
        rotation_matrix = np.eye(3)  
        translation_vector = np.zeros((3, 1))  
        target_points = source_points

        source_cloud = PointCloud(source_points)
        target_cloud = PointCloud(target_points)
        recovered_frame = source_cloud.register(target_cloud)

        self.assert_matrices_close(recovered_frame.r, rotation_matrix)
        self.assert_vectors_close(recovered_frame.p, translation_vector)


    def test_case_5_random_rotation_translation(self):
        """ Test a random rotation and random translation """
        source_points = np.random.uniform(0, 10, (3, 10)) 
        rotation_matrix = generate_random_rotation()
        translation_vector = np.random.uniform(0, 10, (3, 1))
        target_points = rotation_matrix.dot(source_points) + translation_vector

        source_cloud = PointCloud(source_points)
        target_cloud = PointCloud(target_points)
        recovered_frame = source_cloud.register(target_cloud)

        self.assert_matrices_close(recovered_frame.r, rotation_matrix)
        self.assert_vectors_close(recovered_frame.p, translation_vector)


# Run the unittests
if __name__ == '__main__':
    unittest.main()