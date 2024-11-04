import scipy.linalg as linalg

class TransformationFrame:
    """
    A class to represent transformations in a coordinate frame.
    """
    def __init__(self, rotation_matrix, translation_vector):
        """
        Initializes a transformation with a rotation and translation.

        :param rotation_matrix: The matrix that represents the rotation of the frame.
        :param translation_vector: The vector that represents the translation of the frame.

        :type rotation_matrix: numpy.ndarray, a square matrix (typically 3x3)
        :type translation_vector: numpy.ndarray, a column vector (typically 3x1)
        """
        self.rotation_matrix = rotation_matrix
        self.translation_vector = translation_vector

    @property
    def inverse(self):
        """
        Computes the inverse of the current transformation.

        :return: A new TransformationFrame representing the inverse transformation.
        :rtype: TransformationFrame
        """
        inv_rotation = linalg.inv(self.rotation_matrix)
        inv_translation = -inv_rotation.dot(self.translation_vector)
        return TransformationFrame(inv_rotation, inv_translation)

    def combine(self, other_frame):
        """
        Composes this frame with another frame transformation.

        :param other_frame: Another TransformationFrame instance to combine with this frame.
        :type other_frame: TransformationFrame

        :return: A new TransformationFrame resulting from the combination.
        :rtype: TransformationFrame
        """
        combined_rotation = self.rotation_matrix.dot(other_frame.rotation_matrix)
        combined_translation = self.rotation_matrix.dot(other_frame.translation_vector) + self.translation_vector
        return TransformationFrame(combined_rotation, combined_translation)
