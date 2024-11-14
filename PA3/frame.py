import numpy as np
import scipy.linalg as scialg

class Frame:
    """
    Represents a transformation defined by rotation and translation matrices.
    """
    def __init__(self, rotation, translation):
        """
        Creates a frame transformation using a rotation matrix and translation vector.
        :param rotation: Rotation matrix (3x3).
        :param translation: Translation vector (3,).
        """
        self.rotation = rotation
        self.translation = translation

    @property
    def inv(self):
        """
        Calculates the inverse of the transformation frame.
        :return: A new Frame object representing the inverse rotation and translation
        """
        r_inv = self.rotation.T
        t_inv = -r_inv @ self.translation
        return Frame(r_inv, t_inv)

    def compose(self, other):
        """
        Combines this frame transformation with another, resulting in a cumulative transformation.
        :param other: Another Frame instance to combine with
        :return: A new Frame representing the combined transformation
        """
        new_rotation = self.rotation @ other.rotation
        new_translation = self.rotation @ other.translation + self.translation
        return Frame(new_rotation, new_translation)

    def transform_point(self, point):
        """
        Applies the frame transformation to a point.
        :param point: The point to transform (numpy array of shape (3,))
        :return: Transformed point (numpy array of shape (3,))
        """
        return self.rotation @ point + self.translation