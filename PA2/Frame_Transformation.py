import scipy.linalg as scialg

class Frame:
    """
    Represents a transformation defined by rotation and translation matrices.
    """
    def __init__(self, rotation, translation):
        """
        Creates a frame transformation using a rotation matrix and translation vector.

        :param rotation: Matrix specifying rotation in the transformation
        :param translation: Vector specifying translation in the transformation

        :type rotation: numpy.array of shape (N, N), generally (3, 3)
        :type translation: numpy.array of shape (N, 1)
        """
        self.rotation = rotation  
        self.translation = translation

    @property
    def inv(self):
        """
        Calculates the inverse of the transformation frame.
        
        :return: A new Frame object representing the inverse rotation and translation
        :rtype: Frame
        """
        # Calculate the inverse of the rotation matrix
        r_inv = scialg.inv(self.rotation)
        # Return a Frame with the inverse rotation and adjusted translation
        return Frame(r_inv, -r_inv.dot(self.translation))

    def compose(self, f):
        """
        Combines this frame transformation with another, resulting in a cumulative transformation.

        :param f: Another Frame instance to combine with
        :type f: Frame

        :return: A new Frame representing the combined transformation
        :rtype: Frame
        """
        # Compute the combined rotation by multiplying the rotation matrices
        new_rotation = self.rotation.dot(f.rotation)
        # Compute the combined translation by applying current rotation to f's translation and adding current translation
        new_translation = self.rotation.dot(f.translation) + self.translation
        # Return a new Frame with the combined rotation and translation
        return Frame(new_rotation, new_translation)
