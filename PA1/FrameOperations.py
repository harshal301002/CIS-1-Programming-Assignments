import numpy as np
import scipy.linalg as scialg

# Math Functions for vectors
def vector_add(v1, v2):
    return np.add(v1, v2)

def vector_sub(v1, v2):
    return np.subtract(v1, v2)

def vector_dot(v1, v2):
    return np.dot(v1, v2)

def vector_cross(v1, v2):
    return np.cross(v1, v2)

def vector_magnitude(v):
    return np.linalg.norm(v)

def normalize(v):
    return v / np.linalg.norm(v)

def skew(a):
    [ax, ay, az] = a.flat
    ska = np.array([
        [0,  -az,  ay],
        [az,  0,  -ax],
        [-ay, ax,  0 ]
    ])
    return ska

# Frame class for handling transformations (rotation + translation)
class Frame:
    """
    Class representing a coordinate frame transformation (rotation and translation).
:param r: Rotation matrix of the frame transformation.
:param p: Translation vector of the frame transformation.

:type r: numpy.array, shape (N, N), typically (3, 3)
:type p: numpy.array, shape (N, 1), typically (3, 1)

    """
    def __init__(self, r=np.zeros((3,3)), p=np.zeros((3,1))):

        self.r = r  # Rotation matrix
        self.p = p  # Translation vector

    @property
    def inv(self):
        """
        The inverse transformation
        :return: A Frame transformation with components [r, p] corresponding to the inverse of the current
                 transformation
        :rtype: Frame
        """

        r_inv = scialg.inv(self.r)
        return Frame(r_inv, -r_inv.dot(self.p))


    def compose(self, f):
        """
        Frame composition with another frame f

        :param f: The Frame to compose with
        ::type f: Frame

        :return: A Frame transformation with components corresponding to the composition of the components of the
                 current frame with those of f
        :rtype: Frame
        """
        return Frame(self.r.dot(f.r), self.r.dot(f.p) + self.p)


    def apply_to_point(self, point):
        return self.r.dot(point) + self.p


    def apply_to_cloud(self, point_cloud):
        return self.r.dot(point_cloud) + self.p

    def __repr__(self):
        return f"Frame(rotation={self.r}, translation={self.p})"
