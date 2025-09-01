import numpy as np
from dataclasses import dataclass


@dataclass
class Matrix4:
    data: np.ndarray

    def __init__(self, data=None):
        if data is None:
            self.data = np.eye(4, dtype=np.float32)
        else:
            arr = np.array(data, dtype=np.float32)
            if arr.shape != (4, 4):
                raise ValueError("Matrix4 must be initialized with a 4x4 array")
            self.data = arr

    # --- static constructors ---
    @staticmethod
    def identity():
        return Matrix4(np.eye(4, dtype=np.float32))

    @staticmethod
    def translation(x, y, z):
        m = np.eye(4, dtype=np.float32)
        m[0, 3] = x
        m[1, 3] = y
        m[2, 3] = z
        return Matrix4(m)

    @staticmethod
    def scaling(sx, sy, sz):
        m = np.eye(4, dtype=np.float32)
        m[0, 0] = sx
        m[1, 1] = sy
        m[2, 2] = sz
        return Matrix4(m)

    @staticmethod
    def rotation_x(theta):
        c, s = np.cos(theta), np.sin(theta)
        m = np.eye(4, dtype=np.float32)
        m[1, 1] = c
        m[1, 2] = -s
        m[2, 1] = s
        m[2, 2] = c
        return Matrix4(m)

    @staticmethod
    def rotation_y(theta):
        c, s = np.cos(theta), np.sin(theta)
        m = np.eye(4, dtype=np.float32)
        m[0, 0] = c
        m[0, 2] = s
        m[2, 0] = -s
        m[2, 2] = c
        return Matrix4(m)

    @staticmethod
    def rotation_z(theta):
        c, s = np.cos(theta), np.sin(theta)
        m = np.eye(4, dtype=np.float32)
        m[0, 0] = c
        m[0, 1] = -s
        m[1, 0] = s
        m[1, 1] = c
        return Matrix4(m)

    # --- operator overloads ---
    def __add__(self, other):
        return Matrix4(self.data + other.data)

    def __sub__(self, other):
        return Matrix4(self.data - other.data)

    def __neg__(self):
        return Matrix4(-self.data)

    def __mul__(self, scalar):
        if isinstance(scalar, (int, float)):
            return Matrix4(self.data * scalar)
        raise TypeError("Can only multiply Matrix4 by scalar")

    def __rmul__(self, scalar):
        return self.__mul__(scalar)

    def __truediv__(self, scalar):
        if isinstance(scalar, (int, float)):
            return Matrix4(self.data / scalar)
        raise TypeError("Can only divide Matrix4 by scalar")

    def __matmul__(self, other: "Matrix4"):
        return Matrix4(self.data @ other.data)

    def __eq__(self, other):
        return np.allclose(self.data, other.data)

    def __getitem__(self, key):
        return self.data[key]

    def __setitem__(self, key, value):
        self.data[key] = value

    def __repr__(self):
        return f"Matrix4(\n{self.data}\n)"

    # --- transforms ---
    def transform_point(self, point):
        v = np.array([point.x, point.y, point.z, 1.0], dtype=np.float32)
        result = self.data @ v
        return type(point)(result[0], result[1], result[2])

    def transform_vector(self, vec):
        v = np.array([vec.x, vec.y, vec.z, 0.0], dtype=np.float32)
        result = self.data @ v
        return type(vec)(result[0], result[1], result[2])
