from dataclasses import dataclass

import numpy as np

from core.math.vectors import Point3, Vector3


class MatrixError(Exception):
    pass


@dataclass
class Matrix2:
    data: np.ndarray

    def __init__(self, data=None):
        if data is None:
            self.data = np.eye(2, dtype=np.float32)
        else:
            arr = np.array(data, dtype=np.float32)
            if arr.shape != (2, 2):
                raise ValueError("Matrix2 must be initialized with a 2x2 array")
            self.data = arr

    @staticmethod
    def identity():
        return Matrix2(np.eye(2, dtype=np.float32))

    @staticmethod
    def scaling(sx: float, sy: float):
        m = np.eye(2, dtype=np.float32)
        m[0, 0] = sx
        m[1, 1] = sy
        return Matrix2(m)

    @staticmethod
    def rotation(theta: float):
        c, s = np.cos(theta), np.sin(theta)
        m = np.array([[c, -s], [s, c]], dtype=np.float32)
        return Matrix2(m)

    @staticmethod
    def shear(shx: float = 0.0, shy: float = 0.0):
        m = np.array([[1.0, shx], [shy, 1.0]], dtype=np.float32)
        return Matrix2(m)

    def __add__(self, other: "Matrix2"):
        return Matrix2(self.data + other.data)

    def __sub__(self, other: "Matrix2"):
        return Matrix2(self.data - other.data)

    def __neg__(self):
        return Matrix2(-self.data)

    def __mul__(self, scalar: float):
        if isinstance(scalar, (int, float)):
            return Matrix2(self.data * scalar)
        raise TypeError("Can only multiply Matrix2 by scalar")

    def __rmul__(self, scalar: float):
        return self.__mul__(scalar)

    def __truediv__(self, scalar: float):
        if isinstance(scalar, (int, float)):
            return Matrix2(self.data / scalar)
        raise TypeError("Can only divide Matrix2 by scalar")

    def __matmul__(self, other: "Matrix2"):
        return Matrix2(self.data @ other.data)

    def __eq__(self, other: "Matrix2"):
        if not isinstance(other, Matrix2):
            return False
        return np.allclose(self.data, other.data)

    def __getitem__(self, key):
        return self.data[key]

    def __setitem__(self, key, value):
        self.data[key] = value

    def __repr__(self):
        return f"Matrix2(\n{self.data}\n)"

    def transpose(self) -> "Matrix2":
        return Matrix2(self.data.T)

    def determinant(self) -> np.float32:
        return self.data[0, 0] * self.data[1, 1] - self.data[1, 0] * self.data[0, 1]

    def inverse(self) -> "Matrix2":
        a = self.data[0, 0]
        b = self.data[0, 1]
        c = self.data[1, 0]
        d = self.data[1, 1]

        det = self.determinant()

        if det == 0:
            raise MatrixError("Matrix is not inversible (det = 0)")

        det_inv = 1 / det

        return det_inv * Matrix2(data=np.array([[d, -b], [-c, a]]))


@dataclass
class Matrix3:
    data: np.ndarray

    def __init__(self, data=None):
        if data is None:
            self.data = np.eye(3, dtype=np.float32)
        else:
            arr = np.array(data, dtype=np.float32)
            if arr.shape != (3, 3):
                raise ValueError("Matrix3 must be initialized with a 3x3 array")
            self.data = arr

    @staticmethod
    def identity():
        return Matrix3(np.eye(3, dtype=np.float32))

    @staticmethod
    def scaling(sx: float, sy: float, sz: float = 1.0):
        m = np.eye(3, dtype=np.float32)
        m[0, 0] = sx
        m[1, 1] = sy
        m[2, 2] = sz
        return Matrix3(m)

    @staticmethod
    def rotation_x(theta: float):
        c, s = np.cos(theta), np.sin(theta)
        m = np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float32)
        return Matrix3(m)

    @staticmethod
    def rotation_y(theta: float):
        c, s = np.cos(theta), np.sin(theta)
        m = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float32)
        return Matrix3(m)

    @staticmethod
    def rotation_z(theta: float):
        c, s = np.cos(theta), np.sin(theta)
        m = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32)
        return Matrix3(m)

    @staticmethod
    def from_euler(rx: float, ry: float, rz: float):
        return Matrix3.rotation_z(rz) @ Matrix3.rotation_y(ry) @ Matrix3.rotation_x(rx)

    @staticmethod
    def homogeneous_2d_translation(tx: float, ty: float):
        m = np.eye(3, dtype=np.float32)
        m[0, 2] = tx
        m[1, 2] = ty
        return Matrix3(m)

    @staticmethod
    def homogeneous_2d_rotation(theta: float):
        c, s = np.cos(theta), np.sin(theta)
        m = np.array(
            [
                [c, -s, 0],
                [s, c, 0],
                [0, 0, 1],
            ],
            dtype=np.float32,
        )
        return Matrix3(m)

    @staticmethod
    def homogeneous_2d_scaling(sx: float, sy: float):
        m = np.eye(3, dtype=np.float32)
        m[0, 0] = sx
        m[1, 1] = sy
        return Matrix3(m)

    @staticmethod
    def homogeneous_2d(
        tx: float = 0.0, ty: float = 0.0, theta: float = 0.0, sx: float = 1.0, sy: float = 1.0
    ):
        T = Matrix3.homogeneous_2d_translation(tx, ty)
        R = Matrix3.homogeneous_2d_rotation(theta)
        S = Matrix3.homogeneous_2d_scaling(sx, sy)
        return T @ R @ S

    def __add__(self, other: "Matrix3"):
        return Matrix3(self.data + other.data)

    def __sub__(self, other: "Matrix3"):
        return Matrix3(self.data - other.data)

    def __neg__(self):
        return Matrix3(-self.data)

    def __mul__(self, scalar: float):
        if isinstance(scalar, (int, float)):
            return Matrix3(self.data * scalar)
        raise TypeError("Can only multiply Matrix3 by scalar")

    def __rmul__(self, scalar: float):
        return self.__mul__(scalar)

    def __truediv__(self, scalar: float):
        if isinstance(scalar, (int, float)):
            return Matrix3(self.data / scalar)
        raise TypeError("Can only divide Matrix3 by scalar")

    def __matmul__(self, other: "Matrix3"):
        return Matrix3(self.data @ other.data)

    def __eq__(self, other: "Matrix2"):
        if not isinstance(other, Matrix3):
            return False
        return np.allclose(self.data, other.data)

    def __getitem__(self, key):
        return self.data[key]

    def __setitem__(self, key, value):
        self.data[key] = value

    def __repr__(self):
        return f"Matrix3(\n{self.data}\n)"

    def transpose(self) -> "Matrix3":
        return Matrix3(self.data.T)

    def submatrix(self, row: int, column: int) -> Matrix2:
        if not (0 <= row < 3 and 0 <= column < 3):
            raise IndexError("Row and column must be in range [0, 2]")

        mask_rows = [i for i in range(3) if i != row]
        mask_cols = [j for j in range(3) if j != column]

        return Matrix2(data=self.data[np.ix_(mask_rows, mask_cols)])

    def minor(self, row: int, column: int) -> np.float32:
        sub = self.submatrix(row, column)
        return sub.determinant()

    def cofactor(self, row: int, column: int) -> np.float32:
        minor = self.minor(row, column)

        if (row + column) % 2 == 0:
            return minor
        else:
            return -minor

    def determinant(self) -> np.float32:
        c1 = self.cofactor(0, 0)
        c2 = self.cofactor(0, 1)
        c3 = self.cofactor(0, 2)

        return self.data[0, 0] * c1 + self.data[0, 1] * c2 + self.data[0, 2] * c3

    def inverse(self) -> "Matrix3":
        mat = Matrix3()

        for y in range(3):
            for x in range(3):
                c = self.cofactor(y, x)

                # x, y instead of y, x (transpose)
                mat[x, y] = c / self.determinant()

        return mat


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

    @staticmethod
    def identity():
        return Matrix4(np.eye(4, dtype=np.float32))

    @staticmethod
    def translation(x, y, z):
        m = Matrix4.identity()
        m[0, 3] = x
        m[1, 3] = y
        m[2, 3] = z
        return m

    @staticmethod
    def scaling(sx, sy, sz):
        m = Matrix4.identity()
        m[0, 0] = sx
        m[1, 1] = sy
        m[2, 2] = sz
        return m

    @staticmethod
    def rotation_x(theta):
        c, s = np.cos(theta), np.sin(theta)
        m = Matrix4.identity()
        m[1, 1] = c
        m[1, 2] = -s
        m[2, 1] = s
        m[2, 2] = c
        return m

    @staticmethod
    def rotation_y(theta):
        c, s = np.cos(theta), np.sin(theta)
        m = Matrix4.identity()
        m[0, 0] = c
        m[0, 2] = s
        m[2, 0] = -s
        m[2, 2] = c
        return m

    @staticmethod
    def rotation_z(theta):
        c, s = np.cos(theta), np.sin(theta)
        m = Matrix4.identity()
        m[0, 0] = c
        m[0, 1] = -s
        m[1, 0] = s
        m[1, 1] = c
        return m

    @staticmethod
    def shear(xy, xz, yx, yz, zx, zy):
        m = Matrix4.identity()

        m[0, 1] = xy
        m[0, 2] = xz
        m[1, 0] = yx
        m[1, 2] = yz
        m[2, 0] = zx
        m[2, 1] = zy

        return m

    @staticmethod
    def view_transform(p_from: Point3, p_to: Point3, v_up: Vector3) -> "Matrix4":
        v_forward = (p_to - p_from).normalize()
        v_up_n = v_up.normalized()
        v_left = v_forward.cross(v_up_n)
        v_true_up = v_left.cross(v_forward)

        m_orientation = Matrix4(
            np.array(
                [
                    [v_left.x, v_left.y, v_left.z, 0.0],
                    [v_true_up.x, v_true_up.y, v_true_up.z, 0.0],
                    [-v_forward.x, -v_forward.y, -v_forward.z, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            )
        )

        return m_orientation @ Matrix4.translation(-p_from.x, -p_from.y, -p_from.z)

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

    def transpose(self):
        return Matrix4(self.data.T)

    def submatrix(self, row: int, column: int) -> Matrix3:
        if not (0 <= row < 4 and 0 <= column < 4):
            raise IndexError("Row and column must be in range [0, 3]")

        mask_rows = [i for i in range(4) if i != row]
        mask_cols = [j for j in range(4) if j != column]

        return Matrix3(data=self.data[np.ix_(mask_rows, mask_cols)])

    def minor(self, row: int, column: int) -> np.float32:
        sub = self.submatrix(row, column)
        return sub.determinant()

    def cofactor(self, row: int, column: int) -> np.float32:
        minor = self.minor(row, column)

        if (row + column) % 2 == 0:
            return minor
        else:
            return -minor

    def determinant(self) -> "Matrix4":
        c1 = self.cofactor(0, 0)
        c2 = self.cofactor(0, 1)
        c3 = self.cofactor(0, 2)
        c4 = self.cofactor(0, 3)

        return (
            self.data[0, 0] * c1
            + self.data[0, 1] * c2
            + self.data[0, 2] * c3
            + self.data[0, 3] * c4
        )

    def inverse(self) -> "Matrix4":
        mat = Matrix4()

        for y in range(4):
            for x in range(4):
                c = self.cofactor(y, x)

                # x, y instead of y, x (transpose)
                mat[x, y] = c / self.determinant()

        return mat

    def translate(self, vec: Vector3) -> "Matrix4":
        return Matrix4(self.data) @ Matrix4.translation(vec.x, vec.y, vec.z)

    def scale(self, vec: Vector3) -> "Matrix4":
        return Matrix4(self.data) @ Matrix4.scaling(vec.x, vec.y, vec.z)

    def rotate_along_x(self, theta: np.float32) -> "Matrix4":
        return Matrix4(self.data) @ Matrix4.rotation_x(theta)

    def rotate_along_y(self, theta: np.float32) -> "Matrix4":
        return Matrix4(self.data) @ Matrix4.rotation_y(theta)

    def rotate_along_z(self, theta: np.float32) -> "Matrix4":
        return Matrix4(self.data) @ Matrix4.rotation_z(theta)
