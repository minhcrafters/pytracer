from dataclasses import dataclass
from typing import Iterable, Tuple, Union
import numpy as np

Number = Union[float, int, np.number]


@dataclass
class Vector2:
    x: np.float32 = 0.0
    y: np.float32 = 0.0

    @classmethod
    def zero(cls) -> "Vector2":
        return cls(0.0, 0.0)

    @classmethod
    def one(cls) -> "Vector2":
        return cls(1.0, 1.0)

    @classmethod
    def unit_x(cls) -> "Vector2":
        return cls(1.0, 0.0)

    @classmethod
    def unit_y(cls) -> "Vector2":
        return cls(0.0, 1.0)

    @classmethod
    def from_iterable(cls, it: Iterable[Number]) -> "Vector2":
        a = np.array(it)
        if len(a) != 2:
            raise ValueError("Iterable must be length 2")
        return cls(a[0], a[1])

    def __repr__(self) -> str:
        return f"Vector2(x={np.float32(self.x):.6f}, y={np.float32(self.y):.6f})"

    def to_array(self, dtype=np.float32) -> np.ndarray:
        return np.array([self.x, self.y], dtype=dtype)

    def copy(self) -> "Vector2":
        return Vector3(self.x, self.y)

    def _coerce_operand(self, other):
        if isinstance(other, Vector2):
            return other
        if isinstance(other, Number):
            return Vector2(np.float32(other), np.float32(other))
        raise TypeError(f"Unsupported operand type: {type(other)}")

    def __add__(self, other):
        o = self._coerce_operand(other)
        return Vector2(self.x + o.x, self.y + o.y)

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        o = self._coerce_operand(other)
        return Vector2(self.x - o.x, self.y - o.y)

    def __rsub__(self, other):
        o = self._coerce_operand(other)
        return Vector2(o.x - self.x, o.y - self.y)

    def __neg__(self):
        return Vector2(-self.x, -self.y)

    def __mul__(self, other):
        if isinstance(other, Number):
            s = np.float32(other)
            return Vector2(self.x * s, self.y * s)
        if isinstance(other, Vector2):
            return Vector2(self.x * other.x, self.y * other.y)
        return NotImplemented

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        if isinstance(other, Number):
            s = np.float32(other)
            if s == 0:
                raise ZeroDivisionError("division by zero")
            return Vector2(self.x / s, self.y / s)
        if isinstance(other, Vector2):
            return Vector2(self.x / other.x, self.y / other.y)
        return NotImplemented

    def __iadd__(self, other):
        o = self._coerce_operand(other)
        self.x += o.x
        self.y += o.y
        return self

    def __isub__(self, other):
        o = self._coerce_operand(other)
        self.x -= o.x
        self.y -= o.y
        return self

    def __imul__(self, other):
        if isinstance(other, Number):
            s = np.float32(other)
            self.x *= s
            self.y *= s
            return self
        if isinstance(other, Vector2):
            self.x *= other.x
            self.y *= other.y
            return self
        return NotImplemented

    def __itruediv__(self, other):
        if isinstance(other, Number):
            s = np.float32(other)
            if s == 0:
                raise ZeroDivisionError("division by zero")
            self.x /= s
            self.y /= s
            return self
        if isinstance(other, Vector3):
            self.x /= other.x
            self.y /= other.y
            return self
        return NotImplemented

    def __eq__(self, other) -> bool:
        if not isinstance(other, Vector2):
            return False
        return bool((self.x == other.x) and (self.y == other.y))

    def __iter__(self):
        yield np.float32(self.x)
        yield np.float32(self.y)

    def __getitem__(self, idx: int) -> float:
        if idx == 0:
            return np.float32(self.x)
        if idx == 1:
            return np.float32(self.y)
        raise IndexError("Index out of range for Vector2")

    def dot(self, other: "Vector2") -> np.float32:
        return np.float32(self.x * other.x + self.y * other.y)

    def cross(self, other: "Vector2") -> "Vector2":
        cx = self.y * other.z - self.z * other.y
        cy = self.z * other.x - self.x * other.z
        return Vector2(cx, cy)

    @property
    def sqr_magnitude(self) -> np.float32:
        return np.float32(self.x * self.x + self.y * self.y)

    @property
    def magnitude(self) -> np.float32:
        return np.float32(np.sqrt(np.float32(self.sqr_magnitude)))

    def normalized(self) -> "Vector2":
        mag = np.float32(self.magnitude)
        if mag == 0.0:
            return Vector2.zero()
        inv = np.float32(1.0 / mag)
        return Vector2(self.x * inv, self.y * inv)

    def normalize(self) -> "Vector2":
        """In-place normalization. Returns self."""
        mag = np.float32(self.magnitude)
        if mag == 0:
            self.x = self.y = np.float32(0.0)
            return self
        inv = np.float32(1.0 / mag)
        self.x *= inv
        self.y *= inv
        return self

    def distance_to(self, other: "Vector2") -> np.float32:
        return (self - other).magnitude

    def angle_to(self, other: "Vector2") -> float:
        """Return angle in radians between self and other."""
        denom = np.float32(self.magnitude * other.magnitude)
        if denom == 0:
            raise ValueError("Cannot compute angle with zero-length vector")
        cos_theta = np.float32(self.dot(other)) / denom
        cos_theta = max(-1.0, min(1.0, cos_theta))
        return np.acos(cos_theta)

    def project_onto(self, other: "Vector2") -> "Vector2":
        """Project self onto 'other' (other may be non-normalized)."""
        denom = np.float32(other.sqr_magnitude)
        if denom == 0:
            return Vector2.zero()
        factor = np.float32(self.dot(other)) / denom
        return Vector3(other.x * np.float32(factor), other.y * np.float32(factor))

    def reflect(self, normal: "Vector2") -> "Vector2":
        """Reflect this vector across a normal. Normal is expected to be normalized."""
        n = normal.normalized()
        two_dot = np.float32(2.0) * self.dot(n)
        return Vector2(self.x - two_dot * n.x, self.y - two_dot * n.y)

    def lerp(self, other: "Vector2", t: float) -> "Vector2":
        t = np.float32(t)
        return Vector2(
            self.x + (other.x - self.x) * t,
            self.y + (other.y - self.y) * t,
        )

    def clamp_magnitude(self, max_len: float) -> "Vector2":
        max_len = np.float32(max_len)
        sq = np.float32(self.sqr_magnitude)
        if sq <= max_len * max_len:
            return self.copy()
        return self.normalized() * max_len

    def almost_equal(self, other: "Vector2", eps: float = 1e-6) -> bool:
        return abs(np.float32(self.x - other.x)) <= eps and abs(np.float32(self.y - other.y)) <= eps

    def rotate_around_axis(self, axis: "Vector2", angle_radians: float) -> "Vector2":
        """
        Rotate this vector around given axis by angle (radians).
        Uses Rodrigues' rotation formula. Axis does not need to be normalized.
        """
        k = axis.normalized()
        v = self
        cos_t = np.cos(angle_radians)
        sin_t = np.sin(angle_radians)

        term1 = v * cos_t
        term2 = k.cross(v) * sin_t
        term3 = k * (k.dot(v) * (1.0 - cos_t))
        return Vector3(term1.x + term2.x + term3.x, term1.y + term2.y + term3.y)

    def to_tuple(self) -> Tuple[float, float]:
        return (np.float32(self.x), np.float32(self.y))


@dataclass
class Vector3:
    x: np.float32 = 0.0
    y: np.float32 = 0.0
    z: np.float32 = 0.0

    @classmethod
    def zero(cls) -> "Vector3":
        return cls(0.0, 0.0, 0.0)

    @classmethod
    def one(cls) -> "Vector3":
        return cls(1.0, 1.0, 1.0)

    @classmethod
    def unit_x(cls) -> "Vector3":
        return cls(1.0, 0.0, 0.0)

    @classmethod
    def unit_y(cls) -> "Vector3":
        return cls(0.0, 1.0, 0.0)

    @classmethod
    def unit_z(cls) -> "Vector3":
        return cls(0.0, 0.0, 1.0)

    @classmethod
    def from_iterable(cls, it: Iterable[Number]) -> "Vector3":
        a = np.array(it)
        if len(a) != 3:
            raise ValueError("Iterable must be length 3")
        return cls(a[0], a[1], a[2])

    @classmethod
    def from_xyzw(cls, it: Iterable[Number]) -> "Vector3":
        a = np.array(it)
        if len(a) != 4:
            raise ValueError("Iterable must have length 4")
        if a[3] != 0.0:
            raise ValueError("Not a vector")
        return cls(a[0], a[1], a[2])

    def __repr__(self) -> str:
        return f"Vector3(x={np.float32(self.x):.6f}, y={np.float32(self.y):.6f}, z={np.float32(self.z):.6f})"

    def to_array(self, dtype=np.float32) -> np.ndarray:
        return np.array([self.x, self.y, self.z], dtype=dtype)

    def copy(self) -> "Vector3":
        return Vector3(self.x, self.y, self.z)

    def _coerce_operand(self, other):
        if isinstance(other, (Vector3, Point3)):
            return other
        if isinstance(other, (int, float, np.number)):
            return Vector3(np.float32(other), np.float32(other), np.float32(other))
        raise TypeError(f"Unsupported operand type: {type(other)}")

    def __add__(self, other):
        o = self._coerce_operand(other)
        return Vector3(self.x + o.x, self.y + o.y, self.z + o.z)

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        o = self._coerce_operand(other)
        return Vector3(self.x - o.x, self.y - o.y, self.z - o.z)

    def __rsub__(self, other):
        o = self._coerce_operand(other)
        return Vector3(o.x - self.x, o.y - self.y, o.z - self.z)

    def __neg__(self):
        return Vector3(-self.x, -self.y, -self.z)

    def __mul__(self, other):
        if isinstance(other, (int, float, np.number)):
            s = np.float32(other)
            return Vector3(self.x * s, self.y * s, self.z * s)
        if isinstance(other, Vector3):
            return Vector3(self.x * other.x, self.y * other.y, self.z * other.z)
        return NotImplemented

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        if isinstance(other, (int, float, np.number)):
            s = np.float32(other)
            if s == 0:
                raise ZeroDivisionError("division by zero")
            return Vector3(self.x / s, self.y / s, self.z / s)
        if isinstance(other, Vector3):
            return Vector3(self.x / other.x, self.y / other.y, self.z / other.z)
        return NotImplemented

    def __iadd__(self, other):
        o = self._coerce_operand(other)
        self.x += o.x
        self.y += o.y
        self.z += o.z
        return self

    def __isub__(self, other):
        o = self._coerce_operand(other)
        self.x -= o.x
        self.y -= o.y
        self.z -= o.z
        return self

    def __imul__(self, other):
        if isinstance(other, (int, float, np.number)):
            s = np.float32(other)
            self.x *= s
            self.y *= s
            self.z *= s
            return self
        if isinstance(other, Vector3):
            self.x *= other.x
            self.y *= other.y
            self.z *= other.z
            return self
        return NotImplemented

    def __itruediv__(self, other):
        if isinstance(other, (int, float, np.number)):
            s = np.float32(other)
            if s == 0:
                raise ZeroDivisionError("division by zero")
            self.x /= s
            self.y /= s
            self.z /= s
            return self
        if isinstance(other, Vector3):
            self.x /= other.x
            self.y /= other.y
            self.z /= other.z
            return self
        return NotImplemented

    def __eq__(self, other) -> bool:
        if not isinstance(other, Vector3):
            return False
        return bool((self.x == other.x) and (self.y == other.y) and (self.z == other.z))

    def __iter__(self):
        yield np.float32(self.x)
        yield np.float32(self.y)
        yield np.float32(self.z)

    def __getitem__(self, idx: int) -> float:
        if idx == 0:
            return np.float32(self.x)
        if idx == 1:
            return np.float32(self.y)
        if idx == 2:
            return np.float32(self.z)
        raise IndexError("Index out of range for Vector3")

    def dot(self, other: "Vector3") -> np.float32:
        return np.float32(self.x * other.x + self.y * other.y + self.z * other.z)

    def cross(self, other: "Vector3") -> "Vector3":
        cx = self.y * other.z - self.z * other.y
        cy = self.z * other.x - self.x * other.z
        cz = self.x * other.y - self.y * other.x
        return Vector3(cx, cy, cz)

    @property
    def sqr_magnitude(self) -> np.float32:
        return np.float32(self.x * self.x + self.y * self.y + self.z * self.z)

    @property
    def magnitude(self) -> np.float32:
        return np.float32(np.sqrt(np.float32(self.sqr_magnitude)))

    def normalized(self) -> "Vector3":
        mag = np.float32(self.magnitude)
        if mag == 0.0:
            return Vector3.zero()
        inv = np.float32(1.0 / mag)
        return Vector3(self.x * inv, self.y * inv, self.z * inv)

    def normalize(self) -> "Vector3":
        """In-place normalization. Returns self."""
        mag = np.float32(self.magnitude)
        if mag == 0:
            self.x = self.y = self.z = np.float32(0.0)
            return self
        inv = np.float32(1.0 / mag)
        self.x *= inv
        self.y *= inv
        self.z *= inv
        return self

    def distance_to(self, other: "Vector3") -> np.float32:
        return (self - other).magnitude

    def angle_to(self, other: "Vector3") -> float:
        """Return angle in radians between self and other."""
        denom = np.float32(self.magnitude * other.magnitude)
        if denom == 0:
            raise ValueError("Cannot compute angle with zero-length vector")
        cos_theta = np.float32(self.dot(other)) / denom
        cos_theta = max(-1.0, min(1.0, cos_theta))
        return np.acos(cos_theta)

    def project_onto(self, other: "Vector3") -> "Vector3":
        """Project self onto 'other' (other may be non-normalized)."""
        denom = np.float32(other.sqr_magnitude)
        if denom == 0:
            return Vector3.zero()
        factor = np.float32(self.dot(other)) / denom
        return Vector3(
            other.x * np.float32(factor), other.y * np.float32(factor), other.z * np.float32(factor)
        )

    def reflect(self, normal: "Vector3") -> "Vector3":
        """Reflect this vector across a normal. Normal is expected to be normalized."""
        n = normal.normalized()
        two_dot = np.float32(2.0) * self.dot(n)
        return Vector3(self.x - two_dot * n.x, self.y - two_dot * n.y, self.z - two_dot * n.z)

    def lerp(self, other: "Vector3", t: float) -> "Vector3":
        t32 = np.float32(t)
        return Vector3(
            self.x + (other.x - self.x) * t32,
            self.y + (other.y - self.y) * t32,
            self.z + (other.z - self.z) * t32,
        )

    def clamp_magnitude(self, max_len: float) -> "Vector3":
        max_len = np.float32(max_len)
        sq = np.float32(self.sqr_magnitude)
        if sq <= max_len * max_len:
            return self.copy()
        return self.normalized() * max_len

    def almost_equal(self, other: "Vector3", eps: float = 1e-6) -> bool:
        return (
            abs(np.float32(self.x - other.x)) <= eps
            and abs(np.float32(self.y - other.y)) <= eps
            and abs(np.float32(self.z - other.z)) <= eps
        )

    def rotate_around_axis(self, axis: "Vector3", angle_radians: np.float32) -> "Vector3":
        """
        Rotate this vector around given axis by angle (radians).
        Uses Rodrigues' rotation formula. Axis does not need to be normalized.
        """
        k = axis.normalized()
        v = self
        cos_t = np.cos(angle_radians)
        sin_t = np.sin(angle_radians)

        term1 = v * cos_t
        term2 = k.cross(v) * sin_t
        term3 = k * (k.dot(v) * (1.0 - cos_t))
        return Vector3(
            term1.x + term2.x + term3.x, term1.y + term2.y + term3.y, term1.z + term2.z + term3.z
        )

    def to_tuple(self) -> Tuple[float, float, float]:
        return (np.float32(self.x), np.float32(self.y), np.float32(self.z))

    def to_xyzw(self) -> np.ndarray:
        return np.array(
            [np.float32(self.x), np.float32(self.y), np.float32(self.z), 0.0], dtype=np.float32
        )


@dataclass
class Point3:
    x: np.float32 = np.float32(0.0)
    y: np.float32 = np.float32(0.0)
    z: np.float32 = np.float32(0.0)

    def __post_init__(self):
        self.x = np.float32(self.x)
        self.y = np.float32(self.y)
        self.z = np.float32(self.z)

    @classmethod
    def origin(cls) -> "Point3":
        return cls(0.0, 0.0, 0.0)

    @classmethod
    def from_iterable(cls, it: Iterable[Number]) -> "Point3":
        a = np.array(it)
        if len(a) != 3:
            raise ValueError("Iterable must have length 3")
        return cls(a[0], a[1], a[2])

    @classmethod
    def from_xyzw(cls, it: Iterable[Number]) -> "Point3":
        a = np.array(it)
        if len(a) != 4:
            raise ValueError("Iterable must have length 4")
        if a[3] != 1.0:
            raise ValueError("Not a point")
        return cls(a[0], a[1], a[2])

    @classmethod
    def from_vector(cls, v: "Vector3") -> "Point3":
        return cls(v.x, v.y, v.z)

    def to_vector(self) -> "Vector3":
        return Vector3(self.x, self.y, self.z)

    def to_tuple(self) -> Tuple[float, float, float]:
        return (np.float32(self.x), np.float32(self.y), np.float32(self.z))

    def to_array(self, dtype=np.float32) -> np.ndarray:
        return np.array([self.x, self.y, self.z], dtype=dtype)

    def to_xyzw(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z, 1.0], dtype=np.float32)

    def copy(self) -> "Point3":
        return Point3(self.x, self.y, self.z)

    def __repr__(self) -> str:
        return (
            f"Point3({np.float32(self.x):.6f}, {np.float32(self.y):.6f}, {np.float32(self.z):.6f})"
        )

    def _coerce_vector(self, other):
        if isinstance(other, Vector3):
            return other
        if isinstance(other, (int, float, np.number)):
            return Vector3(np.float32(other), np.float32(other), np.float32(other))
        raise TypeError(f"Unsupported operand for Point3 and {type(other)}")

    def __add__(self, other):
        """
        Point + Vector -> Point
        Numeric scalar will be interpreted as uniform translation (rarely used).
        """
        if isinstance(other, Vector3) or isinstance(other, (int, float, np.number)):
            v = self._coerce_vector(other)
            return Point3(self.x + v.x, self.y + v.y, self.z + v.z)
        raise TypeError("Can only add a Vector3 (or scalar) to a Point3")

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        """
        Point - Point -> Vector
        Point - Vector -> Point (translate backwards)
        """
        if isinstance(other, Point3):
            return Vector3(self.x - other.x, self.y - other.y, self.z - other.z)
        if isinstance(other, Vector3) or isinstance(other, (int, float, np.number)):
            v = self._coerce_vector(other)
            return Point3(self.x - v.x, self.y - v.y, self.z - v.z)
        raise TypeError("Can subtract a Point3 or Vector3 from a Point3")

    def __mul__(self, other):
        """
        Point * Point -> Point
        Point * Vector -> Point
        """
        if isinstance(other, Point3):
            return Vector3(self.x * other.x, self.y * other.y, self.z * other.z)
        if isinstance(other, Vector3) or isinstance(other, (int, float, np.number)):
            v = self._coerce_vector(other)
            return Point3(self.x * v.x, self.y * v.y, self.z * v.z)
        raise TypeError("Can subtract a Point3 or Vector3 from a Point3")

    def __eq__(self, other) -> bool:
        if not isinstance(other, Point3):
            return False
        return bool((self.x == other.x) and (self.y == other.y) and (self.z == other.z))

    def distance_to(self, other: "Point3") -> np.float32:
        return np.float32(
            ((self.x - other.x) ** 2 + (self.y - other.y) ** 2 + (self.z - other.z) ** 2) ** 0.5
        )

    def distance_squared_to(self, other: "Point3") -> np.float32:
        return np.float32(
            (self.x - other.x) ** 2 + (self.y - other.y) ** 2 + (self.z - other.z) ** 2
        )

    def translate(self, v: "Vector3") -> "Point3":
        """Return a new Point translated by vector v."""
        return Point3(self.x + v.x, self.y + v.y, self.z + v.z)

    def lerp_to(self, other: "Point3", t: float) -> "Point3":
        """Linear interpolate between this point and other by t in [0,1]."""
        t32 = np.float32(t)
        return Point3(
            self.x + (other.x - self.x) * t32,
            self.y + (other.y - self.y) * t32,
            self.z + (other.z - self.z) * t32,
        )

    def almost_equal(self, other: "Point3", eps: float = 1e-6) -> bool:
        return (
            abs(np.float32(self.x - other.x)) <= eps
            and abs(np.float32(self.y - other.y)) <= eps
            and abs(np.float32(self.z - other.z)) <= eps
        )


@dataclass
class Point2:
    x: np.float32 = np.float32(0.0)
    y: np.float32 = np.float32(0.0)

    def __post_init__(self):
        self.x = np.float32(self.x)
        self.y = np.float32(self.y)

    @classmethod
    def origin(cls) -> "Point2":
        return cls(0.0, 0.0)

    @classmethod
    def from_iterable(cls, it: Iterable[Number]) -> "Point2":
        a = np.array(it)
        if len(a) != 2:
            raise ValueError("Iterable must have length 2")
        return cls(a[0], a[1])

    @classmethod
    def from_vector(cls, v: "Vector2") -> "Point2":
        return cls(v.x, v.y)

    def to_vector(self) -> "Vector2":
        return Vector2(self.x, self.y)

    def to_tuple(self) -> Tuple[float, float]:
        return (np.float32(self.x), np.float32(self.y))

    def to_array(self, dtype=np.float32) -> np.ndarray:
        return np.array([self.x, self.y], dtype=dtype)

    def copy(self) -> "Point2":
        return Point2(self.x, self.y)

    def __repr__(self) -> str:
        return f"Point2({np.float32(self.x):.6f}, {np.float32(self.y):.6f})"

    def _coerce_vector(self, other: "Vector2") -> "Vector2":
        if isinstance(other, Vector2):
            return other
        if isinstance(other, (int, float, np.number)):
            return Vector2(np.float32(other), np.float32(other))
        raise TypeError(f"Unsupported operand for Point2 and {type(other)}")

    def __add__(self, other: "Vector2"):
        """
        Point + Vector -> Point
        Numeric scalar interpreted as uniform translation.
        """
        if isinstance(other, Vector2) or isinstance(other, (int, float, np.number)):
            v = self._coerce_vector(other)
            return Point2(self.x + v.x, self.y + v.y)
        raise TypeError("Can only add a Vector2 (or scalar) to a Point2")

    def __radd__(self, other: "Vector2"):
        return self.__add__(other)

    def __sub__(self, other: "Vector2"):
        """
        Point - Point -> Vector2
        Point - Vector -> Point (translate backwards)
        """
        if isinstance(other, Point2):
            return Vector2(self.x - other.x, self.y - other.y)
        if isinstance(other, Vector2) or isinstance(other, (int, float, np.number)):
            v = self._coerce_vector(other)
            return Point2(self.x - v.x, self.y - v.y)
        raise TypeError("Can subtract a Point2 or Vector2 from a Point2")

    def __eq__(self, other) -> bool:
        if not isinstance(other, Point2):
            return False
        return bool((self.x == other.x) and (self.y == other.y))

    def distance_to(self, other: "Point2") -> np.float32:
        return np.float32(((self.x - other.x) ** 2 + (self.y - other.y) ** 2) ** 0.5)

    def distance_squared_to(self, other: "Point2") -> np.float32:
        return np.float32((self.x - other.x) ** 2 + (self.y - other.y) ** 2)

    def translate(self, v: "Vector2") -> "Point2":
        """Return a new Point translated by vector v."""
        return Point2(self.x + v.x, self.y + v.y)

    def lerp_to(self, other: "Point2", t: float) -> "Point2":
        """Linear interpolate between this point and other by t in [0,1]."""
        t32 = np.float32(t)
        return Point2(self.x + (other.x - self.x) * t32, self.y + (other.y - self.y) * t32)

    def almost_equal(self, other: "Point2", eps: float = 1e-6) -> bool:
        return abs(np.float32(self.x - other.x)) <= eps and abs(np.float32(self.y - other.y)) <= eps
