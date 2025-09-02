import colorsys
import re
from dataclasses import dataclass
from typing import Tuple, Union

import numpy as np

from .math.vectors import Point3, Vector3

Number = Union[float, int, np.number]
ColorLike = Union[
    "Color", Tuple[Number, Number, Number], Tuple[Number, Number, Number, Number], np.ndarray
]


HEX_RE = re.compile(r"^#?([0-9a-fA-F]{3,8})$")


@dataclass
class Color(Point3):
    """
    Color inherits from Point3:
      - x maps to r (red), y -> g, z -> b
      - channels stored as normalized floats in [0.0, 1.0]
      - alpha 'a' stored separately as normalized float in [0.0, 1.0]
    This keeps Point3/Vector3 interoperability while offering color utilities.
    """

    a: np.float32 = np.float32(1.0)

    def __post_init__(self):
        # Let Point3 coerce x,y,z to float32
        super().__post_init__()
        # Clamp to [0,1]
        self.x = np.float32(max(0.0, min(1.0, float(self.x))))
        self.y = np.float32(max(0.0, min(1.0, float(self.y))))
        self.z = np.float32(max(0.0, min(1.0, float(self.z))))
        self.a = np.float32(max(0.0, min(1.0, float(self.a))))

    # --- Aliases for clarity ---
    @property
    def r(self) -> np.float32:
        return self.x

    @r.setter
    def r(self, val: Number):
        self.x = np.float32(max(0.0, min(1.0, float(val))))

    @property
    def g(self) -> np.float32:
        return self.y

    @g.setter
    def g(self, val: Number):
        self.y = np.float32(max(0.0, min(1.0, float(val))))

    @property
    def b(self) -> np.float32:
        return self.z

    @b.setter
    def b(self, val: Number):
        self.z = np.float32(max(0.0, min(1.0, float(val))))

    # --- Constructors ---
    @classmethod
    def from_floats(cls, r: float, g: float, b: float, a: float = 1.0) -> "Color":
        return cls(np.float32(r), np.float32(g), np.float32(b), np.float32(a))

    @classmethod
    def from_uint8(cls, r: int, g: int, b: int, a: int = 255) -> "Color":
        return cls(
            np.float32(r / 255.0),
            np.float32(g / 255.0),
            np.float32(b / 255.0),
            np.float32(a / 255.0),
        )

    @classmethod
    def from_hex(cls, hexstr: str) -> "Color":
        m = HEX_RE.match(hexstr.strip())
        if not m:
            raise ValueError(f"Invalid hex color: {hexstr!r}")
        s = m.group(1)
        if len(s) in (3, 4):
            parts = [c * 2 for c in s]
            if len(s) == 3:
                r, g, b = parts
                a = "ff"
            else:
                r, g, b, a = parts
        elif len(s) in (6, 8):
            if len(s) == 6:
                r, g, b = s[0:2], s[2:4], s[4:6]
                a = "ff"
            else:
                r, g, b, a = s[0:2], s[2:4], s[4:6], s[6:8]
        else:
            raise ValueError(f"Invalid hex length in {hexstr!r}")
        return cls.from_uint8(int(r, 16), int(g, 16), int(b, 16), int(a, 16))

    # --- Representations ---
    def to_tuple(self) -> Tuple[float, float, float, float]:
        return (float(self.r), float(self.g), float(self.b), float(self.a))

    def to_uint8_tuple(self) -> Tuple[int, int, int, int]:
        return (
            int(round(self.r * 255.0)),
            int(round(self.g * 255.0)),
            int(round(self.b * 255.0)),
            int(round(self.a * 255.0)),
        )

    def to_hex(self, include_alpha: bool = False) -> str:
        r, g, b, a = self.to_uint8_tuple()
        if include_alpha:
            return "#{:02x}{:02x}{:02x}{:02x}".format(r, g, b, a)
        return "#{:02x}{:02x}{:02x}".format(r, g, b)

    def to_array(self, dtype=np.float32) -> np.ndarray:
        return np.array([self.r, self.g, self.b, self.a], dtype=dtype)

    def __repr__(self) -> str:
        r, g, b, a = self.to_tuple()
        return f"Color(r={r:.3f}, g={g:.3f}, b={b:.3f}, a={a:.3f})"

    def __eq__(self, other):
        if not isinstance(other, Color):
            return False
        return bool(
            np.isclose(self.x, other.x)
            and np.isclose(self.y, other.y)
            and np.isclose(self.z, other.z)
            and np.isclose(self.a, other.a)
        )

    # --- Color operations (work in normalized float space) ---
    def with_alpha(self, alpha: Number) -> "Color":
        return Color(self.r, self.g, self.b, np.float32(max(0.0, min(1.0, float(alpha)))))

    def invert(self) -> "Color":
        return Color(1.0 - self.r, 1.0 - self.g, 1.0 - self.b, self.a)

    def grayscale(self, method: str = "luminosity") -> "Color":
        if method == "average":
            lum = (self.r + self.g + self.b) / 3.0
        else:
            lum = 0.2126 * self.r + 0.7152 * self.g + 0.0722 * self.b
        return Color(lum, lum, lum, self.a)

    def lighten(self, factor: float) -> "Color":
        return Color(
            min(1.0, self.r * factor), min(1.0, self.g * factor), min(1.0, self.b * factor), self.a
        )

    def lerp(self, other: "Color", t: float) -> "Color":
        t = max(0.0, min(1.0, float(t)))
        return Color(
            self.r + (other.r - self.r) * t,
            self.g + (other.g - self.g) * t,
            self.b + (other.b - self.b) * t,
            self.a + (other.a - self.a) * t,
        )

    # Alpha compositing (source over dest), done in linear space would be more correct,
    # but this operates in normalized sRGB for convenience.
    def blend_over(self, dest: "Color") -> "Color":
        sr, sg, sb, sa = self.to_tuple()
        dr, dg, db, da = dest.to_tuple()
        out_a = sa + da * (1.0 - sa)
        if out_a == 0.0:
            return Color(0.0, 0.0, 0.0, 0.0)
        out_r = (sr * sa + dr * da * (1.0 - sa)) / out_a
        out_g = (sg * sa + dg * da * (1.0 - sa)) / out_a
        out_b = (sb * sa + db * da * (1.0 - sa)) / out_a
        return Color(out_r, out_g, out_b, out_a)

    # HSV helpers
    def to_hsv(self) -> Tuple[float, float, float, float]:
        h, s, v = colorsys.rgb_to_hsv(self.r, self.g, self.b)
        return (h, s, v, self.a)

    def with_hsv(
        self, h: float = None, s: float = None, v: float = None, a: float = None
    ) -> "Color":
        cur_h, cur_s, cur_v, cur_a = self.to_hsv()
        new_h = cur_h if h is None else float(h) % 1.0
        new_s = cur_s if s is None else max(0.0, min(1.0, float(s)))
        new_v = cur_v if v is None else max(0.0, min(1.0, float(v)))
        new_a = cur_a if a is None else max(0.0, min(1.0, float(a)))
        r, g, b = colorsys.hsv_to_rgb(new_h, new_s, new_v)
        return Color(r, g, b, new_a)

    # Convenience: convert to a Vector3 (useful when mixing with Vector3 math)
    def to_vector3(self) -> "Vector3":
        return Vector3(self.r, self.g, self.b)
