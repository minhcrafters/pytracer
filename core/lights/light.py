from dataclasses import dataclass

from core.color import Color
from core.math.vectors import Point3


@dataclass
class Light:
    position: Point3
    intensity: Color

    @classmethod
    def default(cls):
        return cls(Point3(0, 0, 0), Color(1, 1, 1))
