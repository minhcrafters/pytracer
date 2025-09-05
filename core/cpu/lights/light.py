from dataclasses import dataclass

from core.cpu.color import Color
from core.cpu.math.vectors import Point3


@dataclass
class Light:
    position: Point3
    intensity: Color

    @classmethod
    def default(cls):
        return cls(Point3(0, 0, 0), Color(1, 1, 1))
