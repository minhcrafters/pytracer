from dataclasses import dataclass

from core.color import Color
from core.math.vectors import Point3


@dataclass
class Light:
    position: Point3
    intensity: Color
