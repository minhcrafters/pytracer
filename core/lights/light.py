from dataclasses import dataclass

from core.color import Color
from core.materials.material import Material
from core.math.vectors import Point3, Vector3
from core.objects.shapes.shape import Shape
from core.rays.ray import Ray


@dataclass
class Light:
    position: Point3
    intensity: Color

    @classmethod
    def default(cls):
        return cls(Point3(0, 0, 0), Color(1, 1, 1))
