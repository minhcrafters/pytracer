from typing import TYPE_CHECKING

from core.materials.material import Material
from core.math.matrices import Matrix4
from core.math.vectors import Point3

if TYPE_CHECKING:
    from core.rays.intersections import Intersections


class Shape:
    def __init__(
        self,
        transform: Matrix4 = Matrix4.identity(),
        material: Material = Material.default(),
    ):
        self.transform = transform
        self.material = material

    def __repr__(self):
        return f"Shape(transform={self.transform})"

    def intersect(self, ray) -> "Intersections":
        return NotImplemented

    def normal_at(self, point: Point3) -> Point3:
        return NotImplemented
