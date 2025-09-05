from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from core.cpu.math.matrices import Matrix4
from core.cpu.math.vectors import Point3, Vector3
from core.cpu.objects.shapes.shape import Shape

if TYPE_CHECKING:
    from core.cpu.rays.intersection import Intersection


@dataclass
class Ray:
    origin: Point3
    dir: Vector3

    def get_position(self, t: np.float32) -> Point3:
        pos = self.origin + self.dir * t
        return pos

    @staticmethod
    def hit(ray: "Ray", obj: Shape) -> list["Intersection"]:
        inters = obj.intersect(ray).intersections

        inters = list(filter(lambda x: x.t > 0, inters))

        inters.sort(key=lambda x: x.t)

        return inters if inters else None

    def transform(self, mat: Matrix4) -> "Ray":
        origin = self.origin.to_xyzw() @ mat[:].T
        dir = self.dir.to_xyzw() @ mat[:].T

        return Ray(Point3.from_xyzw(origin), Vector3.from_xyzw(dir))
