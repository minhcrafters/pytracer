from dataclasses import dataclass

import numpy as np

from core.math.matrices import Matrix4
from core.math.vectors import Point3, Vector3
from core.rays.intersection import Intersection
from core.objects.shapes.shape import Shape


@dataclass
class Ray:
    origin: Point3
    dir: Vector3
    t: np.float32 = np.inf

    def get_position(self, t: np.float32) -> Point3:
        pos = self.origin + self.dir * t
        return pos

    @staticmethod
    def hit(ray: "Ray", obj: Shape) -> list[Intersection]:
        inters = obj.intersect(ray).intersections

        inters = list(filter(lambda x: x.t > 0, inters))

        inters.sort(key=lambda x: x.t)

        return inters if inters else None

    @staticmethod
    def transform(ray: "Ray", mat: Matrix4) -> "Ray":
        origin = ray.origin.to_xyzw() @ mat[:].T
        dir = ray.dir.to_xyzw() @ mat[:].T

        return Ray(Point3.from_xyzw(origin), Vector3.from_xyzw(dir), ray.t)
