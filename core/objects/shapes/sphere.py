from typing import Optional
import numpy as np

from core.math.vectors import Point3
from core.rays.intersection import Intersection
from core.rays.intersections import Intersections
from core.objects.shapes.shape import Shape


class Sphere(Shape):
    def __init__(
        self,
        center: Optional[Point3] = None,
        radius: np.float32 = 1.0,
    ):
        super().__init__()

        self.center = center if center is not None else Point3(0, 0, 0)
        self.radius = radius

    def __repr__(self):
        return f"Sphere(center={self.center}, radius={self.radius}, transform={self.transform}, material={self.material})"

    def _intersect(self, ray):
        # quadratic: |O + tD|^2 - r^2 = 0
        L = ray.origin - self.center
        a = ray.dir.dot(ray.dir)
        b = 2 * ray.dir.dot(L)
        c = L.dot(L) - (self.radius**2)
        det = b**2 - 4 * a * c

        intersections: list[Intersection] = []
        count = 0

        if det > 0:
            intersections.append(Intersection((-b + np.sqrt(det)) / (2 * a), self))
            intersections.append(Intersection((-b - np.sqrt(det)) / (2 * a), self))
            count = 2
        elif det == 0:
            intersections.append(Intersection(-b / (2 * a), self))
            intersections.append(Intersection(-b / (2 * a), self))
            count = 2
        elif det < 0:
            pass

        intersections.sort(key=lambda x: x.t)

        return Intersections(count, intersections)

    def _normal_at(self, local_point: Point3):
        return local_point - self.center
