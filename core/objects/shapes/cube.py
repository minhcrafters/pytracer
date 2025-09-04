from typing import Optional
import numpy as np

from core.constants import EPSILON
from core.math.vectors import Point3, Vector3
from core.objects.shapes.shape import Shape
from core.opt.bounds import Bounds
from core.rays.intersection import Intersection
from core.rays.intersections import Intersections


class Cube(Shape):
    def __init__(
        self,
        center: Optional[Point3] = None,
    ):
        super().__init__()
        self.center = center if center is not None else Point3(0, 0, 0)

    @classmethod
    def solid(cls, center: Point3 = Point3(0, 0, 0)):
        return cls(center)

    @classmethod
    def glass(cls, center: Point3 = Point3(0, 0, 0)):
        sphere = cls(center)
        sphere.material.transparency = 1.0
        sphere.material.reflective = 0.9
        sphere.material.ior = 1.5
        return sphere

    def __repr__(self):
        return f"Cube(center={self.center}, transform={self.transform}, material={self.material})"

    def local_intersect(self, ray):
        x_tmin, x_tmax = self._check_axis(
            ray.origin.x, ray.dir.x, self.bounds.minimum.x, self.bounds.maximum.x
        )

        if x_tmin > x_tmax:
            return Intersections()

        y_tmin, y_tmax = self._check_axis(
            ray.origin.y, ray.dir.y, self.bounds.minimum.y, self.bounds.maximum.y
        )

        if y_tmin > y_tmax:
            return Intersections()

        z_tmin, z_tmax = self._check_axis(
            ray.origin.z, ray.dir.z, self.bounds.minimum.z, self.bounds.maximum.z
        )

        if z_tmin > z_tmax:
            return Intersections()

        tmin = max(x_tmin, y_tmin, z_tmin)
        tmax = min(x_tmax, y_tmax, z_tmax)

        if tmin > tmax:
            return Intersections()

        return Intersections([Intersection(tmin, self), Intersection(tmax, self)])

    def local_normal_at(self, local_point: Point3):
        max_c = max(np.abs(local_point.x), np.abs(local_point.y), np.abs(local_point.z))

        if max_c == np.abs(local_point.x):
            return Vector3(local_point.x, 0, 0)

        if max_c == np.abs(local_point.y):
            return Vector3(0, local_point.y, 0)

        return Vector3(0, 0, local_point.z)

    @property
    def bounds(self):
        return Bounds(Point3(-1, -1, -1), Point3(1, 1, 1))
