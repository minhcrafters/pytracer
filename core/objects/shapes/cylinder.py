from typing import Optional
import numpy as np

from core.color import Color
from core.constants import EPSILON
from core.math.vectors import Point3, Vector3
from core.opt.bounds import Bounds
from core.rays.intersection import Intersection
from core.rays.intersections import Intersections
from core.objects.shapes.shape import Shape
from core.rays.ray import Ray


class Cylinder(Shape):
    def __init__(
        self,
        center: Optional[Point3] = None,
        radius: np.float32 = 1.0,
        minimum: np.float32 = -np.inf,
        maximum: np.float32 = np.inf,
        closed: bool = False,
    ):
        super().__init__()

        self.center = center if center is not None else Point3(0, 0, 0)
        self.radius = radius
        self.minimum = minimum
        self.maximum = maximum
        self.closed = closed

    @classmethod
    def solid(cls, center: Point3 = Point3(0, 0, 0), radius: int = 1.0):
        return cls(center, radius)

    @classmethod
    def glass(cls, center: Point3 = Point3(0, 0, 0), radius: int = 1.0):
        sphere = cls(center, radius)
        sphere.material.color = Color(0, 0, 0)
        sphere.material.transparency = 1.0
        sphere.material.reflective = 0.9
        sphere.material.ior = 1.5
        return sphere

    def __repr__(self):
        return f"Cylinder(center={self.center}, radius={self.radius}, transform={self.transform}, material={self.material})"

    def local_intersect(self, ray):
        a = ray.dir.x**2 + ray.dir.z**2

        if np.isclose(a, 0.0):
            # self._intersect_caps(ray, inters)
            return Intersections()

        b = 2 * ray.origin.x * ray.dir.x + 2 * ray.origin.z * ray.dir.z
        c = ray.origin.x**2 + ray.origin.z**2 - (self.radius**2)

        disc = b**2 - 4 * a * c

        if disc < 0:
            return Intersections()

        t0 = (-b - np.sqrt(disc)) / (2 * a)
        t1 = (-b + np.sqrt(disc)) / (2 * a)

        if t0 > t1:
            t0, t1 = t1, t0

        inters = Intersections()

        y0 = ray.origin.y + ray.dir.y * t0
        if self.minimum < y0 < self.maximum:
            inters.add(Intersection(t0, self))

        y1 = ray.origin.y + ray.dir.y * t1
        if self.minimum < y1 < self.maximum:
            inters.add(Intersection(t1, self))

        self._intersect_caps(ray, inters)

        return inters

    def local_normal_at(self, local_point: Point3):
        dist = local_point.x**2 + local_point.z**2

        if dist < 1 and local_point.y >= self.maximum - EPSILON:
            return Vector3(0, 1, 0)
        if dist < 1 and local_point.y <= self.minimum - EPSILON:
            return Vector3(0, -1, 0)

        return Vector3(local_point.x, 0, local_point.z)

    @property
    def bounds(self):
        if self.closed:
            return Bounds(Point3(-1, self.minimum, -1), Point3(1, self.maximum, 1))

        return Bounds(Point3(-1, -np.inf, -1), Point3(1, np.inf, 1))

    def _check_cap(self, ray: Ray, t: np.float32) -> bool:
        x = ray.origin.x + ray.dir.x * t
        z = ray.origin.z + ray.dir.z * t

        return (x**2 + z**2) <= self.radius**2

    def _intersect_caps(self, ray: Ray, inters: Intersections) -> None:
        if not self.closed or np.isclose(ray.dir.y, 0.0):
            return

        # check for an intersection with the lower end cap by intersecting
        # the ray with the plane at y = cyl.minimum
        t = (self.minimum - ray.origin.y) / ray.dir.y

        if self._check_cap(ray, t):
            inters.add(Intersection(t, self))

        t = (self.maximum - ray.origin.y) / ray.dir.y

        if self._check_cap(ray, t):
            inters.add(Intersection(t, self))
