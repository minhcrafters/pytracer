from typing import Optional
import numpy as np

from core.cpu.color import Color
from core.cpu.math.vectors import Point3
from core.cpu.opt.bounds import Bounds
from core.cpu.rays.intersection import Intersection
from core.cpu.rays.intersections import Intersections
from core.cpu.objects.shapes.shape import Shape


class Sphere(Shape):
    def __init__(
        self,
        center: Optional[Point3] = None,
        radius: np.float32 = 1.0,
    ):
        super().__init__()

        self.center = center if center is not None else Point3(0, 0, 0)
        self.radius = radius

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
        return f"Sphere(center={self.center}, radius={self.radius}, transform={self.transform}, material={self.material})"

    def local_intersect(self, ray):
        # quadratic: |O + tD|^2 - r^2 = 0
        L = ray.origin - self.center
        a = ray.dir.dot(ray.dir)
        b = 2 * ray.dir.dot(L)
        c = L.dot(L) - (self.radius**2)
        det = b**2 - 4 * a * c

        intersections: list[Intersection] = []

        if det > 0:
            intersections.append(Intersection((-b + np.sqrt(det)) / (2 * a), self))
            intersections.append(Intersection((-b - np.sqrt(det)) / (2 * a), self))
        elif det == 0:
            intersections.append(Intersection(-b / (2 * a), self))
            intersections.append(Intersection(-b / (2 * a), self))
        elif det < 0:
            pass

        intersections.sort(key=lambda x: x.t)

        return Intersections(intersections)

    def local_normal_at(self, local_point: Point3):
        return local_point - self.center

    @property
    def bounds(self):
        return Bounds(Point3(-1, -1, -1), Point3(1, 1, 1))
