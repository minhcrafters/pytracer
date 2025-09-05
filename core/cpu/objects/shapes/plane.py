import numpy as np

from core.cpu.constants import EPSILON
from core.cpu.math.vectors import Point3, Vector3
from core.cpu.objects.shapes.shape import Shape
from core.cpu.opt.bounds import Bounds
from core.cpu.rays.intersection import Intersection
from core.cpu.rays.intersections import Intersections


class Plane(Shape):
    def __init__(self):
        super().__init__()

    def __repr__(self):
        return f"Plane(transform={self.transform}, material={self.material})"

    def local_intersect(self, ray):
        if np.abs(ray.dir.y) < EPSILON:
            return Intersections()

        t = -ray.origin.y / ray.dir.y
        return Intersections([Intersection(t, self)])

    def local_normal_at(self, local_point: Point3):
        return Vector3(0, 1, 0)

    @property
    def bounds(self):
        return Bounds(Point3(-np.inf, 0, -np.inf), Point3(np.inf, 0, np.inf))
