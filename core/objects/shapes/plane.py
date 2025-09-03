import numpy as np

from core.constants import EPSILON
from core.math.vectors import Point3, Vector3
from core.objects.shapes.shape import Shape
from core.rays.intersection import Intersection
from core.rays.intersections import Intersections


class Plane(Shape):
    def __init__(self):
        super().__init__()

    def __repr__(self):
        return f"Plane(transform={self.transform}, material={self.material})"

    def _intersect(self, ray):
        if np.abs(ray.dir.y) < EPSILON:
            return Intersections(0, [])

        t = -ray.origin.y / ray.dir.y
        return Intersections(1, [Intersection(t, self)])

    def _normal_at(self, local_point: Point3):
        return Vector3(0, 1, 0)
