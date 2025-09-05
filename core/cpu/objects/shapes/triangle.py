import numpy as np
from core.cpu.constants import EPSILON
from core.cpu.math.vectors import Point3
from core.cpu.objects.shapes.shape import Shape
from core.cpu.opt.bounds import Bounds
from core.cpu.rays.intersection import Intersection
from core.cpu.rays.intersections import Intersections


class Triangle(Shape):
    def __init__(self, p1: Point3, p2: Point3, p3: Point3):
        super().__init__()

        self.p1 = p1
        self.p2 = p2
        self.p3 = p3

        self.e1 = p2 - p1
        self.e2 = p3 - p1

        self.normal = self.e2.cross(self.e1).normalized()

    def __repr__(self):
        return f"Triangle(p1={self.p1}, p2={self.p2}, p3={self.p3})"

    def local_intersect(self, ray):
        dir_cross_e2 = ray.dir.cross(self.e2)
        det = self.e1.dot(dir_cross_e2)

        if np.abs(det) < EPSILON:
            return Intersections()

        f = 1.0 / det

        p1_to_origin = ray.origin - self.p1
        u = f * p1_to_origin.dot(dir_cross_e2)

        if u < 0 or u > 1:
            return Intersections()

        origin_cross_e1 = p1_to_origin.cross(self.e1)
        v = f * ray.dir.dot(origin_cross_e1)

        if v < 0 or u + v > 1:
            return Intersections()

        t = f * self.e2.dot(origin_cross_e1)

        return Intersections([Intersection(t, self)])

    def local_normal_at(self, point):
        return self.normal

    @property
    def bounds(self):
        xs = (self.p1.x, self.p2.x, self.p3.x)
        ys = (self.p1.y, self.p2.y, self.p3.y)
        zs = (self.p1.z, self.p2.z, self.p3.z)

        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        min_z, max_z = min(zs), max(zs)

        if abs(max_x - min_x) < EPSILON:
            min_x -= EPSILON * 0.5
            max_x += EPSILON * 0.5
        if abs(max_y - min_y) < EPSILON:
            min_y -= EPSILON * 0.5
            max_y += EPSILON * 0.5
        if abs(max_z - min_z) < EPSILON:
            min_z -= EPSILON * 0.5
            max_z += EPSILON * 0.5

        return Bounds(minimum=Point3(min_x, min_y, min_z), maximum=Point3(max_x, max_y, max_z))
