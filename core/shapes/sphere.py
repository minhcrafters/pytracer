import numpy as np

from core.rays.ray import Ray

from ..math.vectors import Point3, Vector3
from ..rays.intersection import Intersection
from ..rays.intersections import Intersections
from .shape import Shape


class Sphere(Shape):
    def __init__(
        self,
        id: int,
        center: Point3 = Point3(0, 0, 0),
        radius: np.float32 = 1.0,
    ):
        super().__init__(id)

        self.id = id
        self.center = center
        self.radius = radius

    def __repr__(self):
        return f"Sphere(id={self.id}, center={self.center}, radius={self.radius}, transform={self.transform})"

    def intersect(self, ray):
        ray = Ray.transform(ray, self.transform.inverse())

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

    def normal_at(self, point: Point3):
        world_point = point
        obj_point = Point3.from_xyzw(self.transform.inverse()[:] @ world_point.to_xyzw())
        obj_normal = obj_point - self.center
        world_mat = self.transform.inverse().transpose()[:] @ obj_normal.to_xyzw()
        world_mat[3] = 0
        world_normal = Vector3.from_xyzw(world_mat)
        return world_normal.normalize()
