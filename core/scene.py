from dataclasses import dataclass
from core.color import Color
from core.lights.light import Light
from core.lights.point_light import PointLight
from core.math.matrices import Matrix4
from core.math.vectors import Point3
from core.rays.intersections import Intersections
from core.rays.ray import Ray
from core.shapes.shape import Shape
from core.shapes.sphere import Sphere


@dataclass
class Scene:
    light: Light
    objects: list[Shape]

    @classmethod
    def default(cls):
        light = PointLight(Point3(-10, 10, -10), Color(1, 1, 1))

        s1 = Sphere(1, Point3(0, 0, 0), 1.0)
        s2 = Sphere(2, Point3(2, 0, 0), 1.0)

        s2.transform = Matrix4.scaling(0.5, 0.5, 0.5)

        objs = [s1, s2]

        return cls(light, objs)

    def intersect_scene(self, ray: Ray):
        total_inters = []

        for obj in self.objects:
            res = obj.intersect(ray)

            for r in res:
                total_inters.append(r)

        return Intersections(len(total_inters), total_inters)
