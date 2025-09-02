from dataclasses import dataclass
from core.color import Color
from core.lights.light import Light
from core.lights.point_light import PointLight
from core.materials.material import Material
from core.math.matrices import Matrix4
from core.math.vectors import Point3
from core.rays.computation import Computation
from core.rays.intersection import Intersection
from core.rays.intersections import Intersections
from core.rays.ray import Ray
from core.objects.shapes.shape import Shape
from core.objects.shapes.sphere import Sphere


@dataclass
class Scene:
    light: Light  # TODO: multiple light sources
    objects: list[Shape]

    @classmethod
    def test_scene(cls):
        light = PointLight(Point3(-10, 10, -10), Color(1, 1, 1))

        s1 = Sphere(1)

        s1.material = Material(color=Color(0.8, 1.0, 0.6), diffuse=0.7, specular=0.2)

        s2 = Sphere(2)

        s2.transform = Matrix4.scaling(0.5, 0.5, 0.5)

        objs = [s1, s2]

        return cls(light, objs)

    def intersect_scene(self, ray: Ray) -> Intersections:
        total_inters: list[Intersection] = []

        for obj in self.objects:
            res = obj.intersect(ray)

            for r in res.intersections:
                total_inters.append(r)

        total_inters = list(filter(lambda x: x.t > 0, total_inters))

        total_inters.sort(key=lambda x: x.t)

        return Intersections(len(total_inters), total_inters)

    def shade_hit(self, comps: Computation) -> Color:
        return Light.lighting(
            comps.object.material, self.light, comps.point, comps.eye, comps.normal
        )

    def color_at(self, ray: Ray) -> Color:
        inters = self.intersect_scene(ray)

        if len(inters.intersections) <= 0:
            return Color(0, 0, 0)

        closest_inter = inters.intersections[0]

        comps = closest_inter.prepare_computations(ray)
        hit = self.shade_hit(comps)

        return hit
