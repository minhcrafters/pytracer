import time

import numpy as np

from core.canvas import Canvas
from core.color import Color
from core.lights.light import Light
from core.lights.point_light import PointLight
from core.materials.material import Material
from core.math.matrices import Matrix4
from core.math.vectors import Point3
from core.objects.camera import Camera
from core.objects.shapes.shape import Shape
from core.objects.shapes.sphere import Sphere
from core.rays.computation import Computation
from core.rays.intersection import Intersection
from core.rays.intersections import Intersections
from core.rays.ray import Ray


class Scene:
    def __init__(self, objects: list[Shape] = [], light: Light = Light.default()):
        self.objects = objects
        self.light = light

    @classmethod
    def test_scene(cls):
        light = PointLight(Point3(-10, 10, -10), Color(1, 1, 1))

        s1 = Sphere()

        s1.material = Material(color=Color(0.8, 1.0, 0.6), diffuse=0.7, specular=0.2)

        s2 = Sphere()

        s2.transform = Matrix4.scaling(0.5, 0.5, 0.5)

        objs = [s1, s2]

        return cls(objs, light)

    def add_object(self, obj: Shape) -> None:
        self.objects.append(obj)

    # def set_global_light(self, light: Light) -> None:
    #     self.light = light

    def intersect_scene(self, ray: Ray) -> Intersections:
        total_inters: list[Intersection] = []

        for obj in self.objects:
            res = obj.intersect(ray)

            # broken shape returned
            if isinstance(res, list):
                continue

            total_inters.extend(res.intersections)

        total_inters = list(filter(lambda x: x.t > 0, total_inters))

        total_inters.sort(key=lambda x: x.t)

        return Intersections(total_inters)

    def shade_hit(self, comps: Computation, bounce_limit: int = 4) -> Color:
        is_shadowed = comps.cast_shadows and self.is_shadowed(comps.over_point)
        surface = comps.object.material.lit(
            comps.object,
            self.light,
            comps.over_point,
            comps.eye,
            comps.normal,
            is_shadowed,
        )
        reflected = self.reflected_color(comps, bounce_limit)
        refracted = self.refracted_color(comps, bounce_limit)

        material = comps.object.material

        if material.reflective > 0 and material.transparency > 0:
            reflectance = comps.compute_fresnel()

            return Color.from_vector(
                surface + reflected * reflectance + refracted * (1 - reflectance)
            )

        return Color.from_vector(surface + reflected + refracted)

    def color_at(self, ray: Ray, bounce_limit: int = 4) -> Color:
        inters = self.intersect_scene(ray)

        # find the hit from the resulting intersections
        if inters.count > 0:
            hit = inters.intersections[0]
        else:
            return Color(0, 0, 0)

        comps = hit.prepare_computations(ray, inters)
        color = self.shade_hit(comps, bounce_limit)

        return color

    def is_shadowed(self, point: Point3) -> bool:
        v = self.light.position - point
        dist = v.magnitude
        dir = v.normalized()

        ray = Ray(point, dir)

        inters = self.intersect_scene(ray)

        if inters.count > 0:
            hit = inters.intersections[0]
        else:
            return False

        if hit and hit.t < dist:
            return True

        return False

    def reflected_color(self, comps: Computation, bounce_limit: int = 4) -> Color:
        if comps.object.material.reflective == 0.0:
            return Color(0, 0, 0)

        if bounce_limit <= 0:
            return Color(0, 0, 0)

        reflect_ray = Ray(comps.over_point, comps.reflect)
        color = self.color_at(reflect_ray, bounce_limit - 1)

        return color * comps.object.material.reflective

    def refracted_color(self, comps: Computation, bounce_limit: int = 4) -> Color:
        if comps.object.material.transparency == 0.0:
            return Color(0, 0, 0)

        if bounce_limit <= 0:
            return Color(0, 0, 0)

        n_ratio = comps.n1 / comps.n2
        cos_i = comps.eye.dot(comps.normal)
        sin2_t = n_ratio**2 * (1 - cos_i**2)

        if sin2_t > 1.0:
            # Total internal reflection
            return Color(0, 0, 0)

        cos_t = np.sqrt(1 - sin2_t)
        dir = comps.normal * (n_ratio * cos_i - cos_t) - comps.eye * n_ratio
        refract_ray = Ray(comps.under_point, dir)

        color = self.color_at(refract_ray, bounce_limit - 1) * comps.object.material.transparency

        return color

    def render(self, camera: Camera) -> Canvas:
        canvas = Canvas(camera.hsize, camera.vsize)
        total_pixels = camera.hsize * camera.vsize
        pixel_count = 0

        start_time = time.time()

        for y, x in np.ndindex(camera.vsize, camera.hsize):
            ray = camera.ray_from_pixel(x, y)
            color = self.color_at(ray)
            canvas.set_pixel(x, y, color)

            pixel_count += 1
            percentage = (pixel_count / total_pixels) * 100
            print(f"Rendering: {percentage:.2f}%", end="\r")

        print()

        time_elapsed = time.time() - start_time

        print(f"Took {time_elapsed:.2f} seconds")

        return canvas
