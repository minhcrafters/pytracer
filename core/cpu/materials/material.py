from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from core.cpu.color import Color
from core.cpu.patterns.pattern import Pattern
from core.cpu.math.vectors import Point3, Vector3

if TYPE_CHECKING:
    from core.cpu.lights.light import Light
    from core.cpu.objects.shapes.shape import Shape


@dataclass
class Material:
    color: Color = None
    pattern: Pattern = None
    ambient: np.float32 = 0.1
    diffuse: np.float32 = 0.9
    specular: np.float32 = 0.9
    shininess: np.float32 = 200.0
    reflective: np.float32 = 0.0
    transparency: np.float32 = 0.0
    ior: np.float32 = 1.0

    @classmethod
    def white(cls):
        return cls(Color(1, 1, 1))

    def copy(self):
        return Material(
            self.color,
            self.pattern,
            self.ambient,
            self.diffuse,
            self.specular,
            self.shininess,
            self.reflective,
            self.transparency,
            self.ior,
        )

    def lit(
        self,
        obj: "Shape",
        light: "Light",
        point: Point3,
        eye: Vector3,
        normal: Vector3,
        in_shadow: bool = False,
    ):
        ambient = Color(0, 0, 0)
        diffuse = Color(0, 0, 0)
        specular = Color(0, 0, 0)

        if self.pattern:
            color = self.pattern.at_object(obj, point)
        else:
            color = self.color

        eff_color = color * light.intensity

        light_vec = (light.position - point).normalize()

        ambient = eff_color * self.ambient

        if in_shadow:
            return Color(
                ambient.x,
                ambient.y,
                ambient.z,
            )

        light_dot_normal = light_vec.dot(normal)

        if light_dot_normal >= 0:
            diffuse = eff_color * self.diffuse * light_dot_normal

            reflect_vec = (-light_vec).reflect(normal)
            reflect_dot_eye = reflect_vec.dot(eye)

            if reflect_dot_eye > 0:
                factor = reflect_dot_eye**self.shininess
                specular = light.intensity * self.specular * factor

        return Color(
            ambient.x + diffuse.x + specular.x,
            ambient.y + diffuse.y + specular.y,
            ambient.z + diffuse.z + specular.z,
        )
