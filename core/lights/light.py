from dataclasses import dataclass

from core.color import Color
from core.materials.material import Material
from core.math.vectors import Point3, Vector3
from core.rays.ray import Ray


@dataclass
class Light:
    position: Point3
    intensity: Color

    @classmethod
    def default(cls):
        return cls(Point3(0, 0, 0), Color(1, 1, 1))

    @staticmethod
    def lighting(material: Material, light: "Light", point: Point3, eye: Vector3, normal: Vector3):
        ambient = Color(0, 0, 0)
        diffuse = Color(0, 0, 0)
        specular = Color(0, 0, 0)

        eff_color = material.color * light.intensity

        light_vec = (light.position - point).normalize()

        ambient = eff_color * material.ambient

        light_dot_normal = light_vec.dot(normal)

        if light_dot_normal >= 0:
            diffuse = eff_color * material.diffuse * light_dot_normal

            reflect_vec = (-light_vec).reflect(normal)
            reflect_dot_eye = reflect_vec.dot(eye)

            if reflect_dot_eye > 0:
                factor = reflect_dot_eye**material.shininess
                specular = light.intensity * material.specular * factor

        return Color(
            ambient.x + diffuse.x + specular.x,
            ambient.y + diffuse.y + specular.y,
            ambient.z + diffuse.z + specular.z,
        )
