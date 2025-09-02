from dataclasses import dataclass

import numpy as np
from tqdm import tqdm

from core.canvas import Canvas
from core.color import Color
from core.lights.light import Light
from core.lights.point_light import PointLight
from core.math.matrices import Matrix2, Matrix4
from core.math.vectors import Point3, Vector2, Vector3
from core.rays.ray import Ray
from core.shapes.sphere import Sphere


@dataclass
class Projectile:
    position: Point3
    velocity: Vector3


@dataclass
class Environment:
    gravity: Vector3
    wind: Vector3


def tick(env: Environment, proj: Projectile):
    position = proj.position + proj.velocity
    velocity = proj.velocity + env.gravity + env.wind

    return Projectile(position, velocity)


def main():
    p = Projectile(Vector3(0, 1, 0), Vector3(1, 1.5, 0).normalize() * 12.25)
    e = Environment(Vector3(0, -0.1, 0), Vector3(-0.01, 0, 0))

    size = Vector2(900, 550)

    c = Canvas(*size.to_array().tolist())

    ticks = 0

    while p.position.y > 0:
        p = tick(e, p)
        print(p.position)
        c.set_pixel(int(p.position.x), 550 - int(p.position.y), (1, 0, 0))
        ticks += 1

    print(f"Ticks: {ticks}")

    ppm = c.to_ppm()

    with open("output.ppm", "w") as f:
        f.write(ppm.getvalue())


def test_matrices():
    mat = Matrix2(np.array([[1, 2], [6, 5]]))

    print(mat.determinant())
    print(mat.inverse())


def test_clock():
    size = Vector2(64, 64)

    c = Canvas(*size.to_array().tolist())

    up = Vector3(0, 1, 0).normalize() * (3 / 8) * 64

    for hour in range(12):
        rot = Matrix4.rotation_z(hour * (np.pi / 6))

        mat = up.to_xyzw() @ rot[:].T

        c.set_pixel(size.x / 2 + mat[0], size.y / 2 - mat[1], (1, 1, 1))

    ppm = c.to_ppm()

    with open("output.ppm", "w") as f:
        f.write(ppm.getvalue())


def test_ray_sphere_intersect():
    sphere = Sphere(0, Point3(0, 0, 0), 1.0)
    ray = Ray(Point3(0, 0, 5), Vector3(0, 0, 1), 10)

    intersect = sphere.intersect(ray)

    print(intersect.count)
    print(intersect.intersections)


def test_ray_sphere_hit():
    sphere = Sphere(0, Point3(0, 0, 0), 1.0)
    ray = Ray(Point3(0, 0, -5), Vector3(0, 0, 1), 10)

    print(Ray.hit(ray, sphere))


def test_ray_transform():
    ray = Ray(Point3(1, 2, 3), Vector3(0, 1, 0), 10)

    t_ray = Ray.transform(ray, Matrix4.translation(3, 4, 5))

    print(t_ray.origin)
    print(t_ray.dir)

    s_ray = Ray.transform(ray, Matrix4.scaling(2, 3, 4))

    print(s_ray.origin)
    print(s_ray.dir)


def test_ray_sphere_transform():
    sphere = Sphere(0, center=Point3(0, 0, 0), radius=1.0)
    ray = Ray(Point3(0, 0, -5), Vector3(0, 0, 1), 10)

    sphere.set_transform(Matrix4.scaling(2, 2, 2))

    res = sphere.intersect(ray)

    print(res)


def test_draw_sphere():
    size = Vector2(100, 100)
    c = Canvas(*size.to_array().tolist())

    color = Color(1, 0, 0)

    sphere = Sphere(0, center=Point3(0, 0, 0), radius=1.0)

    sphere.transform = Matrix4.shear(1, 0, 0, 0, 0, 0) @ Matrix4.scaling(0.5, 1, 1)

    ray_origin = Point3(0, 0, -5)
    wall_z = 10.0
    wall_size = 7.0

    px_size = wall_size / size.x
    half = wall_size / 2

    for y in range(size.y):
        world_y = half - px_size * y
        for x in range(size.x):
            world_x = -half + px_size * x

            pos = Point3(world_x, world_y, wall_z)

            ray = Ray(ray_origin, (pos - ray_origin).normalize(), wall_z)

            if Ray.hit(ray, sphere):
                c.set_pixel(x, y, color)

    ppm = c.to_ppm()

    with open("output.ppm", "w") as f:
        f.write(ppm.getvalue())


def test_normal():
    sphere = Sphere(0, center=Point3(0, 0, 0), radius=1.0)

    sphere.transform = Matrix4.scaling(1, 0.5, 1) @ Matrix4.rotation_z(np.pi / 5)

    norm = sphere.normal_at(Point3(0, np.sqrt(2) / 2, -np.sqrt(2) / 2))

    print(norm)


def test_draw_sphere_shaded():
    size = Vector2(100, 100)
    c = Canvas(*size.to_array().tolist())

    sphere = Sphere(0, center=Point3(0, 0, 0), radius=1.0)
    # sphere.transform = Matrix4.shear(0.5, 0, 0, 0, 0, 0) @ Matrix4.scaling(0.5, 1, 1)
    sphere.material.color = Color(1, 0.2, 1)

    light_pos = Point3(-10, 10, -10)
    light_color = Color(1, 1, 1)
    light = PointLight(light_pos, light_color)

    ray_origin = Point3(0, 0, -5)
    wall_z = 10.0
    wall_size = 7.0

    px_size = wall_size / size.x
    half = wall_size / 2

    for y in tqdm(range(size.y)):
        world_y = half - px_size * y
        for x in tqdm(range(size.x)):
            world_x = -half + px_size * x

            pos = Point3(world_x, world_y, wall_z)

            ray = Ray(ray_origin, (pos - ray_origin).normalize(), wall_z * 10)
            hit = Ray.hit(ray, sphere)

            if hit:
                point = ray.get_position(hit[0].t)
                normal = sphere.normal_at(point)
                eye = -ray.dir

                color = Light.lighting(sphere.material, light, point, eye, normal)

                c.set_pixel(x, y, color)

    ppm = c.to_ppm()

    with open("output.ppm", "w") as f:
        f.write(ppm.getvalue())


if __name__ == "__main__":
    # main()
    # test_matrices()
    # test_clock()
    # test_ray_sphere_intersect()
    # test_ray_sphere_hit()
    # test_ray_transform()
    # test_ray_sphere_transform()
    # test_draw_sphere()
    # test_normal()
    test_draw_sphere_shaded()
