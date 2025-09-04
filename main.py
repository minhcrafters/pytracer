from dataclasses import dataclass

import numpy as np

from core.canvas import Canvas
from core.color import Color
from core.lights.light import Light
from core.lights.point_light import PointLight
from core.materials.material import Material
from core.objects.shapes.cube import Cube
from core.objects.shapes.cylinder import Cylinder
from core.patterns.checkered import CheckeredPattern
from core.patterns.striped import StripedPattern
from core.math.matrices import Matrix4
from core.math.vectors import Point3, Vector2, Vector3
from core.objects.camera import Camera
from core.objects.shapes.plane import Plane
from core.objects.shapes.sphere import Sphere
from core.rays.ray import Ray
from core.scene import Scene
from render_preview import render_scene_with_preview


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


def draw_clock():
    size = Vector2(64, 64)

    c = Canvas(*size.to_array().tolist())

    up = Vector3(0, 1, 0).normalize() * (3 / 8) * 64

    for hour in range(12):
        rot = Matrix4.rotation_z(hour * (np.pi / 6))

        mat = up.to_xyzw() @ rot[:].T

        c.set_pixel(int(size.x / 2 + mat[0]), int(size.y / 2 - mat[1]), (1, 1, 1))

    c.save("tests/clock.png")


def draw_sphere():
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

    c.save("tests/sphere.png")


def draw_sphere_shaded():
    size = Vector2(100, 100)
    c = Canvas(*size.to_array().tolist())

    sphere = Sphere(0, center=Point3(0, 0, 0), radius=1.0)
    sphere.material.color = Color(1, 0.2, 1)

    light_pos = Point3(-10, 10, -10)
    light_color = Color(1, 1, 1)
    light = PointLight(light_pos, light_color)

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

            ray = Ray(ray_origin, (pos - ray_origin).normalize(), wall_z * 10)
            hit = Ray.hit(ray, sphere)

            if hit:
                point = ray.get_position(hit[0].t)
                normal = sphere.normal_at(point)
                eye = -ray.dir

                color = sphere.material.lit(sphere, light, point, eye, normal, False)

                c.set_pixel(x, y, color)

    c.save("tests/sphere_shaded.png")


def draw_example_scene():
    scene = Scene()
    scene.light = PointLight(Point3(-10, 10, -10), Color(1, 1, 1))

    floor = Plane()
    floor.material.pattern = CheckeredPattern(b_color=Color(0.5, 0.5, 0.5))
    scene.add_object(floor)

    left_wall = Plane()
    left_wall.material.pattern = CheckeredPattern(b_color=Color(0.5, 0.5, 0.5))
    left_wall.transform = (
        Matrix4.identity()
        .translate(Vector3(0, 0, 5))
        .rotate_along_y(-np.pi / 4)
        .rotate_along_x(np.pi / 2)
    )
    scene.add_object(left_wall)

    right_wall = Plane()
    right_wall.material.pattern = CheckeredPattern(b_color=Color(0.5, 0.5, 0.5))
    right_wall.transform = (
        Matrix4.identity()
        .translate(Vector3(0, 0, 5))
        .rotate_along_y(np.pi / 4)
        .rotate_along_x(np.pi / 2)
    )
    scene.add_object(right_wall)

    middle = Cylinder.solid()
    middle.maximum = 1.0
    middle.transform = (
        Matrix4.identity()
        .translate(Vector3(0, 0.5, 0.5))
        .scale(Vector3(0.5, 0.5, 0.5))
        # .rotate_along_y(np.deg2rad(40))
    )
    # middle.material.color = Color(0.1, 1, 0.5)
    middle.material.diffuse = 0.0
    middle.material.specular = 1.0
    middle.material.reflective = 1.0
    # middle.material.ambient = 0.1
    scene.add_object(middle)

    right = Cube()
    right.transform = (
        Matrix4.identity()
        .translate(Vector3(1.5, 0.5, -0.5))
        .scale(Vector3(0.5, 0.5, 0.5))
        .rotate_along_y(np.deg2rad(-60))
    )
    right.material.color = Color(0.5, 1, 0.1)
    right.material.diffuse = 0.7
    right.material.specular = 0.3
    scene.add_object(right)

    left = Sphere.glass()
    left.transform = (
        Matrix4.identity().translate(Vector3(-1.5, 0.33, -0.75)).scale(Vector3(0.33, 0.33, 0.33))
    )
    # left.material.color = Color(1, 0.8, 0.1)
    # left.material.diffuse = 0.7
    # left.material.specular = 0.3
    scene.add_object(left)

    cam = Camera(100, 50, np.deg2rad(60))
    cam.transform = Matrix4.view_transform(Point3(0, 1.5, -5), Point3(0, 1, 0), Vector3(0, 1, 0))

    canvas = render_scene_with_preview(scene, cam, window_width=1000, window_height=500)

    with open("output.ppm", "w") as f:
        f.write(canvas.to_ppm().getvalue())


if __name__ == "__main__":
    draw_example_scene()
