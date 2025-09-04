from dataclasses import dataclass

import numpy as np

from core.canvas import Canvas
from core.color import Color
from core.lights.light import Light
from core.lights.point_light import PointLight
from core.materials.material import Material
from core.objects.shapes.cube import Cube
from core.objects.shapes.cylinder import Cylinder
from core.objects.shapes.group import Group
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


def draw_teapot_scene():
    scene = Scene()
    scene.light = PointLight(Point3(-10, 10, -10), Color(1, 1, 1))

    floor = Plane()
    floor.material.pattern = CheckeredPattern(b_color=Color(0.5, 0.5, 0.5))
    scene.add_object(floor)

    middle = Cylinder.solid()
    middle.maximum = 1.0
    middle.transform = (
        Matrix4.identity().translate(Vector3(0, 1, 0.5)).scale(Vector3(0.5, 0.5, 0.5))
    )
    middle.material.diffuse = 0.0
    middle.material.specular = 1.0
    middle.material.reflective = 1.0
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
    scene.add_object(left)

    cam = Camera(100, 50, np.deg2rad(60))
    cam.transform = Matrix4.view_transform(Point3(0, 1.5, -5), Point3(0, 1, 0), Vector3(0, 1, 0))

    canvas = render_scene_with_preview(scene, cam, window_width=1000, window_height=500)

    with open("output.ppm", "w") as f:
        f.write(canvas.to_ppm().getvalue())


def draw_teapot_scene():
    scene = Scene()
    scene.light = PointLight(Point3(-10, 10, -10), Color(1, 1, 1))

    floor = Plane()
    floor.material.pattern = CheckeredPattern(b_color=Color(0.5, 0.5, 0.5))
    scene.add_object(floor)

    wall = Plane()
    wall.material.pattern = CheckeredPattern(b_color=Color(0.5, 0.5, 0.5))
    wall.transform = Matrix4.identity().translate(Vector3(0, 0, 5)).rotate_along_x(np.deg2rad(90))
    scene.add_object(wall)

    middle = Group.parse_wavefront("teapot.obj")
    middle.transform = (
        Matrix4.identity().translate(Vector3(0, 1, 0.5)).scale(Vector3(0.5, 0.5, 0.5))
    )
    middle.material.color = Color(1, 0.9, 0.9)
    middle.material.ambient = 0.1
    middle.material.diffuse = 0.9
    middle.material.specular = 1.0
    middle.material.reflective = 1.0
    scene.add_object(middle)

    cam = Camera(100, 50, np.deg2rad(60))
    cam.transform = Matrix4.view_transform(Point3(0, 1.5, -5), Point3(0, 1, 0), Vector3(0, 1, 0))

    canvas = render_scene_with_preview(scene, cam, window_width=1000, window_height=500)

    with open("output.ppm", "w") as f:
        f.write(canvas.to_ppm().getvalue())


def draw_cover_image_scene():
    cam = Camera(100, 100, 0.785)
    cam.transform = Matrix4.view_transform(
        Point3(-6, 6, -10), Point3(6, 0, 6), Vector3(-0.45, 1, 0)
    )

    scene = Scene()
    scene.light = PointLight(Point3(50, 100, -50), Color(1, 1, 1))

    white_mat = Material.white()
    white_mat.diffuse = 0.7
    white_mat.ambient = 0.1
    white_mat.specular = 0.0
    white_mat.reflective = 0.1

    blue_mat = white_mat.copy()
    blue_mat.color = Color(0.537, 0.831, 0.914)

    red_mat = white_mat.copy()
    red_mat.color = Color(0.941, 0.322, 0.388)

    purple_mat = white_mat.copy()
    purple_mat.color = Color(0.373, 0.404, 0.550)

    standard_transform = (
        Matrix4.identity().translate(Vector3(1, -1, 1)).scale(Vector3(0.5, 0.5, 0.5))
    )
    large_obj = standard_transform.scale(Vector3(3.5, 3.5, 3.5))
    medium_obj = standard_transform.scale(Vector3(3, 3, 3))
    small_obj = standard_transform.scale(Vector3(2, 2, 2))

    backdrop = Plane()
    backdrop.material.ambient = 1.0
    backdrop.material.diffuse = 0
    backdrop.material.specular = 0
    backdrop.transform = Matrix4.identity().rotate_along_x(np.pi / 2).translate(Vector3(0, 0, 500))
    scene.add_object(backdrop)

    sphere = Sphere.solid()
    sphere.material.color = Color(0.373, 0.404, 0.550)
    sphere.material.diffuse = 0.2
    sphere.material.ambient = 0.0
    sphere.material.specular = 1.0
    sphere.material.shininess = 200.0
    sphere.material.reflective = 0.7
    sphere.material.transparency = 0.7
    sphere.material.ior = 1.5
    sphere.transform = large_obj.copy()
    scene.add_object(sphere)

    cube1 = Cube()
    cube1.material = white_mat
    cube1.transform = medium_obj.translate(Vector3(4, 0, 0))
    scene.add_object(cube1)

    cube2 = Cube()
    cube2.material = blue_mat
    cube2.transform = large_obj.translate(Vector3(8.5, 1.5, -0.5))
    scene.add_object(cube2)

    cube3 = Cube()
    cube3.material = red_mat
    cube3.transform = large_obj.translate(Vector3(0, 0, 4))
    scene.add_object(cube3)

    cube4 = Cube()
    cube4.material = white_mat
    cube4.transform = small_obj.translate(Vector3(4, 0, 4))
    scene.add_object(cube4)

    cube5 = Cube()
    cube5.material = purple_mat
    cube5.transform = medium_obj.translate(Vector3(7.5, 0.5, 4))
    scene.add_object(cube5)

    cube6 = Cube()
    cube6.material = white_mat
    cube6.transform = medium_obj.translate(Vector3(-0.25, 0.25, 8))
    scene.add_object(cube6)

    cube7 = Cube()
    cube7.material = blue_mat
    cube7.transform = large_obj.translate(Vector3(4, 1, 7.5))
    scene.add_object(cube7)

    cube8 = Cube()
    cube8.material = red_mat
    cube8.transform = medium_obj.translate(Vector3(10, 2, 7.5))
    scene.add_object(cube8)

    cube9 = Cube()
    cube9.material = white_mat
    cube9.transform = small_obj.translate(Vector3(8, 2, 12))
    scene.add_object(cube9)

    cube10 = Cube()
    cube10.material = white_mat
    cube10.transform = small_obj.translate(Vector3(20, 1, 9))
    scene.add_object(cube10)

    cube11 = Cube()
    cube11.material = blue_mat
    cube11.transform = large_obj.translate(Vector3(-0.5, -5, 0.25))
    scene.add_object(cube11)

    cube12 = Cube()
    cube12.material = red_mat
    cube12.transform = large_obj.translate(Vector3(4, -4, 0))
    scene.add_object(cube12)

    cube13 = Cube()
    cube13.material = white_mat
    cube13.transform = large_obj.translate(Vector3(8.5, -4, 0))
    scene.add_object(cube13)

    cube14 = Cube()
    cube14.material = white_mat
    cube14.transform = large_obj.translate(Vector3(0, -4, 4))
    scene.add_object(cube14)

    cube15 = Cube()
    cube15.material = purple_mat
    cube15.transform = large_obj.translate(Vector3(-0.5, -4.5, 8))
    scene.add_object(cube15)

    cube16 = Cube()
    cube16.material = white_mat
    cube16.transform = large_obj.translate(Vector3(0, -8, 4))
    scene.add_object(cube16)

    cube17 = Cube()
    cube17.material = white_mat
    cube17.transform = large_obj.translate(Vector3(-0.5, -8.5, 8))
    scene.add_object(cube17)

    canvas = render_scene_with_preview(scene, cam, window_width=500, window_height=500)

    with open("output.ppm", "w") as f:
        f.write(canvas.to_ppm().getvalue())


if __name__ == "__main__":
    draw_teapot_scene()
