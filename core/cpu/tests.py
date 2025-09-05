import numpy as np

from core.cpu.color import Color
from core.cpu.constants import EPSILON
from core.cpu.lights.point_light import PointLight
from core.cpu.materials.material import Material
from core.cpu.patterns.checkered import CheckeredPattern
from core.cpu.patterns.gradient import GradientPattern
from core.cpu.patterns.ring import RingPattern
from core.cpu.patterns.striped import StripedPattern
from core.cpu.math.matrices import Matrix2, Matrix4
from core.cpu.math.vectors import Point3, Vector3
from core.cpu.camera import Camera
from core.cpu.objects.shapes.plane import Plane
from core.cpu.objects.shapes.cylinder import Cylinder
from core.cpu.objects.shapes.sphere import Sphere
from core.cpu.rays.intersection import Intersection
from core.cpu.rays.intersections import Intersections
from core.cpu.rays.ray import Ray
from core.cpu.scene import Scene


def test_matrices():
    mat = Matrix2(np.array([[1, 2], [6, 5]]))

    assert mat.determinant() == -7.0
    expected_inv = np.array([[-5 / 7, 2 / 7], [6 / 7, -1 / 7]])
    assert np.allclose(mat.inverse()[:], expected_inv)


def test_ray_sphere_intersect():
    sphere = Sphere()
    ray = Ray(Point3(0, 0, 5), Vector3(0, 0, 1))

    intersect = sphere.intersect(ray)

    assert intersect.count == 2


def test_ray_sphere_hit():
    sphere = Sphere()
    ray = Ray(Point3(0, 0, -5), Vector3(0, 0, 1))

    hit = Ray.hit(ray, sphere)

    assert hit is not None
    assert hit[0].t == 4.0


def test_ray_transform():
    ray = Ray(Point3(1, 2, 3), Vector3(0, 1, 0))

    t_ray = Ray.transform(ray, Matrix4.translation(3, 4, 5))

    assert t_ray.origin == Point3(4, 6, 8)
    assert t_ray.dir == Vector3(0, 1, 0)

    s_ray = Ray.transform(ray, Matrix4.scaling(2, 3, 4))

    assert s_ray.origin == Point3(2, 6, 12)
    assert s_ray.dir == Vector3(0, 3, 0)


def test_ray_sphere_transform():
    sphere = Sphere()
    ray = Ray(Point3(0, 0, -5), Vector3(0, 0, 1))

    sphere.transform = Matrix4.scaling(2, 2, 2)

    res = sphere.intersect(ray)

    assert res.count == 2
    assert res.intersections[0].t == 3.0
    assert res.intersections[1].t == 7.0


def test_scene():
    scene = Scene.test_scene()
    ray = Ray(Point3(0, 0, -5), Vector3(0, 0, 1))

    res = scene.intersect_scene(ray)

    assert res.count == 4
    assert res.intersections[0].t == 4
    assert res.intersections[1].t == 4.5
    assert res.intersections[2].t == 5.5
    assert res.intersections[3].t == 6


def test_prepare_computations():
    ray = Ray(Point3(0, 0, -5), Vector3(0, 0, 1))
    sphere = Sphere()

    inter = Intersection(4, sphere)

    comps = inter.prepare_computations(ray, Intersections([inter]))

    assert comps.t == inter.t
    assert comps.object == inter.object
    assert comps.point == Point3(0, 0, -1)
    assert comps.eye == Vector3(0, 0, -1)
    assert comps.normal == Vector3(0, 0, -1)


def test_inside_or_not():
    ray = Ray(Point3(0, 0, -5), Vector3(0, 0, 1))
    sphere = Sphere()

    inter = Intersection(4, sphere)

    comps = inter.prepare_computations(ray, Intersections([inter]))

    assert comps.inside == False


def test_inside_or_not_2():
    ray = Ray(Point3(0, 0, 0), Vector3(0, 0, 1))
    sphere = Sphere()

    inter = Intersection(1, sphere)

    comps = inter.prepare_computations(ray, Intersections([inter]))

    assert comps.point == Point3(0, 0, 1)
    assert comps.eye == Vector3(0, 0, -1)
    assert comps.inside == True
    assert comps.normal == Vector3(0, 0, -1)


def test_shading_intersection():
    scene = Scene.test_scene()

    ray = Ray(Point3(0, 0, -5), Vector3(0, 0, 1))
    shape = scene.objects[0]
    inter = Intersection(4, shape)

    comps = inter.prepare_computations(ray, Intersections([inter]))
    color = scene.shade_hit(comps)

    assert np.allclose(color.to_array(), Color(0.3806612, 0.47582647, 0.2854959).to_array())


def test_shading_intersection_from_inside():
    scene = Scene.test_scene()
    scene.light = PointLight(Point3(0, 0.25, 0), Color(1, 1, 1))

    ray = Ray(Point3(0, 0, 0), Vector3(0, 0, 1))
    shape = scene.objects[1]
    inter = Intersection(0.5, shape)

    comps = inter.prepare_computations(ray, Intersections([inter]))
    color = scene.shade_hit(comps)

    assert np.allclose(color.to_array(), Color(0.9049845, 0.9049845, 0.9049845).to_array())


def test_color_at_miss():
    scene = Scene.test_scene()

    ray = Ray(Point3(0, 0, -5), Vector3(0, 1, 0))
    color = scene.color_at(ray)

    assert np.allclose(color.to_array(), Color(0, 0, 0).to_array())


def test_color_at_hit():
    scene = Scene.test_scene()

    ray = Ray(Point3(0, 0, -5), Vector3(0, 0, 1))
    color = scene.color_at(ray)

    assert np.allclose(color.to_array(), Color(0.3806612, 0.47582647, 0.2854959).to_array())


def test_color_at_intersection():
    scene = Scene.test_scene()

    outer = scene.objects[0]
    outer.material.ambient = 1.0
    inner = scene.objects[1]
    inner.material.ambient = 1.0

    ray = Ray(Point3(0, 0, 0.75), Vector3(0, 0, -1))
    color = scene.color_at(ray)

    assert np.allclose(color.to_array(), inner.material.color.to_array())


def test_transformation_matrix():
    p_from = Point3(0, 0, 0)
    p_to = Point3(0, 0, -1)
    v_up = Vector3(0, 1, 0)

    m_transform = Matrix4.view_transform(p_from, p_to, v_up)

    assert m_transform == Matrix4.identity()


def test_transformation_matrix_scaling():
    p_from = Point3(0, 0, 0)
    p_to = Point3(0, 0, 1)
    v_up = Vector3(0, 1, 0)

    m_transform = Matrix4.view_transform(p_from, p_to, v_up)

    assert m_transform == Matrix4.scaling(-1, 1, -1)


def test_transformation_matrix_translation():
    p_from = Point3(0, 0, 8)
    p_to = Point3(0, 0, 0)
    v_up = Vector3(0, 1, 0)

    m_transform = Matrix4.view_transform(p_from, p_to, v_up)

    assert m_transform == Matrix4.translation(0, 0, -8)


def test_camera():
    cam = Camera(hsize=160, vsize=120, fov=np.pi / 2)

    assert cam.hsize == 160
    assert cam.vsize == 120
    assert cam.fov == np.pi / 2
    assert cam.transform == Matrix4.identity()


def test_camera_pixel_size():
    cam = Camera(hsize=200, vsize=125, fov=np.pi / 2)

    assert np.isclose(cam.pixel_size, 0.01)

    cam = Camera(hsize=125, vsize=200, fov=np.pi / 2)

    assert np.isclose(cam.pixel_size, 0.01)


def test_camera_ray_for_pixel():
    cam = Camera(hsize=201, vsize=101, fov=np.pi / 2)

    ray = cam.ray_from_pixel(100, 50)
    assert ray.origin == Point3(0, 0, 0)
    assert ray.dir == Vector3(0, 0, -1)

    ray = cam.ray_from_pixel(0, 0)
    assert ray.origin == Point3(0, 0, 0)
    assert ray.dir == Vector3(0.66519, 0.33259, -0.66851)

    cam.transform = Matrix4.rotation_y(np.pi / 4) @ Matrix4.translation(0, -2, 5)
    ray = cam.ray_from_pixel(100, 50)
    assert ray.origin == Point3(0, 2, -5)
    assert ray.dir == Vector3(np.sqrt(2) / 2, 0, -np.sqrt(2) / 2)


def test_camera_render():
    scene = Scene.test_scene()
    cam = Camera(hsize=11, vsize=11, fov=np.pi / 2)

    p_from = Point3(0, 0, -5)
    p_to = Point3(0, 0, 0)
    v_up = Vector3(0, 1, 0)

    cam.transform = Matrix4.view_transform(p_from, p_to, v_up)

    canvas = scene.render(cam)

    assert canvas.get_pixel(5, 5) == Color(0.3806612, 0.47582647, 0.2854959)


def test_surface_in_shadow():
    eye = Vector3(0, 0, -1)
    normal = Vector3(0, 0, -1)
    light = PointLight(Point3(0, 0, -10), Color(1, 1, 1))

    in_shadow = True

    shape = Sphere()
    position = Point3(0, 0, 0)

    color = shape.material.lit(shape, light, position, eye, normal, in_shadow)

    assert color == Color(0.1, 0.1, 0.1)


def test_is_shadowed():
    scene = Scene.test_scene()

    point = Point3(0, 10, 0)
    assert scene.is_shadowed(point) == False

    point = Point3(10, -10, 10)
    assert scene.is_shadowed(point) == True

    point = Point3(-20, 20, -20)
    assert scene.is_shadowed(point) == False

    point = Point3(-2, -2, -2)
    assert scene.is_shadowed(point) == False


def test_shade_hit_inters():
    scene = Scene()

    scene.light = PointLight(Point3(0, 0, -10), Color(1, 1, 1))

    s1 = Sphere()
    scene.add_object(s1)

    s2 = Sphere()
    s2.transform = Matrix4.translation(0, 0, 10)
    scene.add_object(s2)

    ray = Ray(Point3(0, 0, 5), Vector3(0, 0, 1))
    inter = Intersection(4, s2)

    comps = inter.prepare_computations(ray, Intersections([inter]))
    color = scene.shade_hit(comps)

    assert color == Color(0.1, 0.1, 0.1)


def test_acne():
    ray = Ray(Point3(0, 0, -5), Vector3(0, 0, 1))

    shape = Sphere()
    shape.transform = Matrix4.translation(0, 0, 1)

    inter = Intersection(5, shape)

    comps = inter.prepare_computations(ray, Intersections([inter]))

    assert comps.over_point.z < -EPSILON / 2
    assert comps.point.z > comps.over_point.z


def test_striped_pattern():
    pattern = StripedPattern()

    assert pattern.a_color == Color(1, 1, 1)
    assert pattern.b_color == Color(0, 0, 0)


def test_striped_pattern_alternating():
    pattern = StripedPattern()

    assert pattern.at(Point3(0, 0, 0)) == Color(1, 1, 1)

    assert pattern.at(Point3(0, 1, 0)) == Color(1, 1, 1)
    assert pattern.at(Point3(0, 2, 0)) == Color(1, 1, 1)

    assert pattern.at(Point3(0, 0, 1)) == Color(1, 1, 1)
    assert pattern.at(Point3(0, 0, 2)) == Color(1, 1, 1)

    assert pattern.at(Point3(0.9, 0, 0)) == Color(1, 1, 1)
    assert pattern.at(Point3(1, 0, 0)) == Color(0, 0, 0)
    assert pattern.at(Point3(-0.1, 0, 0)) == Color(0, 0, 0)
    assert pattern.at(Point3(-1, 0, 0)) == Color(0, 0, 0)
    assert pattern.at(Point3(-1.1, 0, 0)) == Color(1, 1, 1)


def test_material_striped_pattern():
    shape = Sphere()
    mat = shape.material

    mat.pattern = StripedPattern()
    mat.ambient = 1.0
    mat.diffuse = 0.0
    mat.specular = 0.0

    eye = Vector3(0, 0, -1)
    normal = Vector3(0, 0, -1)

    light = PointLight(Point3(0, 0, -10), Color(1, 1, 1))

    c1 = mat.lit(shape, light, Point3(0.9, 0, 0), eye, normal, False)
    c2 = mat.lit(shape, light, Point3(1.1, 0, 0), eye, normal, False)

    assert c1 == Color(1, 1, 1)
    assert c2 == Color(0, 0, 0)


def test_gradient_pattern():
    pattern = GradientPattern()

    assert pattern.at(Point3(0, 0, 0)) == Color(1, 1, 1)
    assert pattern.at(Point3(0.25, 0, 0)) == Color(0.75, 0.75, 0.75)
    assert pattern.at(Point3(0.5, 0, 0)) == Color(0.5, 0.5, 0.5)
    assert pattern.at(Point3(0.75, 0, 0)) == Color(0.25, 0.25, 0.25)


def test_ring_pattern():
    pattern = RingPattern()

    assert pattern.at(Point3(0, 0, 0)) == Color(1, 1, 1)
    assert pattern.at(Point3(1, 0, 0)) == Color(0, 0, 0)
    assert pattern.at(Point3(0, 0, 1)) == Color(0, 0, 0)
    assert pattern.at(Point3(0.708, 0, 0.708)) == Color(0, 0, 0)


def test_checkers_repeat_in_x():
    pattern = CheckeredPattern()

    assert pattern.at(Point3(0, 0, 0)) == Color(1, 1, 1)
    assert pattern.at(Point3(0.99, 0, 0)) == Color(1, 1, 1)
    assert pattern.at(Point3(1.01, 0, 0)) == Color(0, 0, 0)


def test_checkers_repeat_in_y():
    pattern = CheckeredPattern()

    assert pattern.at(Point3(0, 0, 0)) == Color(1, 1, 1)
    assert pattern.at(Point3(0, 0.99, 0)) == Color(1, 1, 1)
    assert pattern.at(Point3(0, 1.01, 0)) == Color(0, 0, 0)


def test_checkers_repeat_in_z():
    pattern = CheckeredPattern()

    assert pattern.at(Point3(0, 0, 0)) == Color(1, 1, 1)
    assert pattern.at(Point3(0, 0, 0.99)) == Color(1, 1, 1)
    assert pattern.at(Point3(0, 0, 1.01)) == Color(0, 0, 0)


def test_reflect_vector():
    shape = Plane()

    ray = Ray(Point3(0, 1, -1), Vector3(0, -np.sqrt(2) / 2, np.sqrt(2) / 2))

    inter = Intersection(np.sqrt(2), shape)

    comps = inter.prepare_computations(ray, Intersections([inter]))

    assert comps.reflect == Vector3(0, np.sqrt(2) / 2, np.sqrt(2) / 2)


def test_default_material_transparency_and_refractive_index():
    m = Material()

    assert m.transparency == 0.0
    assert m.ior == 1.0


def test_glass_sphere():
    s = Sphere.glass()

    assert s.transform == Matrix4.identity()
    assert s.material.transparency == 1.0
    assert s.material.ior == 1.5


def test_finding_n1_and_n2():
    A = Sphere()
    A.transform = Matrix4.scaling(2, 2, 2)
    A.material.ior = 1.5

    B = Sphere()
    B.transform = Matrix4.translation(0, 0, -0.25)
    B.material.ior = 2.0

    C = Sphere()
    C.transform = Matrix4.translation(0, 0, 0.25)
    C.material.ior = 2.5

    ray = Ray(Point3(0, 0, -4), Vector3(0, 0, 1))

    ints = [
        Intersection(2, A),
        Intersection(2.75, B),
        Intersection(3.25, C),
        Intersection(4.75, B),
        Intersection(5.25, C),
        Intersection(6, A),
    ]

    xs = Intersections(ints)

    scene = Scene(objects=[A, B, C])
    scene.intersect_scene(ray)

    comps1 = Intersection.prepare_computations(ints[0], ray, xs)
    assert comps1.n1 == 1.0
    assert comps1.n2 == 1.5

    comps2 = Intersection.prepare_computations(ints[1], ray, xs)
    assert comps2.n1 == 1.5
    assert comps2.n2 == 2.0

    comps3 = Intersection.prepare_computations(ints[2], ray, xs)
    assert comps3.n1 == 2.0
    assert comps3.n2 == 2.5

    comps4 = Intersection.prepare_computations(ints[3], ray, xs)
    assert comps4.n1 == 2.5
    assert comps4.n2 == 2.5

    comps5 = Intersection.prepare_computations(ints[4], ray, xs)
    assert comps5.n1 == 2.5
    assert comps5.n2 == 1.5

    comps6 = Intersection.prepare_computations(ints[5], ray, xs)
    assert comps6.n1 == 1.5
    assert comps6.n2 == 1.0


def test_the_under_point_offset():
    ray = Ray(Point3(0, 0, -5), Vector3(0, 0, 1))

    shape = Sphere.glass()
    shape.transform = Matrix4.translation(0, 0, 1)

    inter = Intersection(5, shape)

    comps = inter.prepare_computations(ray, Intersections([inter]))

    assert comps.under_point.z > -EPSILON / 2
    assert comps.point.z < comps.under_point.z


def test_reflected_color_for_nonreflective_material():
    scene = Scene()

    s = Sphere()
    s.material.ambient = 1

    scene.add_object(s)

    ray = Ray(Point3(0, 0, 0), Vector3(0, 0, 1))

    inter = Intersection(1, s)

    comps = inter.prepare_computations(ray, Intersections([inter]))

    color = scene.reflected_color(comps)

    assert color == Color(0, 0, 0)


def test_reflected_color_for_reflective_material():
    scene = Scene.test_scene()

    shape = Plane()
    shape.material.reflective = 0.5
    shape.transform = Matrix4.translation(0, -1, 0)

    scene.add_object(shape)

    ray = Ray(Point3(0, 0, -3.0), Vector3(0.0, -np.sqrt(2) / 2, np.sqrt(2) / 2))

    inter = Intersection(0.7071067811865476, shape)

    comps = inter.prepare_computations(ray, Intersections([inter]))

    color = scene.reflected_color(comps, 5)

    assert np.allclose(color.to_array(), Color(0.19032, 0.2379, 0.14274).to_array())


def test_reflected_color_at_maximum_recursive_depth():
    scene = Scene()

    shape = Sphere()
    shape.material.reflective = 0.5

    scene.add_object(shape)

    ray = Ray(Point3(0, 0, -3), Vector3(0, 0, 1))

    inter = Intersection(1, shape)

    comps = inter.prepare_computations(ray, Intersections([inter]))

    color = scene.reflected_color(comps, 0)

    assert color == Color(0, 0, 0)


def test_refracted_color_with_an_opaque_surface():
    scene = Scene.test_scene()

    s = scene.objects[0]

    ray = Ray(Point3(0, 0, -5), Vector3(0, 0, 1))

    inters = Intersections([Intersection(4, s), Intersection(6, s)])

    comps = Intersection.prepare_computations(Intersection(4, s), ray, inters)

    color = scene.refracted_color(comps, 5)

    assert color == Color(0, 0, 0)


def test_refracted_color_at_maximum_recursive_depth():
    scene = Scene.test_scene()

    s = scene.objects[0]

    s.material.transparency = 1.0
    s.material.ior = 1.5

    ray = Ray(Point3(0, 0, -5), Vector3(0, 0, 1))

    inters = Intersections([Intersection(4, s), Intersection(6, s)])

    comps = Intersection.prepare_computations(Intersection(4, s), ray, inters)

    color = scene.refracted_color(comps, 0)

    assert color == Color(0, 0, 0)


def test_refracted_color_under_total_internal_reflection():
    scene = Scene.test_scene()

    s = scene.objects[0]

    s.material.transparency = 1.0
    s.material.ior = 1.5

    ray = Ray(Point3(0, 0, np.sqrt(2) / 2), Vector3(0, 1, 0))

    inters = Intersections([Intersection(-np.sqrt(2) / 2, s), Intersection(np.sqrt(2) / 2, s)])

    comps = Intersection.prepare_computations(Intersection(np.sqrt(2) / 2, s), ray, inters)

    color = scene.refracted_color(comps, 5)

    assert color == Color(0, 0, 0)


def test_refracted_color_with_a_refracted_ray():
    scene = Scene.test_scene()

    A = scene.objects[0]

    A.material.ambient = 1.0

    B = scene.objects[1]

    B.material.transparency = 1.0
    B.material.ior = 1.5

    ray = Ray(Point3(0, 0, 0.1), Vector3(0, 1, 0))

    inters = Intersections(
        [
            Intersection(-0.9899, A),
            Intersection(-0.4899, B),
            Intersection(0.4899, B),
            Intersection(0.9899, A),
        ],
    )

    comps = Intersection.prepare_computations(Intersection(0.4899, B), ray, inters)

    color = scene.refracted_color(comps, 5)

    assert np.allclose(color.to_array(), Color(0, 0.99888, 0.04725).to_array())


def test_shade_hit_with_a_reflective_material():
    scene = Scene.test_scene()

    shape = Plane()
    shape.material.reflective = 0.5
    shape.transform = Matrix4.translation(0, -1, 0)

    scene.add_object(shape)

    ray = Ray(Point3(0, 0, -3.0), Vector3(0.0, -np.sqrt(2) / 2, np.sqrt(2) / 2))

    inter = Intersection(0.7071067811865476, shape)

    comps = inter.prepare_computations(ray, Intersections([inter]))

    color = scene.shade_hit(comps)

    assert np.allclose(color.to_array(), Color(0.87677, 0.92436, 0.82918).to_array())


def test_shade_hit_with_a_transparent_material():
    scene = Scene()

    floor = Plane()
    floor.transform = Matrix4.translation(0, -1, 0)
    floor.material.transparency = 0.5
    floor.material.ior = 1.5

    scene.add_object(floor)

    ball = Sphere()
    ball.material.color = Color(1, 0, 0)
    ball.material.ambient = 0.5
    ball.transform = Matrix4.translation(0, -3.5, -0.5)

    scene.add_object(ball)

    ray = Ray(Point3(0, 0, -3), Vector3(0, -np.sqrt(2) / 2, np.sqrt(2) / 2))

    inter = Intersection(np.sqrt(2), floor)

    comps = inter.prepare_computations(ray, Intersections([inter]))

    color = scene.shade_hit(comps, 5)

    assert np.allclose(color.to_array(), Color(0.93642, 0.68642, 0.68642).to_array())


def test_schlick_approximation_under_total_internal_reflection():
    s = Sphere.glass()

    ray = Ray(Point3(0, 0, np.sqrt(2) / 2), Vector3(0, 1, 0))

    inters = Intersections([Intersection(-np.sqrt(2) / 2, s), Intersection(np.sqrt(2) / 2, s)])

    comps = Intersection.prepare_computations(Intersection(np.sqrt(2) / 2, s), ray, inters)

    reflectance = comps.compute_fresnel()

    assert reflectance == 1.0


def test_schlick_approximation_with_perpendicular_viewing_angle():
    s = Sphere.glass()

    ray = Ray(Point3(0, 0, 0), Vector3(0, 1, 0))

    inters = Intersections([Intersection(-1, s), Intersection(1, s)])

    comps = Intersection.prepare_computations(Intersection(1, s), ray, inters)

    reflectance = comps.compute_fresnel()

    assert abs(reflectance - 0.04) < 0.01


def test_schlick_approximation_with_small_angle_and_n2_greater_than_n1():
    s = Sphere.glass()

    ray = Ray(Point3(0, 0.99, -2), Vector3(0, 0, 1))

    inters = Intersections([Intersection(1.8589, s)])

    comps = Intersection.prepare_computations(Intersection(1.8589, s), ray, inters)

    reflectance = comps.compute_fresnel()

    assert abs(reflectance - 0.48873) < 0.01


def test_shade_hit_with_a_reflective_transparent_material():
    scene = Scene()

    floor = Plane()
    floor.transform = Matrix4.translation(0, -1, 0)
    floor.material.reflective = 0.5
    floor.material.transparency = 0.5
    floor.material.ior = 1.5
    scene.add_object(floor)

    ball = Sphere()
    ball.material.color = Color(1, 0, 0)
    ball.material.ambient = 0.5
    ball.transform = Matrix4.translation(0, -3.5, -0.5)
    scene.add_object(ball)

    ray = Ray(Point3(0, 0, -3), Vector3(0, -np.sqrt(2) / 2, np.sqrt(2) / 2))

    inter = Intersection(np.sqrt(2), floor)

    comps = inter.prepare_computations(ray, Intersections([inter]))

    color = scene.shade_hit(comps, 5)

    assert np.allclose(color.to_array(), Color(0.93391, 0.69643, 0.69243).to_array())


def test_cylinder_ray_misses():
    cyl = Cylinder()
    direction = Vector3(0, 1, 0)
    r = Ray(Point3(1, 0, -5), direction)
    xs = cyl.local_intersect(r)
    assert xs.count == 0


def test_ray_strikes_cylinder():
    cyl = Cylinder()
    examples = [
        (Point3(1, 0, -5), Vector3(0, 0, 1), 5, 5),
        (Point3(0, 0, -5), Vector3(0, 0, 1), 4, 6),
        (Point3(0.5, 0, -5), Vector3(0.1, 1, 1).normalized(), 6.80798, 7.08872),
    ]

    for origin, direction, t0_expected, t1_expected in examples:
        r = Ray(origin, direction)
        xs = cyl.local_intersect(r)
        assert xs.count == 2
        assert np.isclose(xs.intersections[0].t, t0_expected)
        assert np.isclose(xs.intersections[1].t, t1_expected)
