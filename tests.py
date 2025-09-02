import numpy as np
from core.color import Color
from core.lights.point_light import PointLight
from core.math.matrices import Matrix2, Matrix4
from core.math.vectors import Point3, Vector3
from core.rays.intersection import Intersection
from core.rays.ray import Ray
from core.scene import Scene
from core.shapes.sphere import Sphere


def test_matrices():
    mat = Matrix2(np.array([[1, 2], [6, 5]]))

    assert mat.determinant() == -7.0
    expected_inv = np.array([[-5 / 7, 2 / 7], [6 / 7, -1 / 7]])
    assert np.allclose(mat.inverse()[:], expected_inv)


def test_ray_sphere_intersect():
    sphere = Sphere(0, Point3(0, 0, 0), 1.0)
    ray = Ray(Point3(0, 0, 5), Vector3(0, 0, 1), 10)

    intersect = sphere.intersect(ray)

    assert intersect.count == 2  # since negative is still correct


def test_ray_sphere_hit():
    sphere = Sphere(0, Point3(0, 0, 0), 1.0)
    ray = Ray(Point3(0, 0, -5), Vector3(0, 0, 1), 10)

    hit = Ray.hit(ray, sphere)

    assert hit is not None
    assert hit[0].t == 4.0


def test_ray_transform():
    ray = Ray(Point3(1, 2, 3), Vector3(0, 1, 0), 10)

    t_ray = Ray.transform(ray, Matrix4.translation(3, 4, 5))

    assert t_ray.origin == Point3(4, 6, 8)
    assert t_ray.dir == Vector3(0, 1, 0)

    s_ray = Ray.transform(ray, Matrix4.scaling(2, 3, 4))

    assert s_ray.origin == Point3(2, 6, 12)
    assert s_ray.dir == Vector3(0, 3, 0)


def test_ray_sphere_transform():
    sphere = Sphere(0, center=Point3(0, 0, 0), radius=1.0)
    ray = Ray(Point3(0, 0, -5), Vector3(0, 0, 1), 10)

    sphere.transform = Matrix4.scaling(2, 2, 2)

    res = sphere.intersect(ray)

    assert res.count == 2
    assert res.intersections[0].t == 3.0
    assert res.intersections[1].t == 7.0


# def test_normal():
#     sphere = Sphere(0, center=Point3(0, 0, 0), radius=1.0)

#     sphere.transform = Matrix4.scaling(1, 0.5, 1) @ Matrix4.rotation_z(np.pi / 5)

#     norm = sphere.normal_at(Point3(0, np.sqrt(2) / 2, -np.sqrt(2) / 2))

#     expected = Vector3(0, 0.970143, -0.242536)

#     assert np.allclose(norm.to_array(), expected.to_array())


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
    sphere = Sphere(0)

    intersection = Intersection(4, sphere)

    comps = intersection.prepare_computations(ray)

    assert comps.t == intersection.t
    assert comps.object == intersection.object
    assert comps.point == Point3(0, 0, -1)
    assert comps.eye == Vector3(0, 0, -1)
    assert comps.normal == Vector3(0, 0, -1)


def test_inside_or_not():
    ray = Ray(Point3(0, 0, -5), Vector3(0, 0, 1))
    sphere = Sphere(0)

    intersection = Intersection(4, sphere)

    comps = intersection.prepare_computations(ray)

    assert comps.inside == False


def test_inside_or_not_2():
    ray = Ray(Point3(0, 0, 0), Vector3(0, 0, 1))
    sphere = Sphere(0)

    intersection = Intersection(1, sphere)

    comps = intersection.prepare_computations(ray)

    assert comps.point == Point3(0, 0, 1)
    assert comps.eye == Vector3(0, 0, -1)
    assert comps.inside == True
    assert comps.normal == Vector3(0, 0, -1)


def test_shading_intersection():
    scene = Scene.test_scene()

    ray = Ray(Point3(0, 0, -5), Vector3(0, 0, 1))
    shape = scene.objects[0]
    inter = Intersection(4, shape)

    comps = inter.prepare_computations(ray)
    color = scene.shade_hit(comps)

    assert np.allclose(color.to_array(), Color(0.3806612, 0.47582647, 0.2854959).to_array())


def test_shading_intersection_from_inside():
    scene = Scene.test_scene()
    scene.light = PointLight(Point3(0, 0.25, 0), Color(1, 1, 1))

    ray = Ray(Point3(0, 0, 0), Vector3(0, 0, 1))
    shape = scene.objects[1]
    inter = Intersection(0.5, shape)

    comps = inter.prepare_computations(ray)
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