import numpy as np
from core.math.matrices import Matrix2, Matrix4
from core.math.vectors import Point3, Vector3
from core.rays.ray import Ray
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


def test_normal():
    sphere = Sphere(0, center=Point3(0, 0, 0), radius=1.0)

    sphere.transform = Matrix4.scaling(1, 0.5, 1) @ Matrix4.rotation_z(np.pi / 5)

    norm = sphere.normal_at(Point3(0, np.sqrt(2) / 2, -np.sqrt(2) / 2))

    expected = Vector3(0, 0.970143, -0.242536)
    
    assert np.allclose(norm.to_array(), expected.to_array())
