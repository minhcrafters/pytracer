from typing import TYPE_CHECKING

from core.math.vectors import Point3, Vector3

if TYPE_CHECKING:
    from core.objects.shapes.shape import Shape


def world_to_object(shape: "Shape", point: Point3) -> Point3:
    if shape.parent:
        point = world_to_object(shape.parent, point)

    return Point3.from_xyzw(shape.transform.inverse()[:] @ point.to_xyzw())


def normal_to_world(shape: "Shape", normal: Vector3) -> Vector3:
    normal = shape.transform.inverse().transpose()[:] @ normal.to_xyzw()
    normal[3] = 0.0
    normal = Vector3.from_xyzw(normal)
    normal = normal.normalized()

    if shape.parent:
        normal = normal_to_world(shape.parent, normal)

    return normal
