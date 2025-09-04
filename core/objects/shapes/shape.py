from typing import TYPE_CHECKING, Optional

from core.materials.material import Material
from core.math.matrices import Matrix4
from core.math.vectors import Point3, Vector3
from core.opt.bounds import _DISPATCH, Bounds
from core.utils import normal_to_world, world_to_object

if TYPE_CHECKING:
    from core.objects.shapes.group import Group
    from core.rays.intersections import Intersections
    from core.rays.ray import Ray


class Shape:
    def __init__(
        self,
        transform: Optional[Matrix4] = None,
        material: Optional[Material] = None,
        cast_shadow: bool = True,
        parent: Optional["Group"] = None,
    ):
        self.transform = transform if transform is not None else Matrix4.identity()
        self.material = material if material is not None else Material.default()
        self.cast_shadow = cast_shadow
        self.parent = parent
        self.local_bounds = None

    def __repr__(self):
        return f"Shape(transform={self.transform}, material={self.material})"

    def intersect(self, ray: "Ray") -> "Intersections":
        """
        When intersecting the shape with a ray, all shapes need to first convert
        the ray into object space, transforming it by the inverse of the shape's
        transformation matrix

        Args:
            ray (Ray): The ray that will intersect

        Returns:
            Intersections: A list of intersections
        """
        ray = ray.transform(self.transform.inverse())
        return self.local_intersect(ray)

    def local_intersect(self, ray: "Ray") -> "Intersections":
        return NotImplemented

    def normal_at(self, point: Point3) -> Vector3:
        """
        When computing the normal vector, all shapes need to first convert the
        point to object space, multiplying it by the inverse of the shape’s transfor-
        mation matrix. Then, after computing the normal they must transform it
        by the inverse of the transpose of the transformation matrix, and then
        normalize the resulting vector before returning it.

        Args:
            point (Point3): Point to get normal from

        Returns:
            Vector3: The normal vector at `point`
        """
        local_point = world_to_object(self, point)
        local_normal = self.local_normal_at(local_point)
        return normal_to_world(self, local_normal)

    def local_normal_at(self, point: Point3) -> Vector3:
        return NotImplemented

    def bounds(self) -> Bounds:
        return NotImplemented
