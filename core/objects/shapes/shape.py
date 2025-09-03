from typing import TYPE_CHECKING, Optional

from core.materials.material import Material
from core.math.matrices import Matrix4
from core.math.vectors import Point3, Vector3

if TYPE_CHECKING:
    from core.rays.intersections import Intersections
    from core.rays.ray import Ray


class Shape:
    def __init__(
        self,
        transform: Optional[Matrix4] = None,
        material: Optional[Material] = None,
    ):
        self.transform = transform if transform is not None else Matrix4.identity()
        self.material = material if material is not None else Material.default()

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
        return NotImplemented
