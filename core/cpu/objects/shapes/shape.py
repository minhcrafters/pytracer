from typing import TYPE_CHECKING, Optional

import numpy as np

from core.cpu.constants import EPSILON
from core.cpu.materials.material import Material
from core.cpu.math.matrices import Matrix4
from core.cpu.math.vectors import Point3, Vector3
from core.cpu.opt.bounds import Bounds
from core.cpu.utils import normal_to_world, world_to_object

if TYPE_CHECKING:
    from core.cpu.objects.shapes.group import Group
    from core.cpu.rays.intersections import Intersections
    from core.cpu.rays.ray import Ray


class Shape:
    def __init__(
        self,
        transform: Optional[Matrix4] = None,
        material: Optional[Material] = None,
        cast_shadow: bool = True,
        parent: Optional["Group"] = None,
    ):
        self.transform = transform if transform is not None else Matrix4.identity()
        self.material = material if material is not None else Material.white()
        self.cast_shadow = cast_shadow
        self.parent = parent
        self.local_bounds = None
        self._inverse_transform = None  # Cache for inverse transform

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

        if not self.check_bounds_intersections(ray):
            return Intersections()

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

    def check_bounds_intersections(self, ray: "Ray") -> bool:
        x_tmin, x_tmax = self._check_axis(
            ray.origin.x, ray.dir.x, self.bounds.minimum.x, self.bounds.maximum.x
        )

        if x_tmin > x_tmax:
            return False

        y_tmin, y_tmax = self._check_axis(
            ray.origin.y, ray.dir.y, self.bounds.minimum.y, self.bounds.maximum.y
        )

        if y_tmin > y_tmax:
            return False

        z_tmin, z_tmax = self._check_axis(
            ray.origin.z, ray.dir.z, self.bounds.minimum.z, self.bounds.maximum.z
        )

        if z_tmin > z_tmax:
            return False

        return True

    def _check_axis(
        self, origin: np.float32, dir: np.float32, min_b: np.float32, max_b: np.float32
    ):
        tmin = min_b - origin
        tmax = max_b - origin

        if np.abs(dir) >= EPSILON:
            tmin = tmin / dir
            tmax = tmax / dir
        else:
            tmin = tmin * np.inf
            tmax = tmax * np.inf

        if tmin > tmax:
            tmin, tmax = tmax, tmin

        return tmin, tmax

    @property
    def bounds(self) -> Bounds:
        return NotImplemented
