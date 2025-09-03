from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from core.constants import EPSILON
from core.rays.computation import Computation
from core.rays.intersections import Intersections

if TYPE_CHECKING:
    from core.rays.ray import Ray
    from core.objects.shapes.shape import Shape


@dataclass
class Intersection:
    t: np.float32
    object: "Shape"

    def prepare_computations(self, ray: "Ray", inters: Intersections) -> Computation:
        comps = Computation()

        comps.t = self.t
        comps.object = self.object
        comps.cast_shadow = self.object.cast_shadow

        comps.point = ray.get_position(comps.t)
        comps.eye = -ray.dir
        comps.normal = self.object.normal_at(comps.point)

        if comps.eye.dot(comps.normal) < 0:
            comps.inside = True
            comps.normal = -comps.normal

        comps.reflect = ray.dir.reflect(comps.normal)

        comps.over_point = comps.point + comps.normal * EPSILON
        comps.under_point = comps.point - comps.normal * EPSILON

        containers: list[Shape] = []

        for i in inters.intersections:
            if i == self:
                if len(containers) == 0:
                    comps.n1 = 1.0
                else:
                    comps.n1 = containers[-1].material.ior

            if containers.count(comps.object) > 0:
                containers.remove(comps.object)
            else:
                containers.append(comps.object)

            if i == self:
                if len(containers) == 0:
                    comps.n2 = 1.0
                else:
                    comps.n2 = containers[-1].material.ior
                break

        return comps
