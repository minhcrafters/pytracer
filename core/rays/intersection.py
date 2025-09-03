from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from core.constants import EPSILON
from core.rays.computation import Computation

if TYPE_CHECKING:
    from core.rays.ray import Ray
    from core.objects.shapes.shape import Shape


@dataclass
class Intersection:
    t: np.float32
    object: "Shape"

    def prepare_computations(self, ray: "Ray") -> Computation:
        comps = Computation()

        comps.t = self.t
        comps.object = self.object

        comps.point = ray.get_position(comps.t)
        comps.eye = -ray.dir
        comps.normal = self.object.normal_at(comps.point)

        if comps.eye.dot(comps.normal) < 0:
            comps.inside = True
            comps.normal = -comps.normal

        comps.reflect = ray.dir.reflect(comps.normal)

        comps.over_point = comps.point + comps.normal * EPSILON

        return comps
