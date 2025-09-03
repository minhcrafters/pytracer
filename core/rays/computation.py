from dataclasses import dataclass

import numpy as np

from core.math.vectors import Point3, Vector3
from core.objects.shapes.shape import Shape


@dataclass
class Computation:
    t: np.float32 = 0.0
    object: Shape | None = None
    point: Point3 | None = None
    eye: Vector3 | None = None
    normal: Vector3 | None = None
    reflect: Vector3 | None = None
    inside: bool = False
    over_point: Point3 | None = None
    under_point: Point3 | None = None
    cast_shadow: bool = True
    n1: np.float32 = 1.0
    n2: np.float32 = 1.0

    @staticmethod
    def fresnel_schlick(comps: "Computation"):
        cos = comps.eye.dot(comps.normal)

        if comps.n1 > comps.n2:
            n = comps.n1 / comps.n2
            sin2_t = n**2 * (1 - cos**2)

            if sin2_t > 1.0:
                return 1.0

            cos_t = np.sqrt(1 - sin2_t)
            cos = cos_t

        r0 = ((comps.n1 - comps.n2) / (comps.n1 + comps.n2)) ** 2

        return r0 + (1 - r0) * (1 - cos) ** 5
