from dataclasses import dataclass

import numpy as np

from core.math.vectors import Point3, Vector3
from core.objects.shapes.shape import Shape


@dataclass
class Computation:
    t: np.float32 = 0.0
    object: Shape = None
    point: Point3 = None
    eye: Vector3 = None
    normal: Vector3 = None
    reflect: Vector3 = None
    inside: bool = False
    over_point: Point3 = None
    under_point: Point3 = None
    cast_shadows: bool = True
    n1: np.float32 = 1.0
    n2: np.float32 = 1.0

    def compute_fresnel(self) -> np.float32:
        # using schlick's approximation

        cos = self.eye.dot(self.normal)

        if self.n1 > self.n2:
            n = self.n1 / self.n2
            sin2_t = np.power(n, 2) * (1 - np.power(cos, 2))

            if sin2_t > 1.0:
                return 1.0

            cos_t = np.sqrt(1 - sin2_t)
            cos = cos_t

        r0 = np.power((self.n1 - self.n2) / (self.n1 + self.n2), 2)

        return r0 + (1 - r0) * np.power(1 - cos, 5)
