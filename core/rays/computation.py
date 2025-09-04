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
    cast_shadows: bool = True
    n1: np.float32 = 1.0
    n2: np.float32 = 1.0

    def compute_fresnel(self) -> np.float32:
        # using schlick's approximation

        cos = self.eye.dot(self.normal)

        # Total internal reflection can only occur if n1 > n2
        if self.n1 > self.n2:
            n = self.n1 / self.n2
            sin2_t = n**2 * (1 - cos**2)

            if sin2_t > 1.0:
                return 1.0

            # For transmission, use cos(theta_t) instead
            cos_t = np.sqrt(1 - sin2_t)
            cos = cos_t

        r0 = ((self.n1 - self.n2) / (self.n1 + self.n2)) ** 2

        return r0 + (1 - r0) * (1 - cos) ** 5
