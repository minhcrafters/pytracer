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
    inside: bool = False
    over_point: Point3 | None = None
