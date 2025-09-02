from dataclasses import dataclass

import numpy as np

from core.shapes.shape import Shape


@dataclass
class Intersection:
    t: np.float32
    object: Shape

    @staticmethod
    def intersect(s: Shape, t: np.float32):
        return Intersection(t, s)
