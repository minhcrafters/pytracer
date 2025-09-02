from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from core.shapes.shape import Shape


@dataclass
class Intersection:
    t: np.float32
    object: 'Shape'

    @staticmethod
    def intersect(s: 'Shape', t: np.float32):
        return Intersection(t, s)
