from dataclasses import dataclass

import numpy as np

from .intersection import Intersection


@dataclass
class Intersections:
    count: np.uint32
    intersections: list[Intersection]
