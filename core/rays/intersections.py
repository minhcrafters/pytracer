from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .intersection import Intersection


@dataclass
class Intersections:
    count: np.uint32
    intersections: list["Intersection"]
