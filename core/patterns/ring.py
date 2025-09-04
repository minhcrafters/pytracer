import numpy as np
from core.color import Color
from core.patterns.pattern import Pattern
from core.math.vectors import Point3


class RingPattern(Pattern):
    def __init__(self, a_color: Color = None, b_color: Color = None):
        super().__init__()

        self.a_color = a_color if a_color is not None else Color(1, 1, 1)
        self.b_color = b_color if b_color is not None else Color(0, 0, 0)

    def at(self, point: Point3):
        if np.floor(np.hypot(point.x, point.z)) % 2 == 0:
            return self.a_color
        else:
            return self.b_color
