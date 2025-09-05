import numpy as np
from core.cpu.color import Color
from core.cpu.patterns.pattern import Pattern
from core.cpu.math.vectors import Point3


class GradientPattern(Pattern):
    def __init__(self, a_color: Color = None, b_color: Color = None):
        super().__init__()

        self.a_color = a_color if a_color is not None else Color(1, 1, 1)
        self.b_color = b_color if b_color is not None else Color(0, 0, 0)

    def at(self, point: Point3):
        dist = self.b_color - self.a_color
        frac = point.x - np.floor(point.x)

        return Color.from_vector(self.a_color + dist * frac)
