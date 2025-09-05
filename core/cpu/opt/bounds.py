from dataclasses import dataclass, field

from core.cpu.math.vectors import Point3


@dataclass
class Bounds:
    minimum: Point3 = field(default_factory=Point3)
    maximum: Point3 = field(default_factory=Point3)
