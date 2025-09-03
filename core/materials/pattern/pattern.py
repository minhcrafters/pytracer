from typing import TYPE_CHECKING
from core.color import Color
from core.math.matrices import Matrix4
from core.math.vectors import Point3

if TYPE_CHECKING:
    from core.objects.shapes.shape import Shape


class Pattern:
    def __init__(self):
        self.transform = Matrix4.identity()

    def at(self, point: Point3) -> Color:
        return NotImplemented

    def at_object(self, obj: "Shape", world_point: Point3) -> Color:
        obj_point = Point3.from_xyzw(obj.transform.inverse()[:] @ world_point.to_xyzw())
        pattern_point = Point3.from_xyzw(self.transform.inverse()[:] @ obj_point.to_xyzw())

        return self.at(pattern_point)

    @staticmethod
    def blend(a: "Pattern", b: "Pattern") -> "BlendPattern":
        return BlendPattern(a, b)


class BlendPattern(Pattern):
    def __init__(self, a: Pattern, b: Pattern):
        super().__init__()
        self.a = a
        self.b = b

    def at(self, point: Point3) -> Color:
        col_a: Color = self.a.at(point)
        col_b: Color = self.b.at(point)
        return (col_a + col_b) * 0.5

    def __repr__(self):
        return f"BlendPattern(a={self.a!r} b={self.b!r})"
