from core.objects.shapes.shape import Shape
from core.rays.intersection import Intersection
from core.rays.intersections import Intersections


class Group(Shape):
    def __init__(self, transform=None, material=None, cast_shadow=True, parent=None):
        super().__init__(transform, material, cast_shadow, parent)

        self.shapes: list[Shape] = []

    def __getitem__(self, idx):
        return self.shapes[idx]

    def add_child(self, *args: Shape) -> None:
        for shape in args:
            shape.parent = self
            self.shapes.append(shape)

    def local_intersect(self, ray):
        total_inters: list[Intersection] = []

        for shape in self.shapes:
            res = shape.intersect(ray)

            for r in res.intersections:
                total_inters.append(r)

        total_inters.sort(key=lambda x: x.t)

        return Intersections(total_inters)
