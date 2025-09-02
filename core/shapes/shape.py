from core.materials.material import Material
from core.math.matrices import Matrix4
from core.math.vectors import Point3, Vector3


class Shape:
    def __init__(
        self,
        id: int,
        transform: Matrix4 = Matrix4.identity(),
        material: Material = Material.default(),
    ):
        self.id = id
        self.transform = transform
        self.material = material

    def __repr__(self):
        return f"Shape(id={self.id}, transform={self.transform})"

    def intersect(self, ray):
        return NotImplemented

    def normal_at(self, point: Point3):
        return NotImplemented
