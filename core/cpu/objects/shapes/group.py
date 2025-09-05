from itertools import product
from pathlib import Path
from typing import Iterable

import numpy as np
import pywavefront
import tqdm
from core.cpu.math.vectors import Point3
from core.cpu.objects.shapes.shape import Shape
from core.cpu.objects.shapes.triangle import Triangle
from core.cpu.opt.bounds import Bounds
from core.cpu.rays.intersection import Intersection
from core.cpu.rays.intersections import Intersections


class Group(Shape):
    def __init__(self, transform=None, material=None, cast_shadow=True, parent=None):
        super().__init__(transform, material, cast_shadow, parent)

        self.children: list[Shape] = []

    @classmethod
    def parse_wavefront(cls, filename: str) -> "Group":
        """
        Parse a Wavefront OBJ file into a Group of Triangles.
        """

        scene = pywavefront.Wavefront(filename, collect_faces=True)

        root = cls()

        for _, mesh in scene.meshes.items():
            subgroup = cls()
            root.add_child(subgroup)

            for face in mesh.faces:
                vertices = [Point3(*scene.vertices[i]) for i in face]

                if len(vertices) == 3:
                    tri = Triangle(vertices[0], vertices[1], vertices[2])
                    subgroup.add_child(tri)
                elif len(vertices) > 3:
                    triangles = cls._fan_triangulate(vertices)
                    subgroup.add_child(*triangles)
                else:
                    continue

        print(f"Loaded OBJ: {filename}")

        return root

    def __getitem__(self, idx):
        return self.children[idx]

    def add_child(self, *args: Shape) -> None:
        for shape in args:
            shape.parent = self
            self.children.append(shape)

    def local_intersect(self, ray):
        if not self.children:
            return Intersections()

        total_inters: list[Intersection] = []

        for child in tqdm.tqdm(self.children, desc="Pre-computing tris"):
            res = child.intersect(ray)

            # if broken shape returned
            if isinstance(res, list):
                continue

            total_inters.extend(res.intersections)

        total_inters.sort(key=lambda x: x.t)

        return total_inters

    @staticmethod
    def _fan_triangulate(vertices: list):
        triangles: list[Triangle] = []

        for i in range(1, len(vertices)):
            tri = Triangle(vertices[0], vertices[i], vertices[i + 1])
            triangles.append(tri)

        return triangles

    @staticmethod
    def _bbox_from_points(points: Iterable[Point3]) -> Bounds:
        xs = [p.x for p in points]
        ys = [p.y for p in points]
        zs = [p.z for p in points]
        return Bounds(Point3(min(xs), min(ys), min(zs)), Point3(max(xs), max(ys), max(zs)))

    @property
    def bounds(self) -> Bounds:
        """
        Compute bounding box in *this group's local space* that encloses all children.
        Skips children whose .bounds is not available (e.g., NotImplemented).
        """
        xs: list[np.float32] = []
        ys: list[np.float32] = []
        zs: list[np.float32] = []

        for child in self.children:
            child_box = getattr(child, "bounds", None)

            if not isinstance(child_box, Bounds):
                continue

            corners = [
                Point3(x, y, z)
                for x, y, z in product(
                    (child_box.minimum.x, child_box.maximum.x),
                    (child_box.minimum.y, child_box.maximum.y),
                    (child_box.minimum.z, child_box.maximum.z),
                )
            ]

            for c in corners:
                transformed_xyzw = child.transform[:] @ c.to_xyzw()
                p = Point3.from_xyzw(transformed_xyzw)
                xs.append(p.x)
                ys.append(p.y)
                zs.append(p.z)

        if not xs:
            return Bounds()

        return Bounds(Point3(min(xs), min(ys), min(zs)), Point3(max(xs), max(ys), max(zs)))
