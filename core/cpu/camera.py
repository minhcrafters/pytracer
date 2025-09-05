import numpy as np

from core.cpu.math.matrices import Matrix4
from core.cpu.math.vectors import Point3
from core.cpu.rays.ray import Ray


class Camera:
    def __init__(self, hsize: np.uint16, vsize: np.uint16, fov: np.float32):
        self.hsize = hsize
        self.vsize = vsize
        self.fov = fov
        self.transform = Matrix4.identity()

        half_view = np.tan(self.fov / 2)
        aspect = self.hsize / self.vsize

        if aspect >= 1:
            self.half_width = half_view
            self.half_height = half_view / aspect
        else:
            self.half_width = half_view * aspect
            self.half_height = half_view

        self.pixel_size: np.float32 = (self.half_width * 2) / self.hsize

    def ray_from_pixel(self, px: np.uint16, py: np.uint16) -> Ray:
        x_offset = (px + 0.5) * self.pixel_size
        y_offset = (py + 0.5) * self.pixel_size

        world_x = self.half_width - x_offset
        world_y = self.half_height - y_offset

        pixel = Point3.from_xyzw(
            self.transform.inverse()[:] @ Point3(world_x, world_y, -1).to_xyzw()
        )
        origin = Point3.from_xyzw(self.transform.inverse()[:] @ Point3(0, 0, 0).to_xyzw())
        dir = (pixel - origin).normalize()

        return Ray(origin, dir)
