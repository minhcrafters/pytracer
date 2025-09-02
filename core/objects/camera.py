import numpy as np

from core.math.matrices import Matrix4


class Camera:
    def __init__(self, hsize: np.uint16, vsize: np.uint16, fov: np.float32):
        self.hsize = hsize
        self.vsize = vsize
        self.fov = fov
        self.transform = Matrix4.identity()
