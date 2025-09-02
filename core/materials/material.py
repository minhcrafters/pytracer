from dataclasses import dataclass

import numpy as np

from core.color import Color


@dataclass
class Material:
    color: Color
    ambient: np.float32 = 0.1
    diffuse: np.float32 = 1.0
    specular: np.float32 = 1.0
    shininess: np.float32 = 200.0

    @classmethod
    def default(cls):
        return cls(Color(1, 1, 1))
