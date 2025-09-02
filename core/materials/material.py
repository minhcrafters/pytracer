from dataclasses import dataclass

import numpy as np

from core.color import Color


@dataclass
class Material:
    color: Color
    ambient: np.float32 = 0.1
    diffuse: np.float32 = 0.9
    specular: np.float32 = 0.9
    shininess: np.float32 = 200.0

    @classmethod
    def default(cls):
        return cls(Color(1, 1, 1))
