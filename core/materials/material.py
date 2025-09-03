from dataclasses import dataclass, field

import numpy as np

from core.color import Color
from core.materials.pattern.pattern import Pattern


@dataclass
class Material:
    color: Color | None = None
    pattern: Pattern | None = None
    ambient: np.float32 = 0.1
    diffuse: np.float32 = 0.9
    specular: np.float32 = 0.9
    shininess: np.float32 = 200.0
    reflective: np.float32 = 0.0

    @classmethod
    def default(cls):
        return cls(Color(1, 1, 1))
