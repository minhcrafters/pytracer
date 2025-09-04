from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .intersection import Intersection


class Intersections:
    def __init__(self, intersections: list["Intersection"] = None):
        self.intersections = intersections if intersections is not None else []
        self.count = len(self.intersections)

    def add(self, inter: "Intersection") -> None:
        self.intersections.append(inter)
        self.count += 1
