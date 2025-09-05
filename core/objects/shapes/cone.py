import numpy as np

from core.constants import EPSILON
from core.objects.shapes import Shape
from core.math.vectors import Vector3
from core.rays.intersection import Intersection
from core.rays.intersections import Intersections


class Cone(Shape):
    def __init__(self):
        super().__init__()
        self.minimum = -np.inf
        self.maximum = np.inf
        self.closed = False

    @classmethod
    def solid(cls):
        cone = cls()
        cone.closed = True
        return cone

    def local_intersect(self, ray):
        # a = dx² + dz² - (dy²)
        # b = 2(rx dx + rz dz) - 2(ry dy)
        # c = rx² + rz² - (ry²)

        a = ray.dir.x**2 - ray.dir.y**2 + ray.dir.z**2
        b = 2 * (ray.origin.x * ray.dir.x - ray.origin.y * ray.dir.y + ray.origin.z * ray.dir.z)
        c = ray.origin.x**2 - ray.origin.y**2 + ray.origin.z**2

        if np.isclose(a, 0.0):
            if np.isclose(b, 0.0):
                return []  # Ray doesn't intersect cone

            # Single intersection
            t = -c / (2 * b)
            if t >= 0:
                # Check for truncation
                y = ray.origin.y + t * ray.dir.y
                if self.minimum < y < self.maximum:
                    return Intersections([Intersection(t, self)])
                else:
                    return Intersections()
            else:
                Intersections()

        # Quadratic equation: at² + bt + c = 0
        discriminant = b**2 - 4 * a * c
        if discriminant < 0:
            return Intersections()  # No real roots

        sqrt_disc = np.sqrt(discriminant)
        t1 = (-b - sqrt_disc) / (2 * a)
        t2 = (-b + sqrt_disc) / (2 * a)

        intersections = []
        # Check intersection with side
        for t in [t1, t2]:
            if t >= 0:
                y = ray.origin.y + t * ray.dir.y
                if self.minimum < y < self.maximum:
                    intersections.append(Intersection(t, self))

        # Check cap intersections if cone is closed
        if self.closed:
            t = (self.minimum - ray.origin.y) / ray.dir.y
            if (
                t >= 0
                and np.isclose(ray.origin.x + t * ray.dir.x, self.minimum)
                and (ray.origin.z + t * ray.dir.z) ** 2 + self.minimum**2 <= self.minimum**2
            ):
                intersections.append(Intersection(t, self))

            t = (self.maximum - ray.origin.y) / ray.dir.y
            if (
                t >= 0
                and np.isclose(ray.origin.x + t * ray.dir.x, self.maximum)
                and (ray.origin.z + t * ray.dir.z) ** 2 + self.maximum**2 <= self.maximum**2
            ):
                intersections.append(Intersection(t, self))

        return intersections

    def local_normal_at(self, point):
        dist = np.sqrt(point.x**2 + point.z**2)

        if dist < 1 and point.y >= self.maximum - EPSILON:
            return Vector3(0, 1, 0)
        elif dist < 1 and point.y <= self.minimum + EPSILON:
            return Vector3(0, -1, 0)
        else:
            # Normal for the side
            y = np.sqrt(point.x**2 + point.z**2)
            if point.y > 0:
                y = -y
            return Vector3(point.x, y, point.z)
