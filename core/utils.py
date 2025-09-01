from core.math.vector import Point3, Vector2


def world_to_screen(world: Point3, size: Vector2, scale: float = 1.0) -> Vector2:
    # TODO: 3D world to screen using proj matrix
    x = int(size.x / 2 + world.x * scale)
    y = int(size.y / 2 - world.y * scale)
    return Vector2(max(size.x - 1, min(x, 0)), max(size.y - 1, min(y, 0)))
