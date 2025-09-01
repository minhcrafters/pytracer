from dataclasses import dataclass

from core.canvas import Canvas
from core.math.vectors import Point3, Vector2, Vector3


@dataclass
class Projectile:
    position: Point3
    velocity: Vector3


@dataclass
class Environment:
    gravity: Vector3
    wind: Vector3


def tick(env: Environment, proj: Projectile):
    position = proj.position + proj.velocity
    velocity = proj.velocity + env.gravity + env.wind

    return Projectile(position, velocity)


def main():
    p = Projectile(Vector3(0, 1, 0), Vector3(1, 1.5, 0).normalize() * 12.25)
    e = Environment(Vector3(0, -0.1, 0), Vector3(-0.01, 0, 0))

    size = Vector2(900, 550)

    c = Canvas(*size.to_array().tolist())

    ticks = 0

    while p.position.y > 0:
        p = tick(e, p)
        print(p.position)
        c.set_pixel(int(p.position.x), 550 - int(p.position.y), (1, 0, 0))
        ticks += 1

    print(f"Ticks: {ticks}")

    ppm = c.to_ppm()

    with open("output.ppm", "w") as f:
        f.write(ppm.getvalue())


if __name__ == "__main__":
    main()
