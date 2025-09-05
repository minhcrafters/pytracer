import argparse

from core.scene_config import load_scene
from render_preview import render_scene

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", required=True)
    args = parser.parse_args()

    scene_data = load_scene(args.scene)
    print(f"Scene {args.scene} loaded successfully!")

    # print(scene_data.scene.objects)

    canvas = render_scene(
        scene_data.scene,
        scene_data.camera,
        scene_data.camera.hsize * 5,
        scene_data.camera.vsize * 5,
    )

    with open("output.ppm", "w") as f:
        f.write(canvas.to_ppm().getvalue())
