import argparse

from core.cpu.scene_config import load_scene
from core.cpu.render_preview import render_scene

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("scene")
    parser.add_argument("--gpu", action="store_true", default=True)
    args = parser.parse_args()

    if args.gpu:
        from core.gpu.render_wgpu import main

        main(args.scene)
        exit()

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
