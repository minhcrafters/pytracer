import json
from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np

from core.cpu.color import Color
from core.cpu.lights.point_light import PointLight
from core.cpu.materials.material import Material
from core.cpu.math.matrices import Matrix4
from core.cpu.math.vectors import Point3, Vector3
from core.cpu.camera import Camera
from core.cpu.objects.shapes.cone import Cone
from core.cpu.objects.shapes.cube import Cube
from core.cpu.objects.shapes.cylinder import Cylinder
from core.cpu.objects.shapes.group import Group
from core.cpu.objects.shapes.plane import Plane
from core.cpu.objects.shapes.sphere import Sphere
from core.cpu.objects.shapes.triangle import Triangle
from core.cpu.patterns.checkered import CheckeredPattern
from core.cpu.patterns.gradient import GradientPattern
from core.cpu.patterns.ring import RingPattern
from core.cpu.patterns.striped import StripedPattern
from core.cpu.scene import Scene

SHAPE_TYPES: Dict[str, Any] = {
    "sphere": Sphere,
    "plane": Plane,
    "cube": Cube,
    "cylinder": Cylinder,
    "cone": Cone,
    "triangle": Triangle,
    "group": Group,
}

PATTERN_TYPES: Dict[str, Any] = {
    "checkered": CheckeredPattern,
    "striped": StripedPattern,
    "ring": RingPattern,
    "gradient": GradientPattern,
}


@dataclass
class SceneData:
    scene: Scene
    camera: Camera


def load_scene(json_path: str) -> SceneData:
    with open(json_path, "r") as f:
        data = json.load(f)

    camera_data = data.get("camera", {})
    camera = _parse_camera(camera_data)

    light_data = data.get("light", {})
    light = _parse_light(light_data)

    scene = Scene([], light)

    objects_list = data.get("objects", [])
    for obj_data in objects_list:
        shape = _parse_object(obj_data)
        scene.add_object(shape)

    return SceneData(scene, camera)


def _parse_camera(data: Dict[str, Any]) -> Camera:
    width = data.get("width", 100)
    height = data.get("height", 50)
    fov = np.deg2rad(data.get("fov", 90))

    view_transform: Dict[str, Any] = data.get("view_transform")
    from_pos = view_transform.get("from", [0, 1.5, -5])
    to_pos = view_transform.get("to", [0, 1, 0])
    up_vec = view_transform.get("up", [0, 1, 0])

    if not view_transform:
        raise ValueError("view_transform is None")

    camera = Camera(width, height, fov)
    camera.transform = Matrix4.view_transform(Point3(*from_pos), Point3(*to_pos), Vector3(*up_vec))
    return camera


def _parse_light(data: Dict[str, Any]) -> PointLight:
    at = data.get("at", [-10, 10, -10])
    intensity = data.get("intensity", [1, 1, 1])
    return PointLight(Point3(*at), Color(*intensity))


def _parse_object(data: Dict[str, Any]) -> Any:
    if "type" not in data:
        raise ValueError("Object must have 'type' key")

    shape_type_str = data["type"]
    shape_type = SHAPE_TYPES.get(shape_type_str)

    if shape_type is None:
        raise ValueError(f"Unknown shape type '{shape_type_str}'")

    shape_kwargs = {}
    for param in ["maximum", "minimum", "closed"]:
        if param in data:
            shape_kwargs[param] = data[param]

    shape = shape_type(**shape_kwargs)

    if "material" in data:
        shape.material = _parse_material(data["material"])

    if "transform" in data:
        shape.transform = _parse_transform(data["transform"])

    if isinstance(shape, Group) and "children" in data:
        for child_data in data["children"]:
            child = _parse_object(child_data)
            shape.add_child(child)

    return shape


def _parse_material(data: Dict[str, Any]) -> Material:
    material = Material.white()

    if "color" in data:
        material.color = Color(*data["color"])

    for prop in ["ambient", "diffuse", "specular", "shininess", "reflective", "transparency"]:
        if prop in data:
            setattr(material, prop, data[prop])

    if "refractive-index" in data:
        material.ior = data["refractive-index"]

    if "pattern" in data:
        material.pattern = _parse_pattern(data["pattern"])

    return material


def _parse_pattern(data: Dict[str, Any]) -> Any:
    if "type" not in data:
        raise ValueError(f"Pattern must have 'type' key containing a pattern type: {data}")

    pattern_type_str = data["type"]
    pattern_type = PATTERN_TYPES.get(pattern_type_str)

    if pattern_type is None:
        raise ValueError(f"Unknown pattern type '{pattern_type_str}'")

    pattern_kwargs = dict(data)
    del pattern_kwargs["type"]

    if "a" in pattern_kwargs:
        pattern_kwargs["a"] = Color(*pattern_kwargs["a"])
    if "b" in pattern_kwargs:
        pattern_kwargs["b"] = Color(*pattern_kwargs["b"])
    if "a_color" in pattern_kwargs:
        pattern_kwargs["a_color"] = Color(*pattern_kwargs["a_color"])
    if "b_color" in pattern_kwargs:
        pattern_kwargs["b_color"] = Color(*pattern_kwargs["b_color"])

    return pattern_type(**pattern_kwargs)


def _parse_transform(transforms: List[Dict[str, Any]]) -> Matrix4:
    transform = Matrix4.identity()
    for t in transforms:
        if "translate" in t:
            transform = transform @ Matrix4.translation(*t["translate"])
        elif "scale" in t:
            transform = transform @ Matrix4.scaling(*t["scale"])
        elif "rotate_x" in t:
            transform = transform @ Matrix4.rotation_x(t["rotate_x"])
        elif "rotate_y" in t:
            transform = transform @ Matrix4.rotation_y(t["rotate_y"])
        elif "rotate_z" in t:
            transform = transform @ Matrix4.rotation_z(t["rotate_z"])
        else:
            raise ValueError(f"Unknown transform {t}")
    return transform
