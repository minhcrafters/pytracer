from io import StringIO
import json, os, subprocess
import numpy as np
from PIL import Image
from wgpu.utils.compute import compute_with_buffers
import pywavefront

SHADER_FILE = "compute.spv"
OBJ_STRIDE = 24  # vec4s per object (0..3 header, 4..7 inv cols, 8..11 fwd cols, 12..14 triangle p1 p2 p3, 15 group children, 16 pattern a_color, 17 pattern b_color, 18..21 pattern inv transform)

# Shape type codes
SHAPE_PLANE = 0
SHAPE_SPHERE = 1
SHAPE_CUBE = 2
SHAPE_CYLINDER = 3
SHAPE_TRIANGLE = 4
SHAPE_GROUP = 5


def build_transform_matrix(transforms):
    M = np.eye(4, dtype=np.float32)
    for t in transforms:
        if "translate" in t:
            x, y, z = t["translate"]
            T = np.eye(4, dtype=np.float32)
            T[0, 3] = x
            T[1, 3] = y
            T[2, 3] = z
            M = M @ T
        elif "scale" in t:
            sx, sy, sz = t["scale"]
            S = np.diag([sx, sy, sz, 1.0]).astype(np.float32)
            M = M @ S
        elif "rotate_x" in t:
            a = np.deg2rad(t["rotate_x"])
            ca = np.cos(a)
            sa = np.sin(a)
            R = np.array(
                [[1, 0, 0, 0], [0, ca, -sa, 0], [0, sa, ca, 0], [0, 0, 0, 1]], dtype=np.float32
            )
            M = M @ R
        elif "rotate_y" in t:
            a = np.deg2rad(t["rotate_y"])
            ca = np.cos(a)
            sa = np.sin(a)
            R = np.array(
                [[ca, 0, sa, 0], [0, 1, 0, 0], [-sa, 0, ca, 0], [0, 0, 0, 1]], dtype=np.float32
            )
            M = M @ R
        elif "rotate_z" in t:
            a = np.deg2rad(t["rotate_z"])
            ca = np.cos(a)
            sa = np.sin(a)
            R = np.array(
                [[ca, -sa, 0, 0], [sa, ca, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float32
            )
            M = M @ R
        elif "shear" in t:
            xy, xz, yx, yz, zx, zy = t["shear"]
            S = np.eye(4, dtype=np.float32)
            S[0, 1] = xy
            S[0, 2] = xz
            S[1, 0] = yx
            S[1, 2] = yz
            S[2, 0] = zx
            S[2, 1] = zy
            M = M @ S
    return M


def fan_triangulate(vertices):
    """Fan triangulation for polygons"""
    if len(vertices) < 3:
        return []
    triangles = []
    for i in range(1, len(vertices) - 1):
        tri = [vertices[0], vertices[i], vertices[i + 1]]
        triangles.append(tri)
    return triangles


def load_obj_to_triangles(obj_path, material):
    """Load OBJ file and convert to triangle objects"""
    scene = pywavefront.Wavefront(obj_path, collect_faces=True)
    triangles = []

    for _, mesh in scene.meshes.items():
        for face in mesh.faces:
            face_vertices = [scene.vertices[i] for i in face]
            if len(face_vertices) == 3:
                tris = [face_vertices]
            elif len(face_vertices) > 3:
                tris = fan_triangulate(face_vertices)
            else:
                continue

            for tri in tris:
                triangle = {
                    "type": "triangle",
                    "p1": tri[0],
                    "p2": tri[1],
                    "p3": tri[2],
                    "material": material,
                }
                triangles.append(triangle)

    return triangles


def flatten_objects(objects):
    flattened = []
    group_children = []

    def add_object(obj):
        idx = len(flattened)
        flattened.append(obj)
        return idx

    for obj in objects:
        if obj.get("type") == "group":
            # Handle OBJ loading
            if "from-obj" in obj:
                obj_path = obj["from-obj"]
                material = obj.get("material", {})
                triangles = load_obj_to_triangles(obj_path, material)
                obj["children"] = triangles

            group_idx = add_object(obj)
            children_start = len(group_children)
            children_indices = []
            for child in obj.get("children", []):
                child_idx = add_object(child)
                group_children.append(child_idx)
                children_indices.append(child_idx)
            obj["children_start"] = children_start
            obj["children_count"] = len(children_indices)
        else:
            add_object(obj)
    return flattened, group_children


def pack_scene(scene):
    objs, group_children = flatten_objects(scene.get("objects", []))
    n = len(objs)
    data = np.zeros((n * OBJ_STRIDE, 4), dtype=np.float32)
    for i, o in enumerate(objs):
        base = i * OBJ_STRIDE
        typ = o.get("type", "sphere")
        tcode = SHAPE_SPHERE
        if typ == "plane":
            tcode = SHAPE_PLANE
        elif typ == "sphere":
            tcode = SHAPE_SPHERE
        elif typ == "cube":
            tcode = SHAPE_CUBE
        elif typ == "cylinder":
            tcode = SHAPE_CYLINDER
        elif typ == "triangle":
            tcode = SHAPE_TRIANGLE
        elif typ == "group":
            tcode = SHAPE_GROUP
        closed = 1.0 if o.get("closed", False) else 0.0
        minY = float(o.get("minimum", -1e6))
        maxY = float(o.get("maximum", 1e6))
        data[base + 0, 0] = float(tcode)
        data[base + 0, 1] = closed
        data[base + 0, 2] = minY
        data[base + 0, 3] = maxY
        mat = o.get("material", {})
        color = mat.get("color", mat.get("pattern", {}).get("a_color", [1.0, 1.0, 1.0]))
        data[base + 1, 0:3] = np.array(color[:3], dtype=np.float32)
        data[base + 1, 3] = float(mat.get("ambient", 0.1))
        data[base + 2, 0] = float(mat.get("diffuse", 0.9))
        data[base + 2, 1] = float(mat.get("specular", 0.0))
        data[base + 2, 2] = float(mat.get("shininess", 50.0))
        data[base + 2, 3] = float(mat.get("reflective", 0.0))
        data[base + 3, 0] = float(mat.get("transparency", 0.0))
        data[base + 3, 1] = float(mat.get("refractive-index", 1.0))
        pat = mat.get("pattern", {})
        ptype = 0
        if isinstance(pat, dict):
            pat_type = pat.get("type")
            if pat_type == "checkered":
                ptype = 1
            elif pat_type == "gradient":
                ptype = 2
            elif pat_type == "ring":
                ptype = 3
            elif pat_type == "striped":
                ptype = 4
            elif pat_type == "blend":
                ptype = 5
        data[base + 3, 2] = float(ptype)

        # compute forward (model) and inverse matrices (numpy row-major)
        M = np.eye(4, dtype=np.float32)
        if "transform" in o:
            M = build_transform_matrix(o["transform"])
        Minv = np.linalg.inv(M).astype(np.float32)

        # Store inverse columns into slots base+4..base+7 (column-major)
        for c in range(4):
            data[base + 4 + c, 0] = Minv[0, c]
            data[base + 4 + c, 1] = Minv[1, c]
            data[base + 4 + c, 2] = Minv[2, c]
            data[base + 4 + c, 3] = Minv[3, c]

        # Store forward (model) columns into slots base+8..base+11
        for c in range(4):
            data[base + 8 + c, 0] = M[0, c]
            data[base + 8 + c, 1] = M[1, c]
            data[base + 8 + c, 2] = M[2, c]
            data[base + 8 + c, 3] = M[3, c]

        # Pack triangle vertices
        if typ == "triangle":
            p1 = np.array(o.get("p1", [0.0, 0.0, 0.0]), dtype=np.float32)
            p2 = np.array(o.get("p2", [1.0, 0.0, 0.0]), dtype=np.float32)
            p3 = np.array(o.get("p3", [0.0, 1.0, 0.0]), dtype=np.float32)
            data[base + 12, 0:3] = p1
            data[base + 13, 0:3] = p2
            data[base + 14, 0:3] = p3

        # Pack group children
        if typ == "group":
            data[base + 15, 0] = float(o.get("children_start", 0))
            data[base + 15, 1] = float(o.get("children_count", 0))

        # Pack pattern data
        a_color = np.array(pat.get("a_color", [1.0, 1.0, 1.0]), dtype=np.float32)
        b_color = np.array(pat.get("b_color", [0.0, 0.0, 0.0]), dtype=np.float32)
        data[base + 16, 0:3] = a_color
        data[base + 17, 0:3] = b_color

        # Pattern transform
        if "transform" in pat:
            pat_M = build_transform_matrix(pat["transform"])
            pat_Minv = np.linalg.inv(pat_M).astype(np.float32)
        else:
            pat_Minv = np.eye(4, dtype=np.float32)
        for c in range(4):
            data[base + 18 + c, 0] = pat_Minv[0, c]
            data[base + 18 + c, 1] = pat_Minv[1, c]
            data[base + 18 + c, 2] = pat_Minv[2, c]
            data[base + 18 + c, 3] = pat_Minv[3, c]
    return data, np.array(group_children, dtype=np.int32)


def build_camera(camera_json):
    w = camera_json["width"]
    h = camera_json["height"]
    fov_deg = camera_json.get("fov", 60.0)
    ft = camera_json.get("view_transform", {})
    cam_from = np.array(ft.get("from", [0, 1.5, -5]), dtype=np.float32)
    cam_to = np.array(ft.get("to", [0, 1, 0]), dtype=np.float32)
    cam_up = np.array(ft.get("up", [0, 1, 0]), dtype=np.float32)
    forward = cam_to - cam_from
    forward /= np.linalg.norm(forward)
    upn = cam_up / np.linalg.norm(cam_up)
    # Use right-handed cross product: up × forward = right
    right = np.cross(upn, forward)
    right /= np.linalg.norm(right)
    true_up = np.cross(forward, right)
    aspect = float(w) / float(h)
    fov = np.deg2rad(fov_deg)
    half_view = np.tan(fov / 2.0)

    if aspect >= 1:
        half_width = half_view
        half_height = half_view / aspect
    else:
        half_width = half_view * aspect
        half_height = half_view

    origin = cam_from
    lower_left = origin + forward - half_width * right - half_height * true_up
    horizontal = 2.0 * half_width * right
    vertical = 2.0 * half_height * true_up

    return {
        "width": int(w),
        "height": int(h),
        "fov": float(fov_deg),
        "origin": origin.astype(np.float32),
        "ll": lower_left.astype(np.float32),
        "hvec": horizontal.astype(np.float32),
        "vvec": vertical.astype(np.float32),
    }


# def compile_glsl_to_spv(glsl_path, spv_path):
#     gc = "./shaderc/bin/glslc"
#     cmd = None
#     if gc:
#         cmd = [gc, "-fshader-stage=compute", glsl_path, "-o", spv_path]
#     else:
#         raise RuntimeError("No GLSL compiler found (glslc).")
#     proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
#     if proc.returncode != 0:
#         raise RuntimeError(
#             f"Shader compile failed:\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
#         )
#     with open(spv_path, "rb") as f:
#         return f.read()


def main(scene_json_path):
    scene = json.load(open(scene_json_path, "r"))
    obj_buf, group_children = pack_scene(scene)
    num_objs = len(scene.get("objects", []))
    cam = build_camera(scene["camera"])
    cam_block = np.zeros((8, 4), dtype=np.float32)
    cam_block[0, 0] = cam["width"]
    cam_block[0, 1] = cam["height"]
    cam_block[0, 2] = cam["fov"]
    cam_block[0, 3] = float(num_objs)
    cam_block[1, 0:3] = cam["origin"]
    cam_block[2, 0:3] = cam["hvec"]
    cam_block[3, 0:3] = cam["vvec"]
    cam_block[4, 0:3] = cam["ll"]
    light = scene.get("light", {})
    light_pos = np.array(light.get("at", [-10, 10, -10]), dtype=np.float32)
    light_int = np.array(light.get("intensity", [1.0, 1.0, 1.0]), dtype=np.float32)
    cam_block[5, 0:3] = light_pos
    cam_block[6, 0:3] = light_int

    width = cam["width"]
    height = cam["height"]

    shader_path = os.path.join(os.path.dirname(__file__), SHADER_FILE)
    if not os.path.exists(shader_path):
        raise RuntimeError(f"{SHADER_FILE} not found")

    # make sure arrays are contiguous
    a0 = np.ascontiguousarray(obj_buf.ravel().astype(np.float32))
    a1 = np.ascontiguousarray(cam_block.ravel().astype(np.float32))

    if len(group_children) == 0:
        group_children = np.array([0], dtype=np.int32)  # dummy to avoid zero size

    a2 = np.ascontiguousarray(group_children.astype(np.int32))

    input_arrays = {0: a0, 1: a1, 3: a2}
    output_arrays = {2: (height * width, 4, "f")}

    outputs = compute_with_buffers(
        input_arrays,
        output_arrays,
        open(shader_path, "rb").read(),
        constants=None,
        n=((width + 7) // 8, (height + 7) // 8, 1),
    )
    mv = outputs[2]
    arr = np.frombuffer(mv, dtype=np.float32).reshape((-1, 4))
    img = np.clip(arr[: width * height, :3], 0.0, 1.0)
    img = (img * 255.0).astype(np.uint8).reshape((height, width, 3))

    def to_ppm(img, cam):
        """
        Returns a PPM (image format) string buffer representation of the canvas.
        """

        header = f"P3\n{cam['width']} {cam['height']}\n255\n"
        buf = StringIO()

        buf.write(header)

        for y, x in np.ndindex(cam["height"], cam["width"]):
            px = img[y, x]

            buf.write(f"{px[0]} {px[1]} {px[2]}\n")

        return buf

    with open("output.ppm", "w") as f:
        f.write(to_ppm(img, cam).getvalue())

    # Image.fromarray(img, mode="RGB").save(out_png)
    # print("Wrote", out_png)
