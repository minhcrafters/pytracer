from core.scene_config import SceneConfig
import pprint, sys

cfg = SceneConfig("examples/cover.yml")

print("== defines ==")
pprint.pprint(cfg.defines)
print("== beginning staged parse ==")

for i, entry in enumerate(cfg.entries):
    print(f"\n--- entry #{i} ---")
    pprint.pprint(entry)
    try:
        if isinstance(entry, dict) and "add" in entry:
            if entry["add"] == "camera":
                cfg.camera = cfg._parse_camera(entry)
            elif entry["add"] == "light":
                cfg.light = cfg._parse_light(entry)
            else:
                obj = cfg._parse_object(entry)
                if obj:
                    cfg.objects.append(obj)
    except Exception as e:
        print("Exception during parsing this entry:", repr(e))
    print(
        "-> post-entry state: camera:",
        bool(cfg.camera),
        "light:",
        getattr(cfg.light, "intensity", None),
    )
