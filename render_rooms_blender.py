import argparse
import json
import math
import os
from pathlib import Path

import bpy
import numpy as np
import shapely.geometry as geom
from mathutils import Matrix, Vector


def reset_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.context.scene.render.engine = "BLENDER_EEVEE"
    bpy.context.scene.eevee.taa_samples = 16
    bpy.context.scene.eevee.taa_render_samples = 32
    bpy.context.scene.eevee.use_gtao = True
    bpy.context.scene.eevee.use_ssr = True
    bpy.context.scene.eevee.use_bloom = True
    world = bpy.context.scene.world
    world.use_nodes = True
    bg = world.node_tree.nodes["Background"]
    bg.inputs[0].default_value = (0.96, 0.97, 1.0, 1.0)
    bg.inputs[1].default_value = 0.9


def load_mesh(mesh_path: Path):
    bpy.ops.import_scene.gltf(filepath=str(mesh_path))
    collection_objects = [
        obj for obj in bpy.context.scene.objects if obj.type in {"MESH", "CURVE"}
    ]
    for obj in collection_objects:
        obj.select_set(True)
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    return collection_objects


def bounding_box(objects):
    mins = Vector((math.inf, math.inf, math.inf))
    maxs = Vector((-math.inf, -math.inf, -math.inf))
    for obj in objects:
        for corner in obj.bound_box:
            world_corner = obj.matrix_world @ Vector(corner)
            mins.x = min(mins.x, world_corner.x)
            mins.y = min(mins.y, world_corner.y)
            mins.z = min(mins.z, world_corner.z)
            maxs.x = max(maxs.x, world_corner.x)
            maxs.y = max(maxs.y, world_corner.y)
            maxs.z = max(maxs.z, world_corner.z)
    return mins, maxs


def ensure_camera(name="RenderCamera"):
    if name in bpy.data.objects:
        cam_obj = bpy.data.objects[name]
        cam_data = cam_obj.data
    else:
        cam_data = bpy.data.cameras.new(name)
        cam_obj = bpy.data.objects.new(name, cam_data)
        bpy.context.scene.collection.objects.link(cam_obj)
    bpy.context.scene.camera = cam_obj
    cam_data.lens = 28.0
    cam_data.clip_start = 0.01
    cam_data.clip_end = 250.0
    return cam_obj


def look_at(camera_obj, target):
    direction = target - camera_obj.location
    direction.normalize()
    up = Vector((0.0, 0.0, 1.0))
    if abs(direction.dot(up)) > 0.999:
        up = Vector((0.0, 1.0, 0.0))
    right = up.cross(direction)
    right.normalize()
    up = direction.cross(right)
    rot = Matrix(
        (
            right.to_4d(),
            up.to_4d(),
            (-direction).to_4d(),
            Vector((0.0, 0.0, 0.0, 1.0)),
        )
    )
    camera_obj.matrix_world = rot


def generate_camera_poses(polygon: geom.Polygon, wall_height: float, n_views: int = 3):
    if polygon.is_empty or polygon.area <= 1e-6:
        return []

    centroid = np.array(polygon.centroid.coords[0], dtype=float)
    coords = np.array(polygon.exterior.coords[:-1], dtype=float)
    centered = coords - centroid
    if centered.shape[0] < 3:
        base_dir = np.array([1.0, 0.0])
    else:
        cov = centered.T @ centered
        eigvals, eigvecs = np.linalg.eigh(cov)
        base_dir = eigvecs[:, np.argmax(eigvals)]
    base_dir = base_dir / np.linalg.norm(base_dir)

    area = polygon.area
    nominal_radius = max(math.sqrt(area / math.pi), 1.0)
    max_distance = min(max(nominal_radius * 1.3, 1.8), 4.0)
    buffer_poly = polygon.buffer(-0.15)

    poses = []
    for angle in np.linspace(0.0, 2.0 * math.pi, num=max(n_views, 1), endpoint=False):
        rot = np.array(
            [
                [math.cos(angle), -math.sin(angle)],
                [math.sin(angle), math.cos(angle)],
            ]
        )
        direction = rot @ base_dir
        direction = direction / np.linalg.norm(direction)
        candidates = [
            centroid + direction * max_distance,
            centroid + direction * (max_distance * 0.65),
            centroid - direction * (max_distance * 0.4),
            centroid,
        ]
        eye_xy = centroid
        for candidate in candidates:
            point = geom.Point(candidate[0], candidate[1])
            if buffer_poly.is_empty or buffer_poly.contains(point):
                eye_xy = candidate
                break

        eye_height = min(max(1.5, wall_height * 0.45), wall_height - 0.2)
        target_height = min(wall_height * 0.5, eye_height - 0.1)
        eye = Vector((float(eye_xy[0]), float(eye_xy[1]), eye_height))
        target = Vector((float(centroid[0]), float(centroid[1]), max(target_height, 0.5)))
        poses.append((eye, target))

    if not poses:
        eye_height = min(max(1.5, wall_height * 0.45), wall_height - 0.2)
        eye = Vector((float(centroid[0]), float(centroid[1]), eye_height))
        target = Vector((float(centroid[0]), float(centroid[1]), eye_height - 0.1))
        poses.append((eye, target))
    return poses


def configure_lighting(bbox_min: Vector, bbox_max: Vector):
    center = (bbox_min + bbox_max) * 0.5
    size = bbox_max - bbox_min
    radius = max(size.x, size.y)

    bpy.ops.object.light_add(type="SUN", location=(center.x, center.y, center.z + radius * 1.5))
    sun = bpy.context.object
    sun.data.energy = 2.5
    sun.rotation_euler = (math.radians(55.0), math.radians(15.0), math.radians(45.0))

    bpy.ops.object.light_add(type="AREA", location=(center.x, center.y, bbox_max.z + radius * 0.3))
    area = bpy.context.object
    area.data.energy = 1500.0
    area.data.size = radius * 1.2


def render_views(
    mesh_path: Path,
    polygons_path: Path,
    output_dir: Path,
    scale: float,
    wall_height: float,
    width: int,
    height: int,
    views_per_room: int,
    topdown_only: bool,
):
    reset_scene()
    objects = load_mesh(mesh_path)
    bbox_min, bbox_max = bounding_box(objects)
    configure_lighting(bbox_min, bbox_max)
    camera = ensure_camera()

    scene = bpy.context.scene
    scene.render.image_settings.file_format = "PNG"
    scene.render.resolution_x = width
    scene.render.resolution_y = height
    scene.render.resolution_percentage = 100

    output_dir.mkdir(parents=True, exist_ok=True)

    with polygons_path.open("r", encoding="utf-8") as fp:
        data = json.load(fp)

    rooms = data.get("rooms", [])
    if not rooms:
        print("[render_rooms_blender] No rooms found in polygons.json")
        return

    if topdown_only:
        camera.location = Vector(
            (
                float((bbox_min.x + bbox_max.x) * 0.5),
                float((bbox_min.y + bbox_max.y) * 0.5),
                float(bbox_max.z + max(bbox_max.x - bbox_min.x, bbox_max.y - bbox_min.y)),
            )
        )
        target = Vector(
            (
                float((bbox_min.x + bbox_max.x) * 0.5),
                float((bbox_min.y + bbox_max.y) * 0.5),
                float((bbox_min.z + bbox_max.z) * 0.5),
            )
        )
        look_at(camera, target)
        output_path = output_dir / "overview_topdown.png"
        scene.render.filepath = str(output_path)
        bpy.ops.render.render(write_still=True)
        print(f"[render_rooms_blender] Saved overview render -> {output_path}")
        return

    for idx, room in enumerate(rooms):
        points = room.get("points")
        if not points:
            continue
        polygon = geom.Polygon(points)
        if polygon.is_empty or polygon.area <= 1e-6:
            continue

        polygon_scaled = geom.Polygon(np.array(points, dtype=float) * scale)
        poses = generate_camera_poses(polygon_scaled, wall_height, n_views=views_per_room)
        label = room.get("class", f"room_{idx}")
        slug = label.replace(" ", "_").lower()

        for view_idx, (eye, target) in enumerate(poses, start=1):
            camera.location = eye
            look_at(camera, target)
            scene.render.filepath = str(output_dir / f"{idx:02d}_{slug}_v{view_idx}.png")
            bpy.ops.render.render(write_still=True)
            print(f"[render_rooms_blender] Saved render for {label} [view {view_idx}]")


def parse_args():
    parser = argparse.ArgumentParser(description="Fallback Blender renderer for room interiors.")
    parser.add_argument("--mesh", type=Path, required=True, help="Path to GLB mesh (walls.glb).")
    parser.add_argument("--polygons", type=Path, required=True, help="Path to polygons.json.")
    parser.add_argument("--output", type=Path, required=True, help="Directory for renders.")
    parser.add_argument("--scale", type=float, required=True, help="Pixel to unit scale used in mesh.")
    parser.add_argument("--wall-height", type=float, default=3.0, help="Wall extrusion height.")
    parser.add_argument("--image-width", type=int, default=1280, help="Render width.")
    parser.add_argument("--image-height", type=int, default=720, help="Render height.")
    parser.add_argument("--views-per-room", type=int, default=3, help="Number of camera angles per room.")
    parser.add_argument(
        "--topdown-only",
        action="store_true",
        help="Generate a single top-down overview render instead of per-room views.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    render_views(
        mesh_path=args.mesh,
        polygons_path=args.polygons,
        output_dir=args.output,
        scale=args.scale,
        wall_height=args.wall_height,
        width=args.image_width,
        height=args.image_height,
        views_per_room=max(1, args.views_per_room),
        topdown_only=args.topdown_only,
    )


if __name__ == "__main__":
    main()
