import argparse
import json
import os
from pathlib import Path

import imageio
import numpy as np

# Default to EGL if nothing else is configured; this matches the original behaviour.
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import pyrender  # noqa: E402
import shapely.geometry as geom  # noqa: E402
import trimesh  # noqa: E402


def resolve_camera_pose(polygon: geom.Polygon, wall_height: float):
    centroid = polygon.centroid
    centroid_xy = np.array([centroid.x, centroid.y], dtype=float)
    coords = np.array(polygon.exterior.coords[:-1], dtype=float)

    if coords.shape[0] < 3:
        direction = np.array([1.0, 0.0])
    else:
        deltas = coords - centroid_xy
        norms = np.linalg.norm(deltas, axis=1)
        if np.allclose(norms, 0.0):
            direction = np.array([1.0, 0.0])
        else:
            direction = deltas[np.argmax(norms)]
            direction /= np.linalg.norm(direction)

    offset = max(np.sqrt(max(polygon.area, 1e-6)), 1.25)
    eye_xy = centroid_xy + direction * offset
    eye_height = min(max(1.6, wall_height * 0.45), wall_height - 0.2)

    eye = np.array([eye_xy[0], eye_xy[1], eye_height])
    target_z = min(wall_height * 0.5, eye_height - 0.15)
    target = np.array([centroid_xy[0], centroid_xy[1], max(target_z, 0.5)])
    return eye, target


def build_camera_pose(eye, target):
    forward = target - eye
    norm = np.linalg.norm(forward)
    if norm < 1e-6:
        forward = np.array([0.0, 1.0, 0.0])
        norm = 1.0
    forward /= norm

    up = np.array([0.0, 0.0, 1.0])
    right = np.cross(up, forward)
    if np.linalg.norm(right) < 1e-6:
        up = np.array([0.0, 1.0, 0.0])
        right = np.cross(up, forward)
    right /= np.linalg.norm(right)
    up = np.cross(forward, right)

    pose = np.eye(4)
    pose[:3, 0] = right
    pose[:3, 1] = up
    pose[:3, 2] = forward
    pose[:3, 3] = eye
    return pose


def _scaled_ring(coords, scale: float):
    try:
        arr = np.asarray(coords, dtype=float)
    except (TypeError, ValueError):
        return None
    if arr.ndim != 2 or arr.shape[0] < 3:
        return None
    if arr.shape[1] < 2:
        return None
    if arr.shape[1] > 2:
        arr = arr[:, :2]
    return arr * scale


def _clean_polygon(polygon):
    if polygon is None or polygon.area <= 1e-8:
        return None
    if polygon.is_valid:
        return polygon

    fixed = polygon.buffer(0)
    if isinstance(fixed, geom.Polygon):
        polygon = fixed
    elif isinstance(fixed, geom.MultiPolygon):
        candidates = [p for p in fixed.geoms if p.area > 1e-8]
        if not candidates:
            return None
        polygon = max(candidates, key=lambda p: p.area)
    else:
        return None

    return polygon if polygon.area > 1e-8 else None


def build_room_polygon(points_data, scale: float):
    if points_data in (None, []):
        return None

    if isinstance(points_data, np.ndarray):
        return build_room_polygon(points_data.tolist(), scale)

    if isinstance(points_data, dict):
        coords = points_data.get("coordinates")
        if coords is not None:
            return build_room_polygon(coords, scale)
        shell = (
            points_data.get("exterior")
            or points_data.get("shell")
            or points_data.get("outer")
        )
        if shell is None:
            return None
        shell_ring = _scaled_ring(shell, scale)
        if shell_ring is None:
            return None
        holes_data = (
            points_data.get("holes")
            or points_data.get("interiors")
            or points_data.get("inner")
        )
        holes = []
        if holes_data:
            for hole in holes_data:
                ring = _scaled_ring(hole, scale)
                if ring is not None:
                    holes.append(ring)
        try:
            polygon = geom.Polygon(shell_ring, holes if holes else None)
        except (TypeError, ValueError):
            return None
        return _clean_polygon(polygon)

    if not isinstance(points_data, (list, tuple)):
        return None

    ring = _scaled_ring(points_data, scale)
    if ring is not None:
        try:
            polygon = geom.Polygon(ring)
        except (TypeError, ValueError):
            polygon = None
        return _clean_polygon(polygon)

    polygons = []
    for part in points_data:
        polygon = build_room_polygon(part, scale)
        if polygon is not None:
            polygons.append(polygon)
    polygons = [poly for poly in polygons if poly.area > 1e-8]
    if not polygons:
        return None
    return max(polygons, key=lambda poly: poly.area)


def render_room_views(mesh_path: Path, polygons_path: Path, output_dir: Path, scale: float, wall_height: float, width: int, height: int):
    with polygons_path.open("r", encoding="utf-8") as fp:
        data = json.load(fp)

    rooms = data.get("rooms", [])
    if not rooms:
        print("[render_rooms] No rooms found; skipping render.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    trimesh_mesh = trimesh.load(mesh_path, force="scene")
    if isinstance(trimesh_mesh, trimesh.Scene):
        combined = trimesh.util.concatenate(trimesh_mesh.dump())
    else:
        combined = trimesh_mesh

    render_mesh = pyrender.Mesh.from_trimesh(combined, smooth=False)
    renderer = pyrender.OffscreenRenderer(width, height)

    for idx, room in enumerate(rooms):
        polygon = build_room_polygon(room.get("points"), scale)
        if polygon is None or polygon.area <= 1e-6:
            continue

        eye, target = resolve_camera_pose(polygon, wall_height)
        camera_pose = build_camera_pose(eye, target)

        scene = pyrender.Scene(bg_color=[1.0, 1.0, 1.0, 1.0], ambient_light=[0.35, 0.35, 0.35, 1.0])
        scene.add(render_mesh)

        camera = pyrender.PerspectiveCamera(yfov=np.deg2rad(60.0), aspectRatio=float(width) / float(height))
        scene.add(camera, pose=camera_pose)

        light_pose = camera_pose.copy()
        light_pose[2, 3] = max(light_pose[2, 3], wall_height - 0.1)
        scene.add(pyrender.PointLight(color=np.ones(3), intensity=1000.0), pose=light_pose)

        try:
            color, _ = renderer.render(scene)
        except Exception as exc:
            print(f"[render_rooms] Failed to render room {idx}: {exc}")
            continue

        image = np.clip(color, 0, 255).astype(np.uint8)
        room_label = room.get("class", f"room_{idx}")
        slug = room_label.replace(" ", "_").lower()
        output_path = output_dir / f"{idx:02d}_{slug}.png"
        imageio.imwrite(output_path, image)
        print(f"[render_rooms] Saved {output_path}")

    renderer.delete()


def main():
    parser = argparse.ArgumentParser(description="Render simple per-room views from GLB mesh.")
    parser.add_argument("--mesh", type=Path, required=True, help="Path to the GLB/mesh file.")
    parser.add_argument("--polygons", type=Path, required=True, help="Path to polygons.json.")
    parser.add_argument("--output", type=Path, required=True, help="Directory for rendered images.")
    parser.add_argument("--scale", type=float, required=True, help="Pixel-to-unit scale used in mesh.")
    parser.add_argument("--wall-height", type=float, default=3.0, help="Wall height used for extrusion.")
    parser.add_argument("--image-width", type=int, default=1280, help="Render width in pixels.")
    parser.add_argument("--image-height", type=int, default=720, help="Render height in pixels.")
    args = parser.parse_args()

    render_room_views(
        args.mesh,
        args.polygons,
        args.output,
        args.scale,
        args.wall_height,
        args.image_width,
        args.image_height,
    )


if __name__ == "__main__":
    main()
