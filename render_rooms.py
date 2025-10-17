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
    representative = polygon.representative_point()
    centroid_xy = np.array([representative.x, representative.y], dtype=float)
    coords = np.array(polygon.exterior.coords[:-1], dtype=float)

    minx, miny, maxx, maxy = polygon.bounds
    diag = np.linalg.norm([maxx - minx, maxy - miny])
    max_distance = max(diag * 1.5, 1.0)

    best_choice = None
    best_span = -np.inf
    for direction in _candidate_directions(coords, centroid_xy):
        forward = _ray_polygon_distance(polygon, centroid_xy, direction, max_distance)
        backward = _ray_polygon_distance(polygon, centroid_xy, -direction, max_distance)
        if forward is None or backward is None:
            continue
        span = forward + backward
        if span <= 1e-3:
            continue
        margin = max(0.25, min(wall_height * 0.15, backward * 0.25))
        usable_back = max(backward - margin, backward * 0.5)
        if usable_back <= 0.1:
            continue
        eye_xy = _find_interior_point(polygon, centroid_xy, direction, usable_back)
        test_point = geom.Point(eye_xy[0], eye_xy[1])
        if not polygon.contains(test_point) and not polygon.touches(test_point):
            continue
        if span > best_span:
            best_span = span
            best_choice = (direction, forward, backward, eye_xy)

    if best_choice is None:
        direction = np.array([1.0, 0.0])
        forward = backward = diag * 0.5 if diag > 0 else 1.0
        eye_xy = centroid_xy - direction * min(0.5, backward * 0.5)
    else:
        direction, forward, backward, eye_xy = best_choice

    eye_height = min(max(1.55, wall_height * 0.45), max(wall_height - 0.2, 1.8))
    target_offset = min(forward * 0.6, max(0.75, forward - 0.25))
    target_xy = centroid_xy + direction * target_offset * 0.5
    target_point = geom.Point(target_xy[0], target_xy[1])
    if not polygon.contains(target_point) and not polygon.touches(target_point):
        target_xy = centroid_xy

    eye = np.array([eye_xy[0], eye_xy[1], eye_height])
    target_z = min(wall_height * 0.55, eye_height - 0.2)
    target = np.array([target_xy[0], target_xy[1], max(target_z, 0.65)])
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


def _ray_polygon_distance(polygon: geom.Polygon, origin: np.ndarray, direction: np.ndarray, max_distance: float):
    direction_norm = np.linalg.norm(direction)
    if direction_norm < 1e-6:
        return None
    unit_dir = direction / direction_norm
    far_point = origin + unit_dir * max_distance
    ray = geom.LineString([tuple(origin), tuple(far_point)])
    intersection = polygon.boundary.intersection(ray)

    def extract_distances(geom_obj):
        distances = []
        if geom_obj.is_empty:
            return distances
        if isinstance(geom_obj, geom.Point):
            vec = np.array([geom_obj.x, geom_obj.y], dtype=float) - origin
            proj = np.dot(vec, unit_dir)
            if proj > 1e-4:
                distances.append(proj)
            return distances
        if hasattr(geom_obj, "geoms"):
            for part in geom_obj.geoms:
                distances.extend(extract_distances(part))
            return distances
        if isinstance(geom_obj, geom.LineString):
            for x, y in geom_obj.coords:
                vec = np.array([x, y], dtype=float) - origin
                proj = np.dot(vec, unit_dir)
                if proj > 1e-4:
                    distances.append(proj)
        return distances

    distances = extract_distances(intersection)
    if not distances:
        return None
    return min(distances)


def _unique_directions(candidates):
    result = []
    for vec in candidates:
        norm = np.linalg.norm(vec)
        if norm < 1e-6:
            continue
        unit = vec / norm
        if any(np.allclose(unit, existing) or np.allclose(unit, -existing) for existing in result):
            continue
        result.append(unit)
    return result


def _candidate_directions(coords: np.ndarray, centroid_xy: np.ndarray):
    candidates = []
    if coords.shape[0] >= 3:
        centered = coords - centroid_xy
        cov = np.cov(centered.T)
        try:
            eigvals, eigvecs = np.linalg.eigh(cov)
            order = np.argsort(eigvals)[::-1]
            for idx in order:
                candidates.append(eigvecs[:, idx])
        except np.linalg.LinAlgError:
            pass
    candidates.extend([np.array([1.0, 0.0]), np.array([0.0, 1.0])])

    unique = _unique_directions(candidates)
    if not unique:
        unique = [np.array([1.0, 0.0])]
    perp = np.array([-unique[0][1], unique[0][0]])
    return _unique_directions(unique + [perp])


def _find_interior_point(polygon: geom.Polygon, centroid_xy: np.ndarray, direction: np.ndarray, distance: float):
    for scale in np.linspace(1.0, 0.1, 8):
        candidate = centroid_xy - direction * (distance * scale)
        point = geom.Point(candidate[0], candidate[1])
        if polygon.contains(point) or polygon.touches(point):
            return candidate
    return centroid_xy


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
    scene = pyrender.Scene(bg_color=[0.94, 0.94, 0.94, 1.0], ambient_light=[0.32, 0.32, 0.32, 1.0])
    scene.add(render_mesh)

    zfar = max(wall_height * 6.0, 25.0)
    camera = pyrender.PerspectiveCamera(yfov=np.deg2rad(70.0), aspectRatio=float(width) / float(height), znear=0.05, zfar=zfar)
    camera_node = scene.add(camera, pose=np.eye(4))

    key_light = pyrender.PointLight(color=np.ones(3), intensity=600.0)
    key_node = scene.add(key_light, pose=np.eye(4))

    fill_light = pyrender.PointLight(color=np.array([0.9, 0.95, 1.0]), intensity=280.0)
    fill_node = scene.add(fill_light, pose=np.eye(4))

    render_flags = pyrender.RenderFlags.RGBA | pyrender.RenderFlags.SKIP_CULL_FACES

    for idx, room in enumerate(rooms):
        polygon = build_room_polygon(room.get("points"), scale)
        if polygon is None or polygon.area <= 1e-6:
            continue

        eye, target = resolve_camera_pose(polygon, wall_height)
        camera_pose = build_camera_pose(eye, target)

        scene.set_pose(camera_node, camera_pose)

        key_pose = camera_pose.copy()
        key_pose[:3, 3] = eye + np.array([0.0, 0.0, wall_height * 0.25])
        scene.set_pose(key_node, key_pose)

        fill_pose = np.eye(4)
        fill_pose[:3, 3] = np.array([target[0], target[1], wall_height * 0.85])
        scene.set_pose(fill_node, fill_pose)

        try:
            color, _ = renderer.render(scene, flags=render_flags)
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
