import argparse
import json
import math
import os
from pathlib import Path

import imageio
import numpy as np

# Default to EGL if nothing else is configured; this matches the original behaviour.
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import pyrender  # noqa: E402
import shapely.geometry as geom  # noqa: E402
import trimesh  # noqa: E402


def resolve_camera_pose(
    polygon: geom.Polygon,
    wall_height: float,
    vertical_fov: float = math.radians(70.0),
    aspect_ratio: float = 16.0 / 9.0,
):
    adaptive = _resolve_camera_pose_adaptive(polygon, wall_height, vertical_fov, aspect_ratio)
    if adaptive is not None:
        return adaptive
    return _resolve_camera_pose_basic(polygon, wall_height)


def _resolve_camera_pose_adaptive(
    polygon: geom.Polygon,
    wall_height: float,
    vertical_fov: float,
    aspect_ratio: float,
):
    if polygon is None or polygon.is_empty:
        return None

    representative = polygon.representative_point()
    centroid_xy = np.array([representative.x, representative.y], dtype=float)

    try:
        coords = np.array(polygon.exterior.coords[:-1], dtype=float)
    except (AttributeError, TypeError, ValueError):
        return None

    if coords.size == 0:
        return None

    minx, miny, maxx, maxy = polygon.bounds
    diag = np.linalg.norm([maxx - minx, maxy - miny])
    if not math.isfinite(diag) or diag <= 0.0:
        diag = max(maxx - minx, maxy - miny, 1.0)

    horizontal_fov = 2.0 * math.atan(math.tan(vertical_fov / 2.0) * aspect_ratio)
    horizontal_fov = float(np.clip(horizontal_fov, math.radians(30.0), math.radians(140.0)))

    best = None
    candidate_dirs = _polygon_direction_candidates(polygon, coords, centroid_xy)
    for direction in candidate_dirs:
        evaluation = _evaluate_direction(
            polygon,
            coords,
            centroid_xy,
            direction,
            diag,
            wall_height,
            vertical_fov,
            horizontal_fov,
        )
        if evaluation is None:
            continue
        if best is None or evaluation["score"] > best["score"]:
            best = evaluation

    if best is None:
        return None

    eye_xy = best["eye_xy"]
    target_xy = best["target_xy"]
    forward_dir = best["forward_dir"]

    eye_height = min(max(1.55, wall_height * 0.45), max(wall_height - 0.2, 1.8))
    target_depth = np.dot(target_xy - eye_xy, forward_dir)
    vertical_focus = max(target_depth, 0.4)
    floor_angle = math.atan2(eye_height, vertical_focus)
    ceiling_angle = math.atan2(max(wall_height - eye_height, 0.1), vertical_focus)
    max_vertical_span = max(floor_angle, ceiling_angle)
    if vertical_fov < 2.0 * max_vertical_span:
        eye_height = min(max(1.35, wall_height * 0.35), max(wall_height - 0.25, 1.65))

    target_z = min(wall_height * 0.55, eye_height - 0.18)
    if target_z < 0.65:
        target_z = 0.65

    eye = np.array([eye_xy[0], eye_xy[1], eye_height])
    target = np.array([target_xy[0], target_xy[1], target_z])
    return eye, target


def _resolve_camera_pose_basic(polygon: geom.Polygon, wall_height: float):
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


def _polygon_direction_candidates(polygon: geom.Polygon, coords: np.ndarray, centroid_xy: np.ndarray):
    candidates = list(_candidate_directions(coords, centroid_xy))
    try:
        oriented = polygon.minimum_rotated_rectangle
    except (AttributeError, ValueError):
        oriented = None
    if isinstance(oriented, geom.Polygon):
        rect_coords = np.asarray(oriented.exterior.coords[:-1], dtype=float)
        if rect_coords.ndim == 2 and rect_coords.shape[0] >= 2:
            for idx in range(rect_coords.shape[0]):
                current = rect_coords[idx]
                nxt = rect_coords[(idx + 1) % rect_coords.shape[0]]
                edge = nxt - current
                if np.linalg.norm(edge) < 1e-6:
                    continue
                candidates.append(edge)
                candidates.append(np.array([-edge[1], edge[0]], dtype=float))
    unique = _unique_directions(candidates)
    return unique if unique else [np.array([1.0, 0.0], dtype=float)]


def _point_in_polygon(polygon: geom.Polygon, xy: np.ndarray):
    point = geom.Point(float(xy[0]), float(xy[1]))
    return polygon.contains(point) or polygon.touches(point)


def _longest_line_segment(geometry_obj):
    segments = []

    def _collect(item):
        if item.is_empty:
            return
        if isinstance(item, geom.LineString):
            coords = np.asarray(item.coords, dtype=float)
            if coords.ndim == 2 and coords.shape[0] >= 2:
                segments.append((coords[0], coords[-1]))
        elif isinstance(item, geom.MultiLineString):
            for part in item.geoms:
                _collect(part)
        elif isinstance(item, geom.GeometryCollection):
            for part in item.geoms:
                _collect(part)

    _collect(geometry_obj)
    if not segments:
        return None

    lengths = [np.linalg.norm(end - start) for start, end in segments]
    best_idx = int(np.argmax(lengths))
    start, end = segments[best_idx]
    return np.array(start, dtype=float), np.array(end, dtype=float)


def _sample_polygon_points(polygon: geom.Polygon, target_count: int):
    points = []
    try:
        exterior_coords = list(polygon.exterior.coords)
    except AttributeError:
        return points

    for x, y in exterior_coords[:-1]:
        points.append(np.array([x, y], dtype=float))

    if target_count > 0:
        length = polygon.exterior.length
        if length > 1e-6:
            distances = np.linspace(0.0, length, num=target_count, endpoint=False)
            for dist in distances:
                pt = polygon.exterior.interpolate(dist)
                points.append(np.array([pt.x, pt.y], dtype=float))

    for interior in polygon.interiors:
        interior_coords = list(interior.coords)
        for x, y in interior_coords[:-1]:
            points.append(np.array([x, y], dtype=float))

    return points


def _evaluate_direction(
    polygon: geom.Polygon,
    coords: np.ndarray,
    centroid_xy: np.ndarray,
    direction: np.ndarray,
    diag: float,
    wall_height: float,
    vertical_fov: float,
    horizontal_fov: float,
):
    norm = np.linalg.norm(direction)
    if norm < 1e-6:
        return None
    forward_dir = direction / norm
    perp_dir = np.array([-forward_dir[1], forward_dir[0]])

    extent = max(diag * 1.6, 3.5)
    line = geom.LineString(
        [
            tuple(centroid_xy - forward_dir * extent),
            tuple(centroid_xy + forward_dir * extent),
        ]
    )

    segment = _longest_line_segment(polygon.intersection(line))
    if segment is None:
        return None

    start, end = segment
    seg_vec = end - start
    seg_len = np.linalg.norm(seg_vec)
    if seg_len < 1e-3:
        return None
    if np.dot(seg_vec, forward_dir) < 0.0:
        start, end = end, start
        seg_vec = end - start
        seg_len = np.linalg.norm(seg_vec)
        if seg_len < 1e-3:
            return None

    base_eye = float(np.clip(seg_len * 0.08, 0.2, 0.45))
    offsets = [
        base_eye,
        base_eye * 0.7,
        base_eye * 0.5,
        max(seg_len * 0.06, 0.12),
    ]

    eye_xy = None
    for off in offsets:
        off = float(np.clip(off, seg_len * 0.04, seg_len * 0.42))
        candidate = start + forward_dir * off
        if _point_in_polygon(polygon, candidate):
            eye_xy = candidate
            break
    if eye_xy is None:
        return None

    eye_offset = np.dot(eye_xy - start, forward_dir)
    front_guard = float(np.clip(seg_len * 0.12, 0.18, 0.7))
    target_min = max(eye_offset + seg_len * 0.35, eye_offset + 0.55)
    target_max = seg_len - front_guard
    if target_max <= target_min:
        target_max = seg_len - max(front_guard * 0.5, 0.12)
    if target_max <= eye_offset + 0.1:
        target_max = seg_len - 0.08
    if target_max <= eye_offset:
        target_max = seg_len * 0.85

    target_offset = min(max(target_min, eye_offset + 0.3), target_max)
    target_offset = float(np.clip(target_offset, eye_offset + 0.15, seg_len - 0.05))
    target_xy = start + forward_dir * target_offset

    if not _point_in_polygon(polygon, target_xy):
        adjust_vec = centroid_xy - target_xy
        adjusted = False
        for frac in (0.6, 0.45, 0.3):
            candidate = target_xy + adjust_vec * frac
            if _point_in_polygon(polygon, candidate):
                target_xy = candidate
                adjusted = True
                break
        if not adjusted:
            return None

    target_depth = np.dot(target_xy - eye_xy, forward_dir)
    if target_depth <= 0.12:
        return None

    sample_count = max(int(seg_len * 18), 48)
    boundary_pts = _sample_polygon_points(polygon, sample_count)
    if not boundary_pts:
        boundary_pts = [np.array([x, y], dtype=float) for x, y in polygon.exterior.coords[:-1]]

    max_angle = 0.0
    min_depth = float("inf")
    for point in boundary_pts:
        vec = point - eye_xy
        depth = np.dot(vec, forward_dir)
        if depth <= 1e-3:
            return None
        perp = np.dot(vec, perp_dir)
        angle = math.atan2(abs(perp), depth)
        if angle > max_angle:
            max_angle = angle
        if depth < min_depth:
            min_depth = depth

    horizontal_margin = horizontal_fov / 2.0 - max_angle

    eye_height = min(max(1.55, wall_height * 0.45), max(wall_height - 0.2, 1.8))
    floor_angle = math.atan2(eye_height, target_depth)
    ceiling_angle = math.atan2(max(wall_height - eye_height, 0.1), target_depth)
    vertical_margin = vertical_fov / 2.0 - max(floor_angle, ceiling_angle)

    score = horizontal_margin + 0.05 * min_depth + 0.15 * max(vertical_margin, -1.5)
    return {
        "eye_xy": eye_xy,
        "target_xy": target_xy,
        "forward_dir": forward_dir,
        "horizontal_margin": horizontal_margin,
        "vertical_margin": vertical_margin,
        "min_depth": min_depth,
        "score": score,
    }


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
    vertical_fov = float(camera.yfov)
    aspect_ratio = float(width) / float(height)

    for idx, room in enumerate(rooms):
        polygon = build_room_polygon(room.get("points"), scale)
        if polygon is None or polygon.area <= 1e-6:
            continue

        eye, target = resolve_camera_pose(polygon, wall_height, vertical_fov, aspect_ratio)
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
