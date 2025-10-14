import argparse
import json
import os
from pathlib import Path

import imageio
import numpy as np

DEFAULT_GL_BACKENDS = ("egl", "osmesa")
if "PYOPENGL_PLATFORM" not in os.environ:
    os.environ["PYOPENGL_PLATFORM"] = DEFAULT_GL_BACKENDS[0]

import pyrender
from pyrender import RenderFlags
import shapely.geometry as geom
import trimesh


def _principal_directions(polygon, n_views):
    centroid = polygon.centroid
    centroid_xy = np.array([centroid.x, centroid.y])
    coords = np.array(polygon.exterior.coords[:-1])
    if coords.shape[0] < 3:
        base_dir = np.array([1.0, 0.0])
    else:
        centered = coords - centroid_xy
        cov = centered.T @ centered
        if np.linalg.norm(cov) < 1e-6:
            base_dir = np.array([1.0, 0.0])
        else:
            eigvals, eigvecs = np.linalg.eigh(cov)
            base_dir = eigvecs[:, np.argmax(eigvals)]
    base_dir = base_dir / np.linalg.norm(base_dir)

    angles = np.linspace(0.0, 2.0 * np.pi, num=max(n_views, 1), endpoint=False)
    directions = []
    for angle in angles:
        rot = np.array(
            [
                [np.cos(angle), -np.sin(angle)],
                [np.sin(angle), np.cos(angle)],
            ]
        )
        directions.append(rot @ base_dir)
    return centroid_xy, directions


def generate_camera_poses(polygon, wall_height, n_views=3):
    if polygon.area <= 1e-6:
        return []
    centroid_xy, directions = _principal_directions(polygon, n_views)

    area = polygon.area
    nominal_radius = max(np.sqrt(area / np.pi), 1.0)
    max_distance = min(max(nominal_radius * 1.3, 1.8), 3.8)

    buffer_poly = polygon.buffer(-0.15)
    poses = []
    for direction in directions:
        direction = direction / np.linalg.norm(direction)
        candidates = [
            centroid_xy + direction * max_distance,
            centroid_xy + direction * max_distance * 0.65,
            centroid_xy - direction * max_distance * 0.4,
            centroid_xy,
        ]
        eye_xy = centroid_xy
        for candidate in candidates:
            point = geom.Point(candidate[0], candidate[1])
            if buffer_poly.is_empty or buffer_poly.contains(point):
                eye_xy = candidate
                break

        eye_height = min(max(1.5, wall_height * 0.45), wall_height - 0.25)
        eye = np.array([eye_xy[0], eye_xy[1], eye_height])
        target_height = min(wall_height * 0.5, eye_height - 0.1)
        target = np.array([centroid_xy[0], centroid_xy[1], max(target_height, 0.5)])

        poses.append((eye, target))

    # Ensure at least one pose using centroid fallback if all duplicates
    if not poses:
        eye_height = min(max(1.5, wall_height * 0.45), wall_height - 0.25)
        eye = np.array([centroid_xy[0], centroid_xy[1], eye_height])
        target = np.array([centroid_xy[0], centroid_xy[1], eye_height - 0.1])
        poses.append((eye, target))
    return poses


def make_renderer(width: int, height: int):
    backends = []
    current = os.environ.get("PYOPENGL_PLATFORM")
    if current:
        backends.append(current)
    for candidate in DEFAULT_GL_BACKENDS:
        if candidate not in backends:
            backends.append(candidate)
    if "headless" not in backends:
        backends.append("headless")

    last_exc = None
    for backend in backends:
        os.environ["PYOPENGL_PLATFORM"] = backend
        try:
            renderer = pyrender.OffscreenRenderer(width, height)
            try:
                renderer.render(pyrender.Scene())
            except Exception:
                renderer.delete()
                raise
            return backend, renderer
        except Exception as exc:
            last_exc = exc

    os.environ["PYOPENGL_PLATFORM"] = DEFAULT_GL_BACKENDS[0]
    raise RuntimeError("Unable to initialise an offscreen OpenGL context") from last_exc


def build_camera_pose(eye, target, up_vector=np.array([0.0, 0.0, 1.0])):
    forward = target - eye
    norm = np.linalg.norm(forward)
    if norm < 1e-6:
        forward = np.array([0.0, 1.0, 0.0])
        norm = 1.0
    forward /= norm
    z_axis = -forward
    x_axis = np.cross(up_vector, z_axis)
    if np.linalg.norm(x_axis) < 1e-6:
        up_vector = np.array([0.0, 1.0, 0.0])
        x_axis = np.cross(up_vector, z_axis)
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)

    pose = np.eye(4)
    pose[:3, 0] = x_axis
    pose[:3, 1] = y_axis
    pose[:3, 2] = z_axis
    pose[:3, 3] = eye
    return pose


def render_room_views(
    mesh_path: Path,
    polygons_path: Path,
    output_dir: Path,
    scale: float,
    wall_height: float,
    width: int,
    height: int,
):
    with polygons_path.open("r", encoding="utf-8") as fp:
        data = json.load(fp)

    rooms = data.get("rooms", [])
    if not rooms:
        print("No rooms found for rendering.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    trimesh_mesh = trimesh.load(mesh_path, force="scene")
    if isinstance(trimesh_mesh, trimesh.Scene):
        combined = trimesh.util.concatenate(trimesh_mesh.dump())
    else:
        combined = trimesh_mesh

    render_mesh = pyrender.Mesh.from_trimesh(combined, smooth=False)
    render_flags = RenderFlags.SHADOWS_DIRECTIONAL
    reported_backends = set()

    for idx, room in enumerate(rooms):
        points = np.array(room["points"], dtype=float) * scale
        polygon = geom.Polygon(points)
        if polygon.area <= 1e-6:
            continue

        room_label = room.get("class", f"room_{idx}")
        slug = room_label.replace(" ", "_").lower()

        poses = generate_camera_poses(polygon, wall_height, n_views=3)
        centroid = polygon.centroid

        for view_idx, (eye, target) in enumerate(poses, start=1):
            camera_pose = build_camera_pose(eye, target)

            scene = pyrender.Scene(
                bg_color=[0.96, 0.96, 0.97, 1.0],
                ambient_light=[0.14, 0.14, 0.14, 1.0],
            )
            scene.add(render_mesh)

            camera = pyrender.PerspectiveCamera(
                yfov=np.deg2rad(60.0),
                aspectRatio=float(width) / float(height),
            )
            scene.add(camera, pose=camera_pose)

            # soft fill from ceiling centre
            ceiling_light_pose = np.eye(4)
            ceiling_light_pose[:3, 3] = [centroid.x, centroid.y, wall_height - 0.1]
            scene.add(
                pyrender.PointLight(color=np.array([1.0, 0.97, 0.9]), intensity=650.0),
                pose=ceiling_light_pose,
            )

            # spot light to highlight centre
            spot_pose = build_camera_pose(
                np.array([centroid.x, centroid.y, wall_height - 0.05]),
                np.array([centroid.x, centroid.y, wall_height * 0.2]),
            )
            scene.add(
                pyrender.SpotLight(
                    color=np.ones(3) * 0.95,
                    intensity=1300.0,
                    innerConeAngle=np.deg2rad(18.0),
                    outerConeAngle=np.deg2rad(32.0),
                ),
                pose=spot_pose,
            )

            # key directional light aligned with camera for readable shadows
            key_origin = eye + np.array([0.0, 0.0, 1.0])
            key_target = np.array([centroid.x, centroid.y, wall_height * 0.3])
            key_pose = build_camera_pose(key_origin, key_target)
            scene.add(
                pyrender.DirectionalLight(color=np.ones(3) * 0.95, intensity=2.4),
                pose=key_pose,
            )

            # subtle rim light to avoid dark corners
            rim_origin = np.array([centroid.x - 2.5, centroid.y - 1.5, wall_height + 1.0])
            rim_target = np.array([centroid.x, centroid.y, wall_height * 0.5])
            rim_pose = build_camera_pose(rim_origin, rim_target)
            scene.add(
                pyrender.DirectionalLight(color=np.array([0.8, 0.85, 1.0]), intensity=1.2),
                pose=rim_pose,
            )

            try:
                backend, renderer = make_renderer(width, height)
                if backend not in reported_backends:
                    print(f"[render_rooms] Using {backend} backend for offscreen rendering.")
                    reported_backends.add(backend)
            except RuntimeError as exc:
                print(f"[render_rooms] Unable to render rooms: {exc}")
                return
            try:
                try:
                    color, _ = renderer.render(scene, flags=render_flags)
                except Exception:
                    color, _ = renderer.render(scene)
            finally:
                renderer.delete()
            image = np.clip(color, 0, 255).astype(np.uint8)

            image_path = output_dir / f"{idx:02d}_{slug}_v{view_idx}.png"
            imageio.imwrite(image_path, image)
            print(f"Saved render for {room_label} [view {view_idx}] -> {image_path}")



def main():
    parser = argparse.ArgumentParser(description="Render per-room interior images from a GLB mesh.")
    parser.add_argument("--mesh", type=Path, required=True, help="Path to the GLB/mesh file.")
    parser.add_argument("--polygons", type=Path, required=True, help="Path to polygons.json.")
    parser.add_argument("--output", type=Path, required=True, help="Directory to save rendered images.")
    parser.add_argument("--scale", type=float, required=True, help="Pixel-to-unit scale used for the geometry.")
    parser.add_argument("--wall-height", type=float, required=True, help="Wall height used for extrusion.")
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
