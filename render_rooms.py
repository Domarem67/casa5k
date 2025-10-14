import argparse
import json
from pathlib import Path

import imageio
import numpy as np
import pyrender
import shapely.geometry as geom
import trimesh


def resolve_camera_pose(centroid, polygon, wall_height):
    centroid_xy = np.array([centroid.x, centroid.y])
    coords = np.array(polygon.exterior.coords[:-1])
    if coords.shape[0] < 3:
        direction = np.array([1.0, 0.0])
    else:
        centered = coords - centroid_xy
        cov = centered.T @ centered
        if np.linalg.norm(cov) < 1e-6:
            direction = np.array([1.0, 0.0])
        else:
            eigvals, eigvecs = np.linalg.eigh(cov)
            direction = eigvecs[:, np.argmax(eigvals)]
    direction = direction / np.linalg.norm(direction)

    area = polygon.area
    nominal_radius = max(np.sqrt(area / np.pi), 1.0)
    offset_distance = min(max(nominal_radius * 1.2, 1.5), 3.5)

    buffer_poly = polygon.buffer(-0.1)
    candidates = [
        centroid_xy + direction * offset_distance,
        centroid_xy - direction * offset_distance,
        centroid_xy + direction * offset_distance * 0.5,
    ]
    eye_xy = centroid_xy
    for candidate in candidates:
        point = geom.Point(candidate[0], candidate[1])
        if buffer_poly.is_empty or buffer_poly.contains(point):
            eye_xy = candidate
            break

    eye_height = min(max(1.5, wall_height * 0.45), wall_height - 0.3)
    eye = np.array([eye_xy[0], eye_xy[1], eye_height])
    target = np.array([centroid_xy[0], centroid_xy[1], min(wall_height * 0.5, eye_height)])
    return eye, target


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

    renderer = pyrender.OffscreenRenderer(width, height)

    for idx, room in enumerate(rooms):
        points = np.array(room["points"], dtype=float) * scale
        polygon = geom.Polygon(points)
        if polygon.area <= 1e-6:
            continue
        centroid = polygon.centroid

        eye, target = resolve_camera_pose(centroid, polygon, wall_height)
        camera_pose = build_camera_pose(eye, target)

        scene = pyrender.Scene(bg_color=[1, 1, 1, 1], ambient_light=[0.4, 0.4, 0.4, 1.0])
        scene.add(render_mesh)

        camera = pyrender.PerspectiveCamera(
            yfov=np.deg2rad(60.0), aspectRatio=float(width) / float(height)
        )
        scene.add(camera, pose=camera_pose)

        ceiling_light_pose = np.eye(4)
        ceiling_light_pose[:3, 3] = [centroid.x, centroid.y, wall_height - 0.2]
        scene.add(pyrender.PointLight(color=np.ones(3), intensity=900.0), pose=ceiling_light_pose)

        sun_pose = np.eye(4)
        sun_pose[:3, 3] = [centroid.x + 1.0, centroid.y + 1.0, wall_height + 1.5]
        scene.add(pyrender.DirectionalLight(color=np.ones(3), intensity=3.0), pose=sun_pose)

        color, _ = renderer.render(scene)
        image = np.clip(color, 0, 255).astype(np.uint8)

        room_label = room.get("class", f"room_{idx}")
        slug = room_label.replace(" ", "_").lower()
        image_path = output_dir / f"{idx:02d}_{slug}.png"
        imageio.imwrite(image_path, image)
        print(f"Saved render for {room_label} -> {image_path}")

    renderer.delete()


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
