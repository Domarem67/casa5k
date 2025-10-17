import argparse
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import shapely.affinity as affinity
import shapely.geometry as geom
import shapely.ops as ops
import trimesh
from trimesh import creation as tm_creation


ICON_COLOR_TO_LABEL = {
    "#87ceeb": "Window",
    "#ffd700": "Door",
    "#cd853f": "Closet",
    "#ff4500": "Electr. Appl.",
    "#ff69b4": "Toilet",
    "#00bfff": "Sink",
    "#ffa07a": "Sauna bench",
    "#dc143c": "Fire Place",
    "#c71585": "Bathtub",
    "#800000": "Chimney",
    "#000000": "Empty",
}

WALL_COLOR = "#444444"


def parse_path_points(path_d: str):
    coords = [float(c) for c in re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", path_d)]
    if len(coords) % 2 != 0:
        raise ValueError(f"Invalid coordinate count in path: {path_d}")
    points = [(coords[i], coords[i + 1]) for i in range(0, len(coords), 2)]
    if points and points[0] != points[-1]:
        points.append(points[0])
    return points


def polygon_from_points(points):
    poly = geom.Polygon(points)
    if not poly.is_valid:
        poly = poly.buffer(0)
    return poly


def extract_polygons_from_svg(svg_path: Path):
    tree = ET.parse(str(svg_path))
    root = tree.getroot()

    ns = {}
    if root.tag.startswith("{"):
        uri = root.tag[root.tag.find("{") + 1 : root.tag.find("}")]
        ns["svg"] = uri

    def group_elements(group_id):
        search = f".//{{{ns['svg']}}}g" if ns else ".//g"
        for g in root.findall(search):
            if g.attrib.get("id") == group_id:
                yield from list(g)

    wall_polygons = []
    door_polygons = []
    window_polygons = []
    room_polygons = []

    for element in group_elements("walls"):
        if element.tag.split("}")[-1] != "path":
            continue
        d_attr = element.attrib.get("d")
        fill = element.attrib.get("fill", "").lower()
        if not d_attr:
            continue
        if fill and fill != WALL_COLOR:
            continue
        poly = polygon_from_points(parse_path_points(d_attr))
        if poly.area > 0:
            wall_polygons.append(poly)

    for element in group_elements("icons"):
        if element.tag.split("}")[-1] != "path":
            continue
        d_attr = element.attrib.get("d")
        fill = element.attrib.get("fill", "").lower()
        if not d_attr or not fill:
            continue
        label = ICON_COLOR_TO_LABEL.get(fill)
        if label not in {"Door", "Window"}:
            continue
        poly = polygon_from_points(parse_path_points(d_attr))
        if poly.area <= 0:
            continue
        if label == "Door":
            door_polygons.append(poly)
        elif label == "Window":
            window_polygons.append(poly)

    for element in group_elements("rooms"):
        if element.tag.split("}")[-1] != "path":
            continue
        d_attr = element.attrib.get("d")
        if not d_attr:
            continue
        poly = polygon_from_points(parse_path_points(d_attr))
        if poly.area > 0:
            room_polygons.append(poly)

    return wall_polygons, door_polygons, window_polygons, room_polygons


def group_openings_by_wall(walls, openings):
    grouped = []
    for wall in walls:
        intersecting = [opening for opening in openings if wall.intersects(opening)]
        if intersecting:
            grouped.append(ops.unary_union(intersecting))
        else:
            grouped.append(None)
    return grouped


def extrude_polygon(poly, height, scale, z_offset=0.0, cap=True):
    if poly.is_empty:
        return None

    if isinstance(poly, geom.MultiPolygon):
        meshes = [
            extrude_polygon(part, height, scale, z_offset, cap=cap)
            for part in poly.geoms
        ]
        meshes = [m for m in meshes if m is not None]
        if not meshes:
            return None
        return trimesh.util.concatenate(meshes)

    scaled = affinity.scale(poly, xfact=scale, yfact=scale, origin=(0, 0))
    if scaled.area == 0:
        return None
    mesh = trimesh.creation.extrude_polygon(scaled, height, cap=cap)
    if z_offset:
        mesh.apply_translation((0.0, 0.0, z_offset))
    return mesh


def extrude_walls(
    wall_polys,
    door_polys,
    window_polys,
    height,
    scale,
    padding,
    window_sill,
    window_head,
    door_height,
):
    padding = max(0.0, padding)
    door_groups = group_openings_by_wall(wall_polys, door_polys)
    window_groups = group_openings_by_wall(wall_polys, window_polys)

    meshes = []
    for idx, wall in enumerate(wall_polys):
        door_union = door_groups[idx]
        window_union = window_groups[idx]

        base_poly = wall
        if base_poly.is_empty:
            continue

        door_cut = None
        if door_union:
            door_cut = door_union.buffer(padding, join_style=2) if padding else door_union
        window_cut = None
        if window_union:
            window_cut = window_union.buffer(padding, join_style=2) if padding else window_union

        levels = {0.0, height}
        if door_height > 0:
            levels.add(max(0.0, min(door_height, height)))
        if window_sill > 0:
            levels.add(max(0.0, min(window_sill, height)))
        if window_head > 0:
            levels.add(max(0.0, min(window_head, height)))
        levels = sorted(levels)

        for idx_lvl in range(len(levels) - 1):
            start = levels[idx_lvl]
            end = levels[idx_lvl + 1]
            if end <= start:
                continue

            seg_poly = base_poly

            if door_cut and start < door_height:
                seg_poly = seg_poly.difference(door_cut)

            if window_cut and not (end <= window_sill or start >= window_head):
                seg_poly = seg_poly.difference(window_cut)

            if seg_poly.is_empty:
                continue

            seg_mesh = extrude_polygon(seg_poly, end - start, scale, z_offset=start)
            if seg_mesh:
                meshes.append(seg_mesh)

    if not meshes:
        raise RuntimeError("No wall geometry created from SVG after processing.")
    return trimesh.util.concatenate(meshes)


def extrude_floor(rooms, scale, thickness, cap=True):
    if thickness <= 0 or not rooms:
        return None
    union = ops.unary_union(rooms)
    if union.is_empty:
        return None
    polygons = []
    if isinstance(union, geom.Polygon):
        polygons = [union]
    else:
        polygons = [poly for poly in union.geoms if isinstance(poly, geom.Polygon)]
    meshes = []
    for poly in polygons:
        mesh = extrude_polygon(poly, thickness, scale, cap=cap)
        if mesh:
            meshes.append(mesh)
    if not meshes:
        return None
    return trimesh.util.concatenate(meshes)


def extrude_ceiling(rooms, scale, thickness, wall_height):
    if thickness <= 0:
        return None
    mesh = extrude_floor(rooms, scale, thickness, cap=True)
    if mesh is None:
        return None
    mesh.apply_translation((0.0, 0.0, wall_height - thickness))
    return mesh


def export_mesh(geometry, output: Path):
    output = Path(output)
    geometry.export(str(output))
    print(f"Mesh exported to {output.resolve()}")


def main():
    parser = argparse.ArgumentParser(
        description="Build wall + floor mesh from SVG (with door/window handling)."
    )
    parser.add_argument("--svg", type=Path, required=True, help="Path to plan SVG.")
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Mesh output (.glb/.obj/.stl...).",
    )
    parser.add_argument(
        "--height", type=float, default=3.0, help="Wall height."
    )
    parser.add_argument(
        "--scale",
        default="auto",
        help="Scale factor applied to SVG coordinates (float) or 'auto' to infer from doors.",
    )
    parser.add_argument(
        "--door-width-ref",
        type=float,
        default=0.9,
        help="Physical door width reference (m) used when --scale=auto.",
    )
    parser.add_argument(
        "--fallback-scale",
        type=float,
        default=0.01,
        help="Scale used if auto inference fails.",
    )
    parser.add_argument(
        "--opening-padding",
        type=float,
        default=0.0,
        help="Buffer around door/window polygons before subtraction.",
    )
    parser.add_argument(
        "--window-sill",
        type=float,
        default=1.0,
        help="Window sill height.",
    )
    parser.add_argument(
        "--window-head",
        type=float,
        default=2.1,
        help="Window head height.",
    )
    parser.add_argument(
        "--door-height",
        type=float,
        default=2.1,
        help="Door opening height.",
    )
    parser.add_argument(
        "--door-frame-depth",
        type=float,
        default=0.0,
        help="Thickness of the door frame (meters). Frames are disabled by default.",
    )
    parser.add_argument(
        "--floor-thickness",
        type=float,
        default=0.2,
        help="Floor slab thickness (0 disables floor).",
    )
    parser.add_argument(
        "--ceiling-thickness",
        type=float,
        default=0.1,
        help="Ceiling slab thickness (0 disables ceiling).",
    )
    parser.add_argument(
        "--invert-z",
        action="store_true",
        help="Invert vertical axis so ceiling becomes floor.",
    )
    parser.add_argument(
        "--smooth-walls",
        action="store_true",
        help="Post-process walls to merge segments and drop door frames.",
    )
    args = parser.parse_args()

    wall_polys, door_polys, window_polys, room_polys = extract_polygons_from_svg(args.svg)
    if not wall_polys:
        raise RuntimeError("No walls found in SVG (expected <g id=\"walls\">).")
    print(
        f"Loaded {len(wall_polys)} wall polygons, {len(door_polys)} doors, {len(window_polys)} windows, {len(room_polys)} rooms."
    )

    if isinstance(args.scale, str) and args.scale.lower() == "auto":
        door_lengths = []
        for door in door_polys:
            if door.is_empty:
                continue
            if isinstance(door, geom.MultiPolygon):
                parts = door.geoms
            else:
                parts = [door]
            for part in parts:
                minx, miny, maxx, maxy = part.bounds
                width = maxx - minx
                height = maxy - miny
                length = max(width, height)
                if length > 0:
                    door_lengths.append(length)
        if door_lengths:
            avg_length = sum(door_lengths) / len(door_lengths)
            scale = args.door_width_ref / avg_length
            print(
                f"Inferred scale from doors: {scale:.4f} (avg door span {avg_length:.2f} px for {args.door_width_ref} m)."
            )
        else:
            scale = args.fallback_scale
            print(
                f"Warning: no doors found to infer scale. Using fallback {scale}."
            )
    else:
        scale = float(args.scale)
        print(f"Using provided scale: {scale}")

    wall_mesh = extrude_walls(
        wall_polys,
        door_polys,
        window_polys,
        args.height,
        scale,
        args.opening_padding,
        args.window_sill,
        args.window_head,
        args.door_height,
    )

    floor_mesh = extrude_floor(room_polys, scale, args.floor_thickness)
    ceiling_mesh = extrude_ceiling(room_polys, scale, args.ceiling_thickness, args.height)

    def create_door_frames(doors, scale_factor, frame_height, frame_depth, wall_height):
        if frame_depth <= 0 or frame_height <= 0 or not doors:
            return None
        meshes = []
        buffer_px = frame_depth / scale_factor
        for door in doors:
            if door.is_empty:
                continue
            if isinstance(door, geom.MultiPolygon):
                parts = door.geoms
            else:
                parts = [door]
            for part in parts:
                outer = part.buffer(buffer_px, join_style=2)
                frame_shape = outer.difference(part)
                if frame_shape.is_empty:
                    continue
                mesh = extrude_polygon(
                    frame_shape,
                    min(frame_height, wall_height),
                    scale_factor,
                    cap=False,
                )
                if mesh:
                    meshes.append(mesh)
        if not meshes:
            return None
        return trimesh.util.concatenate(meshes)

    frame_mesh = None
    frame_depth = 0.0 if args.smooth_walls else args.door_frame_depth
    if frame_depth > 0:
        frame_mesh = create_door_frames(
            door_polys,
            scale,
            args.door_height,
            frame_depth,
            args.height,
        )

    def _rectangle_axes(polygon):
        try:
            oriented = polygon.minimum_rotated_rectangle
        except Exception:
            oriented = polygon
        coords = np.asarray(oriented.exterior.coords[:-1])
        if coords.shape[0] < 2:
            return None
        vecs = np.diff(np.vstack([coords, coords[0]]), axis=0)
        lengths = np.linalg.norm(vecs, axis=1)
        if lengths.size < 2:
            return None
        order = np.argsort(lengths)
        width_vec = vecs[order[-1]]
        width_len = lengths[order[-1]]
        other_vec = vecs[order[0]]
        other_len = lengths[order[0]]
        if width_len < other_len:
            width_vec, other_vec = other_vec, width_vec
            width_len, other_len = other_len, width_len
        if width_len < 1e-6 or other_len < 1e-6:
            return None
        width_dir = width_vec / width_len
        depth_dir = np.array([-width_dir[1], width_dir[0]])
        center = coords.mean(axis=0)
        return width_dir, depth_dir, width_len, other_len, center

    def create_door_models(doors, scale_factor, panel_thickness=0.04, open_angle_deg=28.0):
        models = []
        angle_rad = np.deg2rad(open_angle_deg)
        for idx, door in enumerate(doors):
            if door.is_empty:
                continue
            axes = _rectangle_axes(door)
            if axes is None:
                continue
            width_dir, depth_dir, width_len, depth_len, center_px = axes

            walkway_dir = width_dir
            walkway_span = width_len
            wall_normal_dir = depth_dir
            wall_thickness_world = max(depth_len * scale_factor, panel_thickness * 1.5)

            center = np.array([center_px[0] * scale_factor, center_px[1] * scale_factor, 0.0])
            door_width = max(walkway_span * scale_factor * 0.97, 0.6)
            door_thickness = min(max(panel_thickness, wall_thickness_world * 0.6), wall_thickness_world * 0.95)
            door_height = min(args.door_height, args.height * 0.97)

            door_box = tm_creation.box(extents=[door_width, door_thickness, door_height])

            orientation = np.eye(4)
            orientation[:3, 0] = np.array([walkway_dir[0], walkway_dir[1], 0.0])
            orientation[:3, 1] = np.array([wall_normal_dir[0], wall_normal_dir[1], 0.0])
            orientation[:3, 2] = np.array([0.0, 0.0, 1.0])
            inset = -wall_thickness_world * 0.5 + door_thickness * 0.5
            orientation[:3, 3] = center + np.array([
                wall_normal_dir[0] * inset,
                wall_normal_dir[1] * inset,
                door_height * 0.5,
            ])
            door_box.apply_transform(orientation)

            hinge_point = center + np.array([
                wall_normal_dir[0] * inset,
                wall_normal_dir[1] * inset,
                door_height * 0.5,
            ]) + np.array([
                walkway_dir[0] * (-0.5 * door_width),
                walkway_dir[1] * (-0.5 * door_width),
                0.0,
            ])
            rotation = trimesh.transformations.rotation_matrix(angle_rad, [0.0, 0.0, 1.0], hinge_point)
            door_box.apply_transform(rotation)

            models.append((f"DoorModel_{idx:02d}", door_box))
        return models

    def create_window_models(windows, scale_factor, sill, head, frame_depth=0.08):
        height = max(head - sill, 0.0)
        if height <= 0.05:
            return []
        models = []
        for idx, window in enumerate(windows):
            if window.is_empty:
                continue
            axes = _rectangle_axes(window)
            if axes is None:
                continue
            width_dir, depth_dir, width_len, depth_len, center_px = axes

            center = np.array([center_px[0] * scale_factor, center_px[1] * scale_factor, 0.0])
            window_width = max(width_len * scale_factor * 0.98, 0.6)
            window_thickness = max(frame_depth, depth_len * scale_factor * 0.5)
            window_height = height

            frame_extents = [window_width, window_thickness, window_height]
            frame = tm_creation.box(extents=frame_extents)

            inset = max(window_width * 0.08, 0.04)
            inner_width = max(window_width - 2 * inset, 0.2)
            inner_height = max(window_height - 2 * inset, 0.2)
            glass_thickness = max(window_thickness * 0.35, 0.02)
            glass = tm_creation.box(extents=[inner_width, glass_thickness, inner_height])
            glass_offset_y = (window_thickness - glass_thickness) * 0.4
            glass.apply_translation([0.0, glass_offset_y, (inner_height - window_height) * 0.5])

            orientation = np.eye(4)
            orientation[:3, 0] = np.array([width_dir[0], width_dir[1], 0.0])
            orientation[:3, 1] = np.array([depth_dir[0], depth_dir[1], 0.0])
            orientation[:3, 2] = np.array([0.0, 0.0, 1.0])
            translate = center + np.array([0.0, 0.0, sill + window_height * 0.5])
            orientation[:3, 3] = translate
            frame.apply_transform(orientation)
            glass.apply_transform(orientation)

            models.append((f"WindowFrame_{idx:02d}", frame))
            models.append((f"WindowGlass_{idx:02d}", glass))
        return models

    geometries = []
    if wall_mesh is not None:
        if args.smooth_walls:
            wall_mesh.merge_vertices()
            wall_mesh.remove_duplicate_faces()
            wall_mesh.remove_degenerate_faces()
            wall_mesh.remove_unreferenced_vertices()
        geometries.append(("Walls", wall_mesh))
    if floor_mesh is not None:
        geometries.append(("Floor", floor_mesh))
    if ceiling_mesh is not None:
        geometries.append(("Ceiling", ceiling_mesh))
    if frame_mesh is not None:
        geometries.append(("Frames", frame_mesh))

    door_models = create_door_models(door_polys, scale)
    window_models = create_window_models(window_polys, scale, args.window_sill, args.window_head)
    geometries.extend(door_models)
    geometries.extend(window_models)

    if not geometries:
        raise RuntimeError("No geometry produced from SVG input.")

    rotation_x = trimesh.transformations.rotation_matrix(
        np.deg2rad(-90.0), [1.0, 0.0, 0.0]
    )
    rotation_y = trimesh.transformations.rotation_matrix(
        np.deg2rad(180.0), [0.0, 1.0, 0.0]
    )
    transform = trimesh.transformations.concatenate_matrices(rotation_y, rotation_x)

    scene = trimesh.Scene()

    for name, mesh_geom in geometries:
        mesh_copy = mesh_geom.copy()
        mesh_copy.apply_transform(transform)
        if args.invert_z:
            z_min = mesh_copy.vertices[:, 2].min()
            z_max = mesh_copy.vertices[:, 2].max()
            mesh_copy.vertices[:, 2] = z_max - (mesh_copy.vertices[:, 2] - z_min)
        scene.add_geometry(mesh_copy, node_name=name, geom_name=name)

    export_mesh(scene, args.output)


if __name__ == "__main__":
    main()
