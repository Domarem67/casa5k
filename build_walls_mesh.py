import argparse
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import shapely.affinity as affinity
import shapely.geometry as geom
import shapely.ops as ops
import trimesh


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
        default=0.05,
        help="Thickness of the door frame (meters). Set to 0 to skip.",
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

    meshes_to_merge = []
    if wall_mesh is not None:
        meshes_to_merge.append(wall_mesh)
    if floor_mesh is not None:
        meshes_to_merge.append(floor_mesh)
    if ceiling_mesh is not None:
        meshes_to_merge.append(ceiling_mesh)

    if not meshes_to_merge:
        raise RuntimeError("No geometry produced from SVG input.")

    combined = trimesh.util.concatenate(meshes_to_merge)

    if frame_mesh is not None:
        combined = trimesh.util.concatenate([combined, frame_mesh])

    if args.invert_z:
        z_min = combined.vertices[:, 2].min()
        z_max = combined.vertices[:, 2].max()
        combined.vertices[:, 2] = z_max - (combined.vertices[:, 2] - z_min)

    if args.smooth_walls:
        combined.merge_vertices()
        combined.remove_duplicate_faces()
        combined.remove_degenerate_faces()
        combined.remove_unreferenced_vertices()

    export_mesh(combined, args.output)


if __name__ == "__main__":
    main()
