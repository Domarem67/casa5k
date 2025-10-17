import argparse
import json
import math
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
import shapely.geometry as geom

from render_rooms import (
    build_room_polygon,
    resolve_camera_pose,
    _ray_polygon_distance,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Generate an SVG overlay that visualises per-room camera placements "
            "and their horizontal field of view."
        )
    )
    parser.add_argument("--polygons", type=Path, required=True, help="Path to polygons.json file.")
    parser.add_argument("--output", type=Path, required=True, help="Path to the SVG file to create.")
    parser.add_argument("--scale", type=float, default=1.0, help="Coordinate scale factor applied to polygons (same as in render step).")
    parser.add_argument("--wall-height", type=float, default=3.0, help="Wall height used when resolving camera poses.")
    parser.add_argument("--image-width", type=int, default=1280, help="Render width used for perspective camera (for FOV).")
    parser.add_argument("--image-height", type=int, default=720, help="Render height used for perspective camera (for FOV).")
    parser.add_argument("--padding", type=float, default=50.0, help="Extra padding around drawing bounds in SVG units.")
    parser.add_argument("--min-fov-depth", type=float, default=1.0, help="Minimum distance (in SVG units) to extend FOV cones.")
    parser.add_argument("--vertical-fov-deg", type=float, default=70.0, help="Vertical field of view (degrees) used by the renderer.")
    return parser.parse_args()


def polygon_outline_path(polygon: geom.Polygon) -> str:
    exterior = coords_to_path(polygon.exterior.coords)
    holes = [coords_to_path(interior.coords) for interior in polygon.interiors]
    return " ".join([exterior, *holes]).strip()


def coords_to_path(coords: Iterable[Tuple[float, float]]) -> str:
    coords = list(coords)
    if not coords:
        return ""
    commands = [f"M {coords[0][0]:.3f},{coords[0][1]:.3f}"]
    for point in coords[1:]:
        commands.append(f"L {point[0]:.3f},{point[1]:.3f}")
    commands.append("Z")
    return " ".join(commands)


def rotate(vec: np.ndarray, angle: float) -> np.ndarray:
    c = math.cos(angle)
    s = math.sin(angle)
    return np.array([vec[0] * c - vec[1] * s, vec[0] * s + vec[1] * c], dtype=float)


def compute_bounds(polygons: Iterable[geom.Polygon]) -> Tuple[float, float, float, float]:
    minx = miny = float("inf")
    maxx = maxy = float("-inf")
    for poly in polygons:
        bx, by, Bx, By = poly.bounds
        minx = min(minx, bx)
        miny = min(miny, by)
        maxx = max(maxx, Bx)
        maxy = max(maxy, By)
    if not math.isfinite(minx):
        raise ValueError("No valid polygons to compute bounds.")
    return minx, miny, maxx, maxy


def create_svg_root(minx: float, miny: float, maxx: float, maxy: float, padding: float):
    width = maxx - minx + 2.0 * padding
    height = maxy - miny + 2.0 * padding
    attrs = {
        "xmlns": "http://www.w3.org/2000/svg",
        "version": "1.1",
        "width": f"{width:.1f}",
        "height": f"{height:.1f}",
        "viewBox": f"{minx - padding:.3f} {miny - padding:.3f} {width:.3f} {height:.3f}",
    }
    return Element("svg", attrs)


def Element(tag: str, attributes: dict, children: List["Element"] = None, text: str = None):
    """Helper to produce an SVG element represented as nested tuples for easy serialisation."""
    return {"tag": tag, "attributes": attributes, "children": children or [], "text": text}


def serialize(element: dict, indent: int = 0) -> str:
    pad = "  " * indent
    attrs = " ".join(f'{key}="{value}"' for key, value in element["attributes"].items())
    if element["children"]:
        opening = f"{pad}<{element['tag']} {attrs}>\n"
        children_str = "".join(serialize(child, indent + 1) for child in element["children"])
        closing = f"{pad}</{element['tag']}>\n"
        return opening + (element["text"] or "") + children_str + closing
    if element["text"]:
        return f"{pad}<{element['tag']} {attrs}>{element['text']}</{element['tag']}>\n"
    return f"{pad}<{element['tag']} {attrs} />\n"


def main():
    args = parse_args()

    with args.polygons.open("r", encoding="utf-8") as fp:
        data = json.load(fp)

    rooms = data.get("rooms", [])
    if not rooms:
        raise ValueError("No rooms detected in polygons.json.")

    vfov = math.radians(max(args.vertical_fov_deg, 1e-3))
    aspect_ratio = float(args.image_width) / float(args.image_height)
    horizontal_fov = 2.0 * math.atan(math.tan(vfov / 2.0) * aspect_ratio)

    room_polygons = []
    visuals = []

    for idx, room in enumerate(rooms):
        polygon = build_room_polygon(room.get("points"), args.scale)
        if polygon is None or polygon.area <= 1e-6:
            continue

        eye, target = resolve_camera_pose(polygon, args.wall_height, vfov, aspect_ratio)
        eye_xy = np.array([eye[0], eye[1]], dtype=float)
        target_xy = np.array([target[0], target[1]], dtype=float)
        forward = target_xy - eye_xy
        forward_norm = np.linalg.norm(forward)
        if forward_norm < 1e-6:
            continue
        forward /= forward_norm

        diag = np.linalg.norm(np.array(polygon.bounds[2:]) - np.array(polygon.bounds[:2]))
        if not math.isfinite(diag) or diag <= 0.0:
            diag = max(polygon.bounds[2] - polygon.bounds[0], polygon.bounds[3] - polygon.bounds[1], 1.0)
        max_distance = max(diag * 1.8, args.min_fov_depth)

        left_dir = rotate(forward, horizontal_fov / 2.0)
        right_dir = rotate(forward, -horizontal_fov / 2.0)

        forward_dist = _ray_polygon_distance(polygon, eye_xy, forward, max_distance) or max_distance
        left_dist = _ray_polygon_distance(polygon, eye_xy, left_dir, max_distance) or max_distance
        right_dist = _ray_polygon_distance(polygon, eye_xy, right_dir, max_distance) or max_distance

        fov_points = [
            eye_xy,
            eye_xy + left_dir * left_dist,
            eye_xy + forward * forward_dist,
            eye_xy + right_dir * right_dist,
        ]

        room_polygons.append(polygon)
        visuals.append(
            {
                "index": idx,
                "label": room.get("class", f"room_{idx}"),
                "polygon": polygon,
                "eye": eye_xy,
                "target": target_xy,
                "fov_points": fov_points,
            }
        )

    if not visuals:
        raise ValueError("Unable to build any room polygons; nothing to draw.")

    minx, miny, maxx, maxy = compute_bounds(room_polygons)
    svg_root = create_svg_root(minx, miny, maxx, maxy, args.padding)

    palette = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
        "#17becf",
    ]

    background = Element(
        "rect",
        {
            "x": f"{minx - args.padding:.3f}",
            "y": f"{miny - args.padding:.3f}",
            "width": f"{maxx - minx + 2 * args.padding:.3f}",
            "height": f"{maxy - miny + 2 * args.padding:.3f}",
            "fill": "#f6f7fb",
        },
    )
    svg_root["children"].append(background)

    outlines_group = Element("g", {"fill": "none", "stroke-width": "2"})
    svg_root["children"].append(outlines_group)

    cameras_group = Element("g", {"stroke-width": "1.5"})
    svg_root["children"].append(cameras_group)

    label_group = Element("g", {"font-family": "Arial, sans-serif", "text-anchor": "start"})
    svg_root["children"].append(label_group)

    bounds_extent = max(maxx - minx, maxy - miny)
    circle_radius = max(bounds_extent * 0.01, 4.0)
    text_offset = circle_radius * 1.6

    for idx, info in enumerate(visuals):
        color = palette[idx % len(palette)]
        polygon_path = polygon_outline_path(info["polygon"])
        outlines_group["children"].append(
            Element(
                "path",
                {
                    "d": polygon_path,
                    "stroke": color,
                    "fill": color,
                    "fill-opacity": "0.05",
                },
            )
        )

        fov_path = coords_to_path([tuple(pt) for pt in info["fov_points"]])
        cameras_group["children"].append(
            Element(
                "path",
                {
                    "d": fov_path,
                    "fill": color,
                    "fill-opacity": "0.12",
                    "stroke": color,
                    "stroke-dasharray": "4 3",
                },
            )
        )

        cameras_group["children"].append(
            Element(
                "line",
                {
                    "x1": f"{info['eye'][0]:.3f}",
                    "y1": f"{info['eye'][1]:.3f}",
                    "x2": f"{info['target'][0]:.3f}",
                    "y2": f"{info['target'][1]:.3f}",
                    "stroke": color,
                },
            )
        )

        cameras_group["children"].append(
            Element(
                "circle",
                {
                    "cx": f"{info['eye'][0]:.3f}",
                    "cy": f"{info['eye'][1]:.3f}",
                    "r": f"{circle_radius:.3f}",
                    "fill": color,
                    "stroke": "#1b1d23",
                },
            )
        )

        label_group["children"].append(
            Element(
                "text",
                {
                    "x": f"{info['eye'][0] + text_offset:.3f}",
                    "y": f"{info['eye'][1] - text_offset:.3f}",
                    "font-size": f"{circle_radius * 1.4:.2f}",
                    "fill": "#111",
                },
                text=f"{info['index']:02d} · {info['label']}",
            )
        )

    svg_string = '<?xml version="1.0" encoding="UTF-8"?>\n' + serialize(svg_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(svg_string, encoding="utf-8")
    print(f"[render_camera_fov] Wrote {args.output}")


if __name__ == "__main__":
    main()
