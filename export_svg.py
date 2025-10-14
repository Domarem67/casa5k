import argparse
import json
from pathlib import Path

import svgwrite


ROOM_COLORS = {
    "Background": "#000000",
    "Outdoor": "#2E8B57",
    "Wall": "#A9A9A9",
    "Kitchen": "#FFA500",
    "Living Room": "#1E90FF",
    "Bedroom": "#DA70D6",
    "Bath": "#4169E1",
    "Hallway": "#B8860B",
    "Railing": "#FFE4B5",
    "Storage": "#8B4513",
    "Garage": "#696969",
    "Other rooms": "#98FB98",
}

ICON_COLORS = {
    "Window": "#87CEEB",
    "Door": "#FFD700",
    "Closet": "#CD853F",
    "Electr. Appl.": "#FF4500",
    "Toilet": "#FF69B4",
    "Sink": "#00BFFF",
    "Sauna bench": "#FFA07A",
    "Fire Place": "#DC143C",
    "Bathtub": "#C71585",
    "Chimney": "#800000",
    "Empty": "#000000",
}

WALL_COLOR = "#444444"


def polygon_path(points, close=True):
    if not points:
        return ""
    start = "M {} {}".format(points[0][0], points[0][1])
    segments = " ".join(f"L {x} {y}" for x, y in points[1:])
    if close:
        return f"{start} {segments} Z"
    return f"{start} {segments}"


def compute_canvas(polygons):
    max_x = 0.0
    max_y = 0.0
    for pts in polygons:
        if not pts:
            continue
        for x, y in pts:
            max_x = max(max_x, x)
            max_y = max(max_y, y)
    width = int(max_x) + 1 if max_x > 0 else 1
    height = int(max_y) + 1 if max_y > 0 else 1
    return width, height


def load_polygons(path: Path):
    with path.open("r", encoding="utf-8") as fp:
        data = json.load(fp)

    icons = data.get("icons", [])
    walls = data.get("walls", [])
    rooms = data.get("rooms", [])

    def normalize_shapes(label, coords, cast_label=str):
        def to_polygons(raw):
            if not raw:
                return []
            # handle nested structures (e.g., multipolygons)
            first = raw[0]
            if isinstance(first, (list, tuple)) and first and isinstance(first[0], (list, tuple)):
                polygons = raw
            else:
                polygons = [raw]
            cleaned = []
            for poly in polygons:
                if not poly:
                    continue
                try:
                    pts = [(float(x), float(y)) for x, y in poly]
                except (TypeError, ValueError):
                    continue
                # SVG needs at least 3 points for a filled polygon
                if len(pts) >= 3:
                    cleaned.append(pts)
            return cleaned

        polys = []
        for poly in to_polygons(coords):
            polys.append((cast_label(label), poly))
        return polys

    icon_polys = []
    for item in icons:
        icon_polys.extend(normalize_shapes(item["class"], item.get("points", []), cast_label=str))

    wall_polys = []
    for item in walls:
        wall_polys.extend(normalize_shapes(item["class"], item.get("points", []), cast_label=str))

    room_polys = []
    for item in rooms:
        room_polys.extend(normalize_shapes(item["class"], item.get("points", []), cast_label=str))

    return icon_polys, wall_polys, room_polys


def export_svg(polygons_path: Path, output_path: Path, scale: float):
    icon_polys, wall_polys, room_polys = load_polygons(polygons_path)

    all_points = []
    for _, pts in room_polys + wall_polys + icon_polys:
        all_points.append(pts)
    width, height = compute_canvas(all_points)
    width *= scale
    height *= scale

    dwg = svgwrite.Drawing(filename=str(output_path), size=(f"{width}px", f"{height}px"))
    dwg.viewbox(0, 0, width, height)

    def scaled_points(pts):
        return [(x * scale, y * scale) for x, y in pts]

    rooms_group = dwg.add(dwg.g(id="rooms"))
    for label, pts in room_polys:
        if not pts:
            continue
        color = ROOM_COLORS.get(label, "#CCCCCC")
        path_data = polygon_path(scaled_points(pts))
        if not path_data:
            continue
        rooms_group.add(
            dwg.path(
                d=path_data,
                fill=color,
                fill_opacity=0.5,
                stroke=color,
                stroke_width=1 * scale,
            )
        )

    walls_group = dwg.add(dwg.g(id="walls"))
    for _, pts in wall_polys:
        if not pts:
            continue
        path_data = polygon_path(scaled_points(pts))
        if not path_data:
            continue
        walls_group.add(
            dwg.path(
                d=path_data,
                fill=WALL_COLOR,
                stroke=WALL_COLOR,
                stroke_width=1 * scale,
            )
        )

    icons_group = dwg.add(dwg.g(id="icons"))
    for label, pts in icon_polys:
        if not pts:
            continue
        path_data = polygon_path(scaled_points(pts))
        if not path_data:
            continue
        color = ICON_COLORS.get(label, "#FFFFFF")
        icons_group.add(
            dwg.path(
                d=path_data,
                fill=color,
                stroke=color,
                stroke_width=1 * scale,
            )
        )

    dwg.save()


def main():
    parser = argparse.ArgumentParser(description="Convert polygons.json to SVG.")
    parser.add_argument("--input", type=Path, required=True, help="Path to polygons.json file.")
    parser.add_argument("--output", type=Path, required=True, help="Path to the output SVG.")
    parser.add_argument(
        "--scale", type=float, default=1.0, help="Scale factor multiplied to the coordinates."
    )
    args = parser.parse_args()

    export_svg(args.input, args.output, args.scale)
    print(f"SVG saved to {args.output}")


if __name__ == "__main__":
    main()
