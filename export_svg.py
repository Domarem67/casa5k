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
    start = "M {} {}".format(points[0][0], points[0][1])
    segments = " ".join(f"L {x} {y}" for x, y in points[1:])
    if close:
        return f"{start} {segments} Z"
    return f"{start} {segments}"


def compute_canvas(polygons):
    max_x = 0.0
    max_y = 0.0
    for pts in polygons:
        for x, y in pts:
            max_x = max(max_x, x)
            max_y = max(max_y, y)
    return int(max_x) + 1, int(max_y) + 1


def load_polygons(path: Path):
    with path.open("r", encoding="utf-8") as fp:
        data = json.load(fp)

    icons = data.get("icons", [])
    walls = data.get("walls", [])
    rooms = data.get("rooms", [])

    normalize = lambda coords: [(float(x), float(y)) for x, y in coords]
    icon_polys = [
        (item["class"], normalize(item["points"]))
        for item in icons
    ]
    wall_polys = [
        (str(item["class"]), normalize(item["points"]))
        for item in walls
    ]
    room_polys = [
        (item["class"], normalize(item["points"]))
        for item in rooms
    ]

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
        color = ROOM_COLORS.get(label, "#CCCCCC")
        rooms_group.add(
            dwg.path(
                d=polygon_path(scaled_points(pts)),
                fill=color,
                fill_opacity=0.5,
                stroke=color,
                stroke_width=1 * scale,
            )
        )

    walls_group = dwg.add(dwg.g(id="walls"))
    for _, pts in wall_polys:
        walls_group.add(
            dwg.path(
                d=polygon_path(scaled_points(pts)),
                fill=WALL_COLOR,
                stroke=WALL_COLOR,
                stroke_width=1 * scale,
            )
        )

    icons_group = dwg.add(dwg.g(id="icons"))
    for label, pts in icon_polys:
        color = ICON_COLORS.get(label, "#FFFFFF")
        icons_group.add(
            dwg.path(
                d=polygon_path(scaled_points(pts)),
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
