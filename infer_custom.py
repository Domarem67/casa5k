import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from floortrans.models import get_model
from floortrans.loaders.augmentations import RotateNTurns
from floortrans.post_prosessing import split_prediction, get_polygons


ROOM_CLASSES = [
    "Background",
    "Outdoor",
    "Wall",
    "Kitchen",
    "Living Room",
    "Bedroom",
    "Bath",
    "Hallway",
    "Railing",
    "Storage",
    "Garage",
    "Other rooms",
]

ICON_CLASSES = [
    "Empty",
    "Window",
    "Door",
    "Closet",
    "Electr. Appl.",
    "Toilet",
    "Sink",
    "Sauna bench",
    "Fire Place",
    "Bathtub",
    "Chimney",
]

ROOM_PALETTE = [
    (0, 0, 0),
    (46, 139, 87),
    (169, 169, 169),
    (255, 165, 0),
    (30, 144, 255),
    (218, 112, 214),
    (65, 105, 225),
    (184, 134, 11),
    (255, 228, 181),
    (139, 69, 19),
    (105, 105, 105),
    (152, 251, 152),
]

ICON_PALETTE = [
    (0, 0, 0),
    (135, 206, 235),
    (255, 215, 0),
    (205, 133, 63),
    (255, 69, 0),
    (255, 105, 180),
    (0, 191, 255),
    (244, 164, 96),
    (220, 20, 60),
    (199, 21, 133),
    (128, 0, 0),
]


def build_palette(colors):
    flat = []
    for color in colors:
        flat.extend(color)
    # pad to 256 * 3 entries required by Pillow
    flat.extend([0] * (768 - len(flat)))
    return flat[:768]


def save_mask(mask, palette, path):
    img = Image.fromarray(mask.astype(np.uint8), mode="P")
    img.putpalette(palette)
    img.save(path)


def load_image(image_path, max_long_edge):
    img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    if max_long_edge and max_long_edge > 0:
        long_edge = float(max(img.shape[:2]))
        if long_edge > max_long_edge:
            scale = max_long_edge / long_edge
            new_w = int(round(img.shape[1] * scale))
            new_h = int(round(img.shape[0] * scale))
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    tensor = torch.from_numpy(img.transpose(2, 0, 1)).float()
    tensor = 2.0 * (tensor / 255.0) - 1.0
    return img, tensor.unsqueeze(0)


def run_model(image_tensor, model, device):
    rotator = RotateNTurns()
    rotations = [(0, 0), (1, -1), (2, 2), (-1, 1)]
    preds = []

    with torch.no_grad():
        for forward, back in rotations:
            rotated = rotator(image_tensor, "tensor", forward)
            pred = model(rotated.to(device))
            pred = rotator(pred, "tensor", back)
            pred = rotator(pred, "points", back)
            pred = F.interpolate(
                pred,
                size=image_tensor.shape[-2:],
                mode="bilinear",
                align_corners=True,
            )
            preds.append(pred.cpu())

    stacked = torch.stack(preds, dim=0)
    return torch.mean(stacked, dim=0)


def export_polygons(output_path, polygons, types, room_polygons, room_types):
    def to_list(poly):
        if isinstance(poly, np.ndarray):
            return poly.tolist()
        try:
            from shapely.geometry import (
                Polygon,
                MultiPolygon,
                GeometryCollection,
                Point,
                LineString,
                LinearRing,
            )
            from shapely.geometry.base import BaseGeometry

            def polygon_coords(geom):
                return [list(map(float, pt)) for pt in np.asarray(geom.exterior.coords)]

            if isinstance(poly, BaseGeometry):
                if poly.is_empty:
                    return []
                if isinstance(poly, Polygon):
                    return polygon_coords(poly)
                if isinstance(poly, MultiPolygon):
                    coords = [
                        polygon_coords(part)
                        for part in poly.geoms
                        if not part.is_empty
                    ]
                    if not coords:
                        return []
                    return coords if len(coords) > 1 else coords[0]
                if isinstance(poly, (LineString, LinearRing)):
                    return [list(map(float, pt)) for pt in poly.coords]
                if isinstance(poly, Point):
                    return [float(poly.x), float(poly.y)]
                if isinstance(poly, GeometryCollection):
                    parts = [
                        to_list(part)
                        for part in poly.geoms
                        if not getattr(part, "is_empty", False)
                    ]
                    parts = [part for part in parts if part not in (None, [], {})]
                    if not parts:
                        return []
                    return parts if len(parts) > 1 else parts[0]
                try:
                    coords = list(poly.coords)
                    if coords:
                        return [list(map(float, pt)) for pt in coords]
                except (AttributeError, NotImplementedError, TypeError):
                    pass
        except Exception:
            pass
        if hasattr(poly, "tolist"):
            return poly.tolist()
        try:
            return list(poly)
        except TypeError:
            return []

    payload = {
        "icons": [
            {
                "class": ICON_CLASSES[int(t["class"])],
                "points": to_list(polygons[i]),
            }
            for i, t in enumerate(types)
            if t["type"] == "icon"
        ],
        "walls": [
            {"class": int(t["class"]), "points": to_list(polygons[i])}
            for i, t in enumerate(types)
            if t["type"] == "wall"
        ],
        "rooms": [
            {
                "class": ROOM_CLASSES[int(t["class"])],
                "points": to_list(room_polygons[i]),
            }
            for i, t in enumerate(room_types)
        ],
    }

    with output_path.open("w", encoding="utf-8") as fp:
        json.dump(payload, fp, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description="Run CubiCasa floorplan inference on an arbitrary PNG/JPG image."
    )
    parser.add_argument(
        "--image",
        type=Path,
        required=True,
        help="Path to the input image (PNG, JPG, ...).",
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=Path("model_best_val_loss_var.pkl"),
        help="Path to the trained weights.",
    )
    parser.add_argument(
        "--arch",
        default="hg_furukawa_original",
        help="Model architecture name.",
    )
    parser.add_argument(
        "--device",
        choices=["cuda", "cpu"],
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Execution device.",
    )
    parser.add_argument(
        "--max-long-edge",
        type=int,
        default=1024,
        help="Resize the long edge of the image (0 keeps original size).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("runs_custom"),
        help="Output directory.",
    )
    parser.add_argument(
        "--save-polygons",
        action="store_true",
        help="Export predicted polygons as JSON.",
    )
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    img_rgb, image_tensor = load_image(args.image, args.max_long_edge)
    image_tensor = image_tensor.to(device)
    height, width = img_rgb.shape[:2]

    model = get_model(args.arch, 51)
    n_classes = 44
    model.conv4_ = torch.nn.Conv2d(256, n_classes, bias=True, kernel_size=1)
    model.upsample = torch.nn.ConvTranspose2d(
        n_classes, n_classes, kernel_size=4, stride=4
    )

    checkpoint = torch.load(args.weights, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()

    prediction = run_model(image_tensor, model, device)
    heatmaps, rooms, icons = split_prediction(prediction, (height, width), [21, 12, 11])

    rooms_seg = np.argmax(rooms, axis=0).astype(np.uint8)
    icons_seg = np.argmax(icons, axis=0).astype(np.uint8)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    np.save(args.out_dir / "rooms.npy", rooms_seg)
    np.save(args.out_dir / "icons.npy", icons_seg)

    room_palette = build_palette(ROOM_PALETTE)
    icon_palette = build_palette(ICON_PALETTE)
    save_mask(rooms_seg, room_palette, args.out_dir / "rooms.png")
    save_mask(icons_seg, icon_palette, args.out_dir / "icons.png")

    polygons, types, room_polygons, room_types = get_polygons(
        (heatmaps, rooms, icons), 0.4, [1, 2]
    )

    if args.save_polygons:
        export_polygons(args.out_dir / "polygons.json", polygons, types, room_polygons, room_types)

    print(f"Rooms mask: {args.out_dir / 'rooms.png'}")
    print(f"Icons mask: {args.out_dir / 'icons.png'}")
    if args.save_polygons:
        print(f"Polygons JSON: {args.out_dir / 'polygons.json'}")
    print(f"{len(room_polygons)} rooms, {len(polygons)} structural elements detected.")


if __name__ == "__main__":
    main()
