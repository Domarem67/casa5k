import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent


def run_command(cmd, cwd, description):
    print(f"[run_full_pipeline] {description}: {' '.join(cmd)}")
    result = subprocess.run(
        cmd,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
    )
    if result.returncode != 0:
        print(result.stdout)
        print(result.stderr, file=sys.stderr)
        raise RuntimeError(f"{description} failed with exit code {result.returncode}")
    if result.stdout.strip():
        print(result.stdout)
    if result.stderr.strip():
        print(result.stderr)


def resolve_scale(scale_arg, polygons_path, door_width_ref, fallback):
    if isinstance(scale_arg, str) and scale_arg.lower() == "auto":
        with polygons_path.open("r", encoding="utf-8") as fp:
            data = json.load(fp)
        lengths = []
        for icon in data.get("icons", []):
            if icon.get("class") != "Door":
                continue
            points = icon.get("points", [])
            if not points:
                continue
            xs = [p[0] for p in points]
            ys = [p[1] for p in points]
            span = max(max(xs) - min(xs), max(ys) - min(ys))
            if span > 0:
                lengths.append(span)
        if lengths:
            avg_span = sum(lengths) / len(lengths)
            return door_width_ref / avg_span
        return fallback
    return float(scale_arg)


def main():
    parser = argparse.ArgumentParser(
        description="End-to-end pipeline: PNG → inference → SVG → mesh."
    )
    parser.add_argument("--image", type=Path, required=True, help="Input PNG/JPG.")
    parser.add_argument(
        "--weights",
        type=Path,
        default=Path("model_best_val_loss_var.pkl"),
        help="Model weights (.pkl).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("runs_full"),
        help="Output directory (results will be placed inside a subfolder).",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default="run",
        help="Name of the output subfolder.",
    )
    parser.add_argument(
        "--max-long-edge",
        type=int,
        default=1024,
        help="Inference resize (0 keeps original).",
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default="cpu",
        help="Device passed to infer_custom.py.",
    )
    parser.add_argument(
        "--wall-height",
        type=float,
        default=3.0,
        help="Extrusion height for walls (same units as scale).",
    )
    parser.add_argument(
        "--scale",
        default="auto",
        help="Pixel-to-unit scale for mesh export (float) or 'auto' to infer from doors.",
    )
    parser.add_argument(
        "--door-width-ref",
        type=float,
        default=0.9,
        help="Physical width assumed for doors when inferring scale (meters).",
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
        default=0.5,
        help="Padding around door/window openings before subtraction.",
    )
    parser.add_argument(
        "--window-sill",
        type=float,
        default=1.0,
        help="Window sill height for wall extrusion.",
    )
    parser.add_argument(
        "--window-head",
        type=float,
        default=2.1,
        help="Window head height for wall extrusion.",
    )
    parser.add_argument(
        "--door-height",
        type=float,
        default=2.1,
        help="Door opening height for frames.",
    )
    parser.add_argument(
        "--door-frame-depth",
        type=float,
        default=0.05,
        help="Door frame thickness (0 to skip).",
    )
    parser.add_argument(
        "--floor-thickness",
        type=float,
        default=0.2,
        help="Floor slab thickness (0 to skip).",
    )
    parser.add_argument(
        "--ceiling-thickness",
        type=float,
        default=0.1,
        help="Ceiling slab thickness (0 to skip).",
    )
    parser.add_argument(
        "--invert-z",
        dest="invert_z",
        action="store_true",
        help="Flip vertical axis so ceiling becomes floor (default).",
    )
    parser.add_argument(
        "--no-invert-z",
        dest="invert_z",
        action="store_false",
        help="Keep original vertical orientation.",
    )
    parser.set_defaults(invert_z=True)
    parser.add_argument(
        "--smooth-walls",
        action="store_true",
        help="Post-process to merge wall segments (also disables door frame).",
    )
    parser.add_argument(
        "--render-rooms",
        dest="render_rooms",
        action="store_true",
        help="Generate interior renders for each room (default).",
    )
    parser.add_argument(
        "--no-render-rooms",
        dest="render_rooms",
        action="store_false",
        help="Skip interior renders.",
    )
    parser.set_defaults(render_rooms=True)
    parser.add_argument(
        "--render-width",
        type=int,
        default=1280,
        help="Render width in pixels.",
    )
    parser.add_argument(
        "--render-height",
        type=int,
        default=720,
        help="Render height in pixels.",
    )
    parser.add_argument(
        "--keep-temp",
        action="store_true",
        help="Do not delete intermediate SVG/JSON files.",
    )
    args = parser.parse_args()

    output_root = args.out_dir / args.run_name
    output_root.mkdir(parents=True, exist_ok=True)

    # 1. Inference (PNG -> polygons.json)
    infer_cmd = [
        sys.executable,
        str(THIS_DIR / "infer_custom.py"),
        "--image",
        str(args.image),
        "--weights",
        str(args.weights),
        "--out-dir",
        str(output_root),
        "--device",
        args.device,
        "--max-long-edge",
        str(args.max_long_edge),
        "--save-polygons",
    ]
    run_command(infer_cmd, THIS_DIR, "Running inference")

    polygons_path = output_root / "polygons.json"
    if not polygons_path.exists():
        raise RuntimeError(f"Expected polygons.json at {polygons_path} not found.")

    resolved_scale = resolve_scale(args.scale, polygons_path, args.door_width_ref, args.fallback_scale)
    print(f"[run_full_pipeline] Using scale {resolved_scale:.6f}")

    # 2. Export SVG
    svg_path = output_root / "plan.svg"
    export_svg_cmd = [
        sys.executable,
        str(THIS_DIR / "export_svg.py"),
        "--input",
        str(polygons_path),
        "--output",
        str(svg_path),
    ]
    run_command(export_svg_cmd, THIS_DIR, "Exporting SVG")

    # 3. Build 3D mesh
    mesh_path = output_root / "walls.glb"
    mesh_cmd = [
        sys.executable,
        str(THIS_DIR / "build_walls_mesh.py"),
        "--svg",
        str(svg_path),
        "--output",
        str(mesh_path),
        "--height",
        str(args.wall_height),
        "--scale",
        f"{resolved_scale}",
        "--door-width-ref",
        str(args.door_width_ref),
        "--fallback-scale",
        str(args.fallback_scale),
        "--opening-padding",
        str(args.opening_padding),
        "--window-sill",
        str(args.window_sill),
        "--window-head",
        str(args.window_head),
        "--door-height",
        str(args.door_height),
        "--door-frame-depth",
        str(args.door_frame_depth),
        "--floor-thickness",
        str(args.floor_thickness),
        "--ceiling-thickness",
        str(args.ceiling_thickness),
    ]
    if args.invert_z:
        mesh_cmd.append("--invert-z")
    if args.smooth_walls:
        mesh_cmd.append("--smooth-walls")
    run_command(mesh_cmd, THIS_DIR, "Building wall mesh")

    if args.render_rooms:
        renders_dir = output_root / "renders"
        render_cmd = [
            sys.executable,
            str(THIS_DIR / "render_rooms.py"),
            "--mesh",
            str(mesh_path),
            "--polygons",
            str(polygons_path),
            "--output",
            str(renders_dir),
            "--scale",
            f"{resolved_scale}",
            "--wall-height",
            str(args.wall_height),
            "--image-width",
            str(args.render_width),
            "--image-height",
            str(args.render_height),
        ]
        run_command(render_cmd, THIS_DIR, "Rendering room images")

    if not args.keep_temp:
        for filename in ["polygons.json", "rooms.npy", "icons.npy"]:
            path = output_root / filename
            if path.exists():
                path.unlink()
        # Optionally keep rooms/icons PNG for visual reference

    print("\nPipeline completed successfully.")
    print(f"→ Mesh: {mesh_path}")
    print(f"→ Rooms mask: {output_root / 'rooms.png'}")
    print(f"→ Icons mask: {output_root / 'icons.png'}")
    print(f"→ SVG: {svg_path}")


if __name__ == "__main__":
    main()
