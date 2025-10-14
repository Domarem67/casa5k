import argparse
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
import trimesh


def load_mesh(mesh_path: Path) -> trimesh.Trimesh:
    mesh = trimesh.load(mesh_path, force="mesh")
    if isinstance(mesh, trimesh.Scene):
        meshes = mesh.dump()
        if not meshes:
            raise RuntimeError("GLB scene does not contain any mesh geometry.")
        mesh = trimesh.util.concatenate([m for m in meshes if isinstance(m, trimesh.Trimesh)])
    return mesh


def face_colors(mesh: trimesh.Trimesh):
    if mesh.visual.kind == "face":
        colors = mesh.visual.face_colors
    elif mesh.visual.kind == "vertex" and mesh.visual.vertex_colors is not None:
        vertex_colors = mesh.visual.vertex_colors
        colors = vertex_colors[mesh.faces].mean(axis=1)
    else:
        colors = np.tile(np.array([[190, 190, 205, 255]], dtype=np.float32), (len(mesh.faces), 1))
    colors = np.clip(colors, 0, 255)
    rgba = [f"rgba({int(c[0])},{int(c[1])},{int(c[2])},{c[3]/255.0:.3f})" for c in colors]
    return rgba


def build_trace(mesh: trimesh.Trimesh):
    vertices = mesh.vertices
    faces = mesh.faces
    colors = face_colors(mesh)
    trace = go.Mesh3d(
        x=vertices[:, 0],
        y=vertices[:, 1],
        z=vertices[:, 2],
        i=faces[:, 0],
        j=faces[:, 1],
        k=faces[:, 2],
        facecolor=colors,
        flatshading=True,
        lighting=dict(ambient=0.4, diffuse=0.7, specular=0.3, roughness=0.9),
        lightposition=dict(x=0.5, y=0.5, z=1.0),
        showscale=False,
    )
    return trace


def configure_scene(mesh: trimesh.Trimesh):
    extents = mesh.extents
    bbox = mesh.bounding_box.bounds
    center = mesh.centroid
    padding = max(extents.max(), 1.0) * 0.1
    x_range = [bbox[0] - padding, bbox[3] + padding]
    y_range = [bbox[1] - padding, bbox[4] + padding]
    z_range = [bbox[2] - padding, bbox[5] + padding]
    aspect = dict(x=extents[0], y=extents[1], z=extents[2] if extents[2] > 0 else 1.0)
    camera_iso = dict(eye=dict(x=1.7, y=1.4, z=1.2), center=dict(x=0, y=0, z=-0.05))

    scene = dict(
        xaxis=dict(visible=False, range=x_range),
        yaxis=dict(visible=False, range=y_range),
        zaxis=dict(visible=False, range=z_range),
        aspectmode="data",
        aspectratio=aspect,
        camera=camera_iso,
    )
    return scene


def render(mesh_path: Path, output_dir: Path, width: int, height: int, topdown: bool):
    output_dir.mkdir(parents=True, exist_ok=True)
    mesh = load_mesh(mesh_path)
    trace = build_trace(mesh)
    scene_conf = configure_scene(mesh)

    fig = go.Figure(data=[trace])
    fig.update_layout(
        scene=scene_conf,
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor="rgba(255,255,255,0)",
    )

    perspective_path = output_dir / "render_isometric.png"
    fig.write_image(str(perspective_path), width=width, height=height, engine="kaleido")
    print(f"[render_rooms_plotly] Saved {perspective_path}")

    if topdown:
        topdown_fig = fig
        topdown_fig.update_layout(
            scene=dict(
                **scene_conf,
                camera=dict(eye=dict(x=0, y=0, z=2.4), up=dict(x=0, y=1, z=0)),
            )
        )
        topdown_path = output_dir / "render_topdown.png"
        topdown_fig.write_image(str(topdown_path), width=width, height=height, engine="kaleido")
        print(f"[render_rooms_plotly] Saved {topdown_path}")


def main():
    parser = argparse.ArgumentParser(description="Software fallback renderer using Plotly + Kaleido.")
    parser.add_argument("--mesh", type=Path, required=True, help="Path to GLB mesh (e.g. walls.glb).")
    parser.add_argument("--output", type=Path, required=True, help="Directory to save renders.")
    parser.add_argument("--image-width", type=int, default=1280, help="Image width in pixels.")
    parser.add_argument("--image-height", type=int, default=720, help="Image height in pixels.")
    parser.add_argument(
        "--topdown",
        dest="topdown",
        action="store_true",
        help="Additionally create a top-down orthographic render.",
    )
    parser.set_defaults(topdown=False)
    args = parser.parse_args()

    # Ensure Kaleido backend is available
    try:
        import kaleido  # noqa: F401
    except Exception as exc:
        raise RuntimeError(
            "Kaleido is required for static image export. Install with `pip install kaleido`."
        ) from exc

    render(args.mesh, args.output, args.image_width, args.image_height, args.topdown)


if __name__ == "__main__":
    pio.kaleido.scope.default_format = "png"
    pio.kaleido.scope.default_width = 1280
    pio.kaleido.scope.default_height = 720
    main()
