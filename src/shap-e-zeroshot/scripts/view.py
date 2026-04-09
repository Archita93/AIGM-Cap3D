import os
import json
import argparse
import trimesh
import numpy as np
import pyrender
from PIL import Image


def build_index(base):
    """Walks the entire objaverse download folder and builds a dict of {uid: filepath}. 
    Objaverse stores GLBs in nested subdirectories so you can't just do a direct path lookup"""

    index = {}
    for root, _, files in os.walk(base):
        for f in files:
            if f.endswith(".glb") or f.endswith(".obj") or f.endswith(".ply"):
                stem = f.rsplit(".", 1)[0]
                index[stem] = os.path.join(root, f)
    return index

def load_mesh(path):
    """Loads a GLB file and handles two cases trimesh can return
    a Scene (multiple sub-meshes with transforms, most GLBs) or a bare Mesh"""

    loaded = trimesh.load(path)
    if isinstance(loaded, trimesh.Scene):
        mesh = loaded.to_geometry()
    else:
        mesh = loaded
 
    if mesh is None or not hasattr(mesh, "vertices") or len(mesh.vertices) == 0:
        return None
 
    mesh.apply_translation(-mesh.centroid)
    if mesh.scale > 0:
        mesh.apply_scale(1.0 / mesh.scale)
    return mesh

def make_scene(mesh):
    """Creates a pyrender scene with a white background and ambient light."""

    scene = pyrender.Scene(ambient_light=[0.4, 0.4, 0.4], bg_color=[1.0, 1.0, 1.0, 1.0])
    
    """converts the trimesh object into a pyrender-renderable mesh"""
    render_mesh = pyrender.Mesh.from_trimesh(mesh, smooth=True)
    scene.add(render_mesh)
    return scene


def camera_pose_for_azim(azim_deg, mesh, elev_deg=20, padding=2.0):
    """Computes a 4x4 camera transform matrix for a given rotation angle around the object"""

    extents = mesh.bounds[1] - mesh.bounds[0]
    radius = np.linalg.norm(extents) * 0.5
    distance = radius * padding * 2.0

    azim = np.radians(azim_deg)
    elev = np.radians(elev_deg)
 
    x = distance * np.cos(elev) * np.sin(azim)
    y = distance * np.sin(elev)
    z = distance * np.cos(elev) * np.cos(azim)
 
    eye = np.array([x, y, z])
    target = np.array([0.0, 0.0, 0.0])
    up = np.array([0.0, 1.0, 0.0])
 
    z_ax = (eye - target); z_ax /= np.linalg.norm(z_ax)
    x_ax = np.cross(up, z_ax); x_ax /= np.linalg.norm(x_ax)
    y_ax = np.cross(z_ax, x_ax)
    pose = np.eye(4)
    pose[:3, 0] = x_ax
    pose[:3, 1] = y_ax
    pose[:3, 2] = z_ax
    pose[:3, 3] = eye
    return pose


def render_frame(scene, camera_pose, renderer):
    """Adds a camera to the scene, renders one frame, then removes the camera.
    # The camera is added/removed each frame because we need a different pose per frame"""
    camera = pyrender.PerspectiveCamera(yfov=np.pi / 4.0, aspectRatio=1.0)
    cam_node = scene.add(camera, pose=camera_pose)
    color, _ = renderer.render(scene)
    scene.remove_node(cam_node)
    return color


def make_gradient_bg(resolution):
    """Creates a dark grey radial-ish vertical gradient."""
    bg = Image.new("RGB", (resolution, resolution))
    for y in range(resolution):
        t = y / resolution
        val = int(40 + (80 - 40) * t)  # dark at top, slightly lighter at bottom
        for x in range(resolution):
            bg.putpixel((x, y), (val, val, val))
    return bg

def make_gif(mesh, out_path, num_frames=24, resolution=512):
    """builds the scene, adds three directional lights spaced 120° apart for even illumination, 
    creates the offscreen renderer, loops through evenly-spaced azimuth angles rendering one 
    frame each, then stitches all frames into an animated GIF using Pillow"""

    """render on black"""
    scene = pyrender.Scene(ambient_light=[0.4, 0.4, 0.4], bg_color=[0.0, 0.0, 0.0, 0.0])
    render_mesh = pyrender.Mesh.from_trimesh(mesh, smooth=True)
    scene.add(render_mesh)

    for light_azim in [45, 180, 270]:
        lpose = camera_pose_for_azim(light_azim, mesh, elev_deg=40)
        light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=4.0)
        scene.add(light, pose=lpose)

    renderer = pyrender.OffscreenRenderer(resolution, resolution)
    bg = make_gradient_bg(resolution)
    frames = []

    for azim in np.linspace(0, 360, num_frames, endpoint=False):
        pose = camera_pose_for_azim(azim, mesh)
        camera = pyrender.PerspectiveCamera(yfov=np.pi / 4.0, aspectRatio=1.0)
        cam_node = scene.add(camera, pose=pose)
        color, depth = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
        scene.remove_node(cam_node)

        """ Composite: use depth mask to separate object from background """
        fg = Image.fromarray(color, "RGBA")
        frame = bg.copy()
        frame.paste(fg, mask=fg.split()[3])  # alpha channel as mask
        frames.append(frame.convert("P", palette=Image.ADAPTIVE))

    renderer.delete()

    frames[0].save(
        out_path,
        save_all=True,
        append_images=frames[1:],
        loop=0,
        duration=80,
        optimize=True,
    )


def run(input_dir, out_dir, num_samples, num_frames, resolution, json_path=None):
    os.makedirs(out_dir, exist_ok=True)
    index = build_index(input_dir)
    print(f"Found {len(index)} objects in {input_dir}")

    if json_path:
        with open(json_path) as f:
            data = json.load(f)
            items = [(item["uid"], item["caption"]) for item in data if item["uid"] in index]
    else:
        # fallback: just use filenames sorted, no captions from JSON
        items = [(stem, None) for stem in sorted(index.keys())]

    count = 0
    for uid, json_caption in items:
        # if count >= num_samples:
        #     break
        path = index[uid]

        # Try to load caption from same dir or previews dir
        caption = json_caption or uid
        for txt_dir in [input_dir, "previews"]:
            txt_path = os.path.join(txt_dir, f"{uid[:8] if json_caption else uid}.txt")
            if os.path.exists(txt_path):
                caption = open(txt_path).read().strip()
                break
 
        print(f"\n[{count}] {caption[:80]}")
 
        try:
            mesh = load_mesh(path)
            if mesh is None:
                print("  Empty mesh, skipping")
                continue

            # Name by position + uid prefix when using JSON, else full stem
            stem = f"{count:02d}_{uid[:8]}" if json_caption else uid
            gif_path = os.path.join(out_dir, f"{stem}.gif")
            txt_out  = os.path.join(out_dir, f"{stem}.txt")

            print(f"  Rendering {num_frames} frames...", end=" ", flush=True)
            make_gif(mesh, gif_path, num_frames=num_frames, resolution=resolution)

            with open(txt_out, "w") as f:
                f.write(caption)

            print(f"Saved: {gif_path}")
            count += 1

        except Exception as e:
            print(f"  Error: {e}")
            continue

    print(f"\nDone! {count} GIFs saved to ./{out_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames",     default=24,  type=int, help="Number of GIF frames")
    parser.add_argument("--resolution", default=512, type=int, help="Resolution in pixels")
    parser.add_argument("--input",   default="previews",           help="Folder with GLB/OBJ/PLY files")
    parser.add_argument("--output",  default="previews_gif",       help="Folder to save GIFs")
    parser.add_argument("--samples", default=5, type=int,          help="Max number of objects to render")
    parser.add_argument("--json", default=None, help="Path to cap3d_split.json to filter by UID order")
    args = parser.parse_args()
 
run(args.input, args.output, args.samples, args.frames, args.resolution, args.json)
