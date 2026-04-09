import json
import numpy as np
import pyrender
import matplotlib.pyplot as plt
import trimesh
from pathlib import Path
import objaverse

from config import (
    FANTASIA3D_ZEROSHOT_MESH_DIR,
    PROCESSED_DIR,
    RESULTS_DIR
)


def normalize_mesh(mesh):
    mesh = mesh.copy()
    mesh.apply_translation(-mesh.centroid)
    scale = np.max(mesh.extents)
    if scale > 0:
        mesh.apply_scale(1.0 / scale)
    return mesh

def set_vertex_color(mesh, color):
    color = np.array(color, dtype=np.uint8)
    mesh.visual.vertex_colors = np.tile(color, (len(mesh.vertices), 1))
    return mesh

def render_views(mesh, n_views=6):
    mesh = pyrender.Mesh.from_trimesh(mesh, smooth=False)

    camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)
    light = pyrender.DirectionalLight(color=np.ones(3), intensity=3.0)

    r = pyrender.OffscreenRenderer(400, 400)

    images = []

    for i in range(n_views):
        angle = i * (2 * np.pi / n_views)

        cam_pose = np.array([
            [np.cos(angle), 0, np.sin(angle), 2.5*np.sin(angle)],
            [0, 1, 0, 0],
            [-np.sin(angle), 0, np.cos(angle), 2.5*np.cos(angle)],
            [0, 0, 0, 1]
        ])

        scene = pyrender.Scene(ambient_light=[0.3, 0.3, 0.3])

        scene.add(mesh)
        scene.add(camera, pose=cam_pose)
        scene.add(light, pose=cam_pose)

        color, _ = r.render(scene)
        images.append(color)

    r.delete()
    return images

def save_grid(images, out_path, cols=3):
    rows = int(np.ceil(len(images) / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))

    axes = np.array(axes).reshape(-1)

    for ax in axes:
        ax.axis("off")

    for i, img in enumerate(images):
        axes[i].imshow(img)

    plt.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

def run():
    with open(PROCESSED_DIR / "downloaded_objects_split.json") as f:
        split = json.load(f)

    sample = split[0]
    uid = sample["uid"]

    # Load meshes
    zeroshot_mesh = trimesh.load(FANTASIA3D_ZEROSHOT_MESH_DIR / f"{uid}.obj", force="mesh")
    objects = objaverse.load_objects(uids=[uid])
    path = list(objects.values())[0]    

    # Normalize
    zeroshot_mesh = normalize_mesh(zeroshot_mesh)
    gt_mesh = normalize_mesh(trimesh.load(path, force="mesh"))

    # Render separately
    zs_imgs = render_views(zeroshot_mesh)
    gt_imgs = render_views(gt_mesh)

    # Save outputs
    save_grid(
        zs_imgs,
        RESULTS_DIR / f"{uid}_zeroshot_views.png",
        cols=3
    )

    save_grid(
        gt_imgs,
        RESULTS_DIR / f"{uid}_gt_views.png",
        cols=3
    )


if __name__ == "__main__":
    run()