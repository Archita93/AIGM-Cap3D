import os
import argparse
import json
import numpy as np
import trimesh
import torch
import clip
import pyrender
from PIL import Image
import csv
import ssl
import certifi
ssl._create_default_https_context = lambda: ssl.create_default_context(cafile=certifi.where())


def build_index(base, exts=(".glb", ".obj", ".ply")):
    index = {}
    for root, _, files in os.walk(base):
        for f in files:
            if any(f.endswith(e) for e in exts):
                stem = f.rsplit(".", 1)[0]
                index[stem] = os.path.join(root, f)
    return index

def load_mesh(path):
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

def camera_pose_for_azim(azim_deg, mesh, elev_deg=20, padding=2.0):
    extents = mesh.bounds[1] - mesh.bounds[0]
    radius = np.linalg.norm(extents) * 0.5
    distance = radius * padding * 2.0
    azim = np.radians(azim_deg)
    elev = np.radians(elev_deg)
    x = distance * np.cos(elev) * np.sin(azim)
    y = distance * np.sin(elev)
    z = distance * np.cos(elev) * np.cos(azim)
    eye = np.array([x, y, z])
    z_ax = eye / np.linalg.norm(eye)
    x_ax = np.cross([0, 1, 0], z_ax); x_ax /= np.linalg.norm(x_ax)
    y_ax = np.cross(z_ax, x_ax)
    pose = np.eye(4)
    pose[:3, 0] = x_ax
    pose[:3, 1] = y_ax
    pose[:3, 2] = z_ax
    pose[:3, 3] = eye
    return pose


def render_mesh_to_image(mesh, resolution=224, azim=45):
    """Render a single view of a mesh, returns PIL Image."""
    scene = pyrender.Scene(ambient_light=[0.5, 0.5, 0.5], bg_color=[1.0, 1.0, 1.0, 1.0])
    scene.add(pyrender.Mesh.from_trimesh(mesh, smooth=True))
 
    for la in [45, 180, 270]:
        lpose = camera_pose_for_azim(la, mesh, elev_deg=40)
        scene.add(pyrender.DirectionalLight(color=[1, 1, 1], intensity=4.0), pose=lpose)
 
    pose = camera_pose_for_azim(azim, mesh)
    cam = pyrender.PerspectiveCamera(yfov=np.pi / 4.0, aspectRatio=1.0)
    cam_node = scene.add(cam, pose=pose)
 
    renderer = pyrender.OffscreenRenderer(resolution, resolution)
    color, _ = renderer.render(scene)
    renderer.delete()
    scene.remove_node(cam_node)
    return Image.fromarray(color)


def chamfer_distance(mesh_a, mesh_b, n_samples=10000):
    """Chamfer distance between two meshes via point cloud sampling."""
    pts_a = trimesh.sample.sample_surface(mesh_a, n_samples)[0]
    pts_b = trimesh.sample.sample_surface(mesh_b, n_samples)[0]
 
    pts_a_t = torch.tensor(pts_a, dtype=torch.float32)
    pts_b_t = torch.tensor(pts_b, dtype=torch.float32)
 
    # (N, M) pairwise distances
    diff_ab = pts_a_t.unsqueeze(1) - pts_b_t.unsqueeze(0)   # (N, M, 3)
    dist_ab = (diff_ab ** 2).sum(-1)                          # (N, M)
 
    cd = dist_ab.min(1).values.mean() + dist_ab.min(0).values.mean()
    return cd.item()


def clip_score(image, text, clip_model, preprocess, device):
    """CLIP cosine similarity between a rendered image and a text caption."""
    img_input = preprocess(image).unsqueeze(0).to(device)
    txt_input = clip.tokenize([text], truncate=True).to(device)
 
    with torch.no_grad():
        img_feat = clip_model.encode_image(img_input)
        txt_feat = clip_model.encode_text(txt_input)
 
    img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
    txt_feat = txt_feat / txt_feat.norm(dim=-1, keepdim=True)
    return (img_feat * txt_feat).sum().item()


def clip_similarity(image_a, image_b, clip_model, preprocess, device):
    """CLIP cosine similarity between two rendered images."""
    def encode(img):
        t = preprocess(img).unsqueeze(0).to(device)
        with torch.no_grad():
            f = clip_model.encode_image(t)
        return f / f.norm(dim=-1, keepdim=True)
 
    fa = encode(image_a)
    fb = encode(image_b)
    return (fa * fb).sum().item()


def main(gt_dir, gen_dir, json_path, out_csv, n_samples, render_res):
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Loading CLIP on {device}...")
    clip_model, preprocess = clip.load("ViT-B/32", device=device)


    # Build indexes
    gt_index  = build_index(gt_dir)
    gen_index = build_index(gen_dir)
 
    # Load dataset order from JSON
    with open(json_path) as f:
        data = json.load(f)
 
    # Match generated stems (e.g. "00_748eefe0") back to UIDs via prefix
    # Generated files are named {count:02d}_{uid[:8]}
    uid_to_stem = {}
    for item in data["train"]:
        uid = item["uid"]
        prefix = uid[:8]
        for stem in gen_index:
            if stem.endswith(prefix):
                uid_to_stem[uid] = stem
                break


    results = []
    count = 0
 
    for item in data["train"]:
        if count >= n_samples:
            break
 
        uid     = item["uid"]
        caption = item["caption"]
        gen_stem = uid_to_stem.get(uid)
 
        if uid not in gt_index:
            print(f"  Missing GT:  {uid[:16]}...")
            continue
        if gen_stem is None or gen_stem not in gen_index:
            print(f"  Missing gen: {uid[:16]}...")
            continue
 
        print(f"\n[{count}] {caption[:70]}")
 
        gt_mesh  = load_mesh(gt_index[uid])
        gen_mesh = load_mesh(gen_index[gen_stem])
 
        if gt_mesh is None or gen_mesh is None:
            print("  Skipping — empty mesh")
            continue

        # 1. Chamfer Distance
        print("  Computing Chamfer Distance...", end=" ", flush=True)
        cd = chamfer_distance(gt_mesh, gen_mesh)
        print(f"{cd:.6f}")
 
        # 2 & 3. Render both meshes, compute CLIP scores
        print("  Rendering for CLIP...", end=" ", flush=True)
        gt_img  = render_mesh_to_image(gt_mesh,  resolution=render_res)
        gen_img = render_mesh_to_image(gen_mesh, resolution=render_res)
        print("done")

        cs  = clip_score(gen_img, caption, clip_model, preprocess, device)
        sim = clip_similarity(gt_img, gen_img, clip_model, preprocess, device)
 
        print(f"  CLIP Score (text-image):  {cs:.4f}")
        print(f"  CLIP Similarity (GT-gen): {sim:.4f}")
 
        results.append({
            "uid":            uid,
            "caption":        caption,
            "chamfer":        round(cd,  6),
            "clip_score":     round(cs,  4),
            "clip_similarity":round(sim, 4),
        })
        count += 1


    if results:
        avg_cd  = np.mean([r["chamfer"]         for r in results])
        avg_cs  = np.mean([r["clip_score"]       for r in results])
        avg_sim = np.mean([r["clip_similarity"]  for r in results])
        print(f"\n{'─'*50}")
        print(f"  Objects evaluated : {len(results)}")
        print(f"  Avg Chamfer Dist  : {avg_cd:.6f}  (lower is better)")
        print(f"  Avg CLIP Score    : {avg_cs:.4f}   (higher is better)")
        print(f"  Avg CLIP Sim      : {avg_sim:.4f}   (higher is better)")
        print(f"{'─'*50}")
 
        # Save CSV
        with open(out_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
            writer.writerow({
                "uid": "AVERAGE", "caption": "",
                "chamfer": avg_cd, "clip_score": avg_cs, "clip_similarity": avg_sim
            })
        print(f"\nResults saved to {out_csv}")
    else:
        print("No pairs evaluated — check your GT/generated folder paths.")
 
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt",         default=os.path.expanduser("~/.objaverse/hf-objaverse-v1/glbs"),
                                        help="Ground truth GLB folder")
    parser.add_argument("--gen",        default="generated",
                                        help="Generated OBJ/PLY folder")
    parser.add_argument("--json",       default="../cap3d_split.json",
                                        help="Path to cap3d_split.json")
    parser.add_argument("--out",        default="eval_results.csv",
                                        help="Output CSV path")
    parser.add_argument("--samples",    default=10, type=int,
                                        help="Number of pairs to evaluate")
    parser.add_argument("--resolution", default=224, type=int,
                                        help="Render resolution for CLIP (224 = CLIP native)")
    args = parser.parse_args()
 
    main(args.gt, args.gen, args.json, args.out, args.samples, args.resolution)
 












