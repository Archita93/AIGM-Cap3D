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
import lpips

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

def f_score(mesh_a, mesh_b, device, threshold=0.01, n_samples=10000):
    pts_a = trimesh.sample.sample_surface(mesh_a, n_samples)[0]
    pts_b = trimesh.sample.sample_surface(mesh_b, n_samples)[0]

    pts_a_t = torch.tensor(pts_a, dtype=torch.float32).to(device)
    pts_b_t = torch.tensor(pts_b, dtype=torch.float32).to(device)

    diff_ab = pts_a_t.unsqueeze(1) - pts_b_t.unsqueeze(0)
    dist_ab = (diff_ab ** 2).sum(-1).sqrt()

    precision = (dist_ab.min(1).values < threshold).float().mean().item()
    recall    = (dist_ab.min(0).values < threshold).float().mean().item()

    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)

def clip_r_precision(image, correct_caption, all_captions, clip_model, preprocess, device, k=1):
    img_input = preprocess(image).unsqueeze(0).to(device)
    txt_inputs = clip.tokenize(all_captions, truncate=True).to(device)

    with torch.no_grad():
        img_feat = clip_model.encode_image(img_input)
        txt_feat = clip_model.encode_text(txt_inputs)

    img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
    txt_feat = txt_feat / txt_feat.norm(dim=-1, keepdim=True)

    sims = (img_feat * txt_feat).sum(-1)
    top_k = sims.topk(k).indices.tolist()
    correct_idx = all_captions.index(correct_caption)
    return 1.0 if correct_idx in top_k else 0.0


def lpips_score(image_a, image_b, loss_fn, device):
    def pil_to_tensor(img):
        arr = np.array(img.resize((224, 224))).astype(np.float32) / 127.5 - 1.0
        return torch.tensor(arr).permute(2, 0, 1).unsqueeze(0).to(device)

    ta = pil_to_tensor(image_a)
    tb = pil_to_tensor(image_b)
    with torch.no_grad():
        return loss_fn(ta, tb).item()
    
def chamfer_distance(mesh_a, mesh_b, device, n_samples=10000):
    """Chamfer distance between two meshes via point cloud sampling."""
    pts_a = trimesh.sample.sample_surface(mesh_a, n_samples)[0]
    pts_b = trimesh.sample.sample_surface(mesh_b, n_samples)[0]

    pts_a_t = torch.tensor(pts_a, dtype=torch.float32).to(device)
    pts_b_t = torch.tensor(pts_b, dtype=torch.float32).to(device)
 
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
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Loading CLIP on {device}...")
    clip_model, preprocess = clip.load("ViT-B/32", device=device)
    loss_fn = lpips.LPIPS(net='alex').to(device)

    # Build indexes
    gt_index  = build_index(gt_dir)
    gen_index = build_index(gen_dir)
 
    # Load dataset order from JSON
    with open(json_path) as f:
        data = json.load(f)
    
    all_captions = [item["caption"] for item in data]

    # Match generated stems (e.g. "00_748eefe0") back to UIDs via prefix
    # Generated files are named {count:02d}_{uid[:8]}
    uid_to_stem = {item["uid"]: item["uid"] for item in data if item["uid"] in gen_index}

    results = []
    count = 0
 
    for item in data:
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

        print("  Rendering for CLIP...", end=" ", flush=True)
        gt_img  = render_mesh_to_image(gt_mesh,  resolution=render_res)
        gen_img = render_mesh_to_image(gen_mesh, resolution=render_res)
        print("done")

        cs  = clip_score(gen_img, caption, clip_model, preprocess, device)
        sim = clip_similarity(gt_img, gen_img, clip_model, preprocess, device)

        fs  = f_score(gt_mesh, gen_mesh, device)
        rp  = clip_r_precision(gen_img, caption, all_captions, clip_model, preprocess, device)
        lp  = lpips_score(gt_img, gen_img, loss_fn, device)
        
        cd = chamfer_distance(gt_mesh, gen_mesh, device)
 
        print(f"  CLIP Score (text-image):  {cs:.4f}")
        print(f"  CLIP Similarity (GT-gen): {sim:.4f}")
        print(f"  CLIP R-Precision:         {rp:.4f}")
        print(f"  LPIPS:                    {lp:.4f}")
        print(f"  F-Score:                  {fs:.4f}")
        print(f"  CMD:                      {cd:.6f}")


        results.append({
            "uid": uid, "caption": caption,
            "chamfer": round(cd, 6),
            "clip_score": round(cs, 4),
            "clip_similarity": round(sim, 4),
            "f_score": round(fs, 4),
            "clip_r_precision": round(rp, 4),
            "lpips": round(lp, 4),
        })        

        count += 1


    if results:
        avg_cd  = np.mean([r["chamfer"]         for r in results])
        avg_cs  = np.mean([r["clip_score"]       for r in results])
        avg_sim = np.mean([r["clip_similarity"]  for r in results])
        avg_fs  = np.mean([r["f_score"]          for r in results])
        avg_rp  = np.mean([r["clip_r_precision"] for r in results])
        avg_lp  = np.mean([r["lpips"]            for r in results])

        print(f"\n{'─'*50}")
        print(f"  Avg F-Score       : {avg_fs:.4f}   (higher is better)")
        print(f"  Avg R-Precision   : {avg_rp:.4f}   (higher is better)")
        print(f"  Avg LPIPS         : {avg_lp:.4f}   (lower is better)")
        print(f"  Avg Chamfer Dist  : {avg_cd:.6f}  (lower is better)")
        print(f"  Avg CLIP Score    : {avg_cs:.4f}   (higher is better)")
        print(f"  Avg CLIP Sim      : {avg_sim:.4f}   (higher is better)")
        print(f"  Objects evaluated : {len(results)}")
        print(f"{'─'*50}")
 
        # Save CSV
        with open(out_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
            writer.writerow({
            "uid": "AVERAGE", "caption": "",
            "chamfer": avg_cd,
            "clip_score": avg_cs,
            "clip_similarity": avg_sim,
            "f_score": avg_fs,
            "clip_r_precision": avg_rp,
            "lpips": avg_lp,
        })
        print(f"\nResults saved to {out_csv}")
    else:
        print("No pairs evaluated — check your GT/generated folder paths.")
 
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt",         default=os.path.expanduser("~/.objaverse/hf-objaverse-v1/glbs"),
                                        help="Ground truth GLB folder")
    parser.add_argument("--gen",        default="generated_finetuned",
                                        help="Generated OBJ/PLY folder")
    parser.add_argument("--json",       default="downloaded_objects_split.json",
                                        help="Path to downloaded_objects_split.json")
    parser.add_argument("--out",        default="eval_results_finetuned.csv",
                                        help="Output CSV path")
    parser.add_argument("--samples",    default=5, type=int,
                                        help="Number of pairs to evaluate")
    parser.add_argument("--resolution", default=224, type=int,
                                        help="Render resolution for CLIP (224 = CLIP native)")
    args = parser.parse_args()
 
    main(args.gt, args.gen, args.json, args.out, args.samples, args.resolution)
 












