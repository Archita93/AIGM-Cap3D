import json
from pathlib import Path
from tqdm import tqdm
import numpy as np
import torch
import clip
from PIL import Image
import shutil
from scipy.spatial import cKDTree
import trimesh
import objaverse
import pyrender
import lpips
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, preprocess = clip.load("ViT-B/32", device=device)
lpips_model = lpips.LPIPS(net='vgg').to(device)


def render_mesh(mesh, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)

    if isinstance(mesh, str):
        mesh = trimesh.load(mesh)

    mesh = pyrender.Mesh.from_trimesh(mesh)

    camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)
    light = pyrender.DirectionalLight(color=np.ones(3), intensity=3.0)

    image_paths = []

    r = pyrender.OffscreenRenderer(224, 224)

    for i in range(4):
        angle = i * (np.pi / 2)

        cam_pose = np.array([
            [np.cos(angle), 0, np.sin(angle), 2.5*np.sin(angle)],
            [0, 1, 0, 0],
            [-np.sin(angle), 0, np.cos(angle), 2.5*np.cos(angle)],
            [0, 0, 0, 1]
        ])

        scene = pyrender.Scene()

        scene.add(mesh)
        scene.add(camera, pose=cam_pose)
        scene.add(light, pose=cam_pose)

        color, _ = r.render(scene)

        img_path = output_dir / f"{i}.png"
        Image.fromarray(color).save(img_path)

        image_paths.append(img_path)

    r.delete()

    return image_paths

def normalize_mesh(mesh):
    vertices = mesh.vertices

    center = vertices.mean(axis=0)
    vertices = vertices - center

    scale = np.max(np.linalg.norm(vertices, axis=1))
    if scale < 1e-6:
        return mesh
    
    vertices = vertices / scale

    mesh.vertices = vertices
    return mesh

def sample_points(mesh, n=10000):
    return mesh.sample(n)

def chamfer_distance(p1, p2):
    tree1 = cKDTree(p1)
    tree2 = cKDTree(p2)

    dist1, _ = tree1.query(p2)
    dist2, _ = tree2.query(p1)

    return np.mean(dist1**2) + np.mean(dist2**2)

def clip_score(image, text):
    image = preprocess(Image.open(image)).unsqueeze(0).to(device)
    text = clip.tokenize([text]).to(device)

    with torch.no_grad():
        img_f = clip_model.encode_image(image)
        txt_f = clip_model.encode_text(text)

        img_f /= img_f.norm(dim=-1, keepdim=True)
        txt_f /= txt_f.norm(dim=-1, keepdim=True)

        return (img_f @ txt_f.T).item()

def clip_image_embedding(image_path):
    image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)

    with torch.no_grad():
        feat = clip_model.encode_image(image)
        feat /= feat.norm(dim=-1, keepdim=True)

    return feat

def clip_similarity(img1, img2):
    f1 = clip_image_embedding(img1)
    f2 = clip_image_embedding(img2)
    return (f1 @ f2.T).item()

def f_score(p1, p2, threshold=0.01):
    tree1 = cKDTree(p1)
    tree2 = cKDTree(p2)

    dist1, _ = tree1.query(p2)
    dist2, _ = tree2.query(p1)

    precision = np.mean(dist2 < threshold)
    recall = np.mean(dist1 < threshold)

    if precision + recall == 0:
        return 0.0

    return 2 * precision * recall / (precision + recall)

def lpips_score(img1_path, img2_path):
    img1 = preprocess(Image.open(img1_path)).unsqueeze(0).to(device)
    img2 = preprocess(Image.open(img2_path)).unsqueeze(0).to(device)

    img1 = img1 * 2 - 1
    img2 = img2 * 2 - 1

    with torch.no_grad():
        dist = lpips_model(img1, img2)

    return dist.item()

def load_gt_mesh(uid: str):
    objs = objaverse.load_objects(uids=[uid])

    if not objs:
        raise ValueError(f"Missing GT for uid: {uid}")

    path = list(objs.values())[0]
    return trimesh.load(path, force="mesh")

def evaluation(pred_dir, dataset, output_path):
    pred_dir = Path(pred_dir)

    chamfers, clip_scores, clip_sims, f_scores, lpips = [], [], [], [], []

    temp_root = Path("temp_eval")
    temp_root.mkdir(exist_ok=True)

    for i, item in enumerate(tqdm(dataset)):
        uid = item["uid"]
        prompt = item["caption"]

        pred_mesh_path = pred_dir / f"{uid}.obj"

        if not pred_mesh_path.exists():
            continue

        try:
            pred_mesh = trimesh.load(pred_mesh_path, force="mesh")
            gt_mesh = load_gt_mesh(uid)

            # temp folders
            render_pred = temp_root / f"{i}_pred"
            render_gt = temp_root / f"{i}_gt"

            if render_pred.exists():
                shutil.rmtree(render_pred)
            if render_gt.exists():
                shutil.rmtree(render_gt)

            render_pred.mkdir(parents=True, exist_ok=True)
            render_gt.mkdir(parents=True, exist_ok=True)

            # normalize meshes
            pred_mesh = normalize_mesh(pred_mesh)
            gt_mesh = normalize_mesh(gt_mesh)

            # render views
            pred_imgs = render_mesh(pred_mesh, render_pred)
            gt_imgs = render_mesh(gt_mesh, render_gt)

            # Chamfer
            p1 = sample_points(pred_mesh)
            p2 = sample_points(gt_mesh)
            chamfers.append(chamfer_distance(p1, p2))

            # CLIP text-score
            clip_scores.append(
                np.mean([clip_score(img, prompt) for img in pred_imgs])
            )

            # CLIP image similarity
            sims = [
                clip_similarity(ip, ig)
                for ip, ig in zip(pred_imgs, gt_imgs)
            ]
            clip_sims.append(np.mean(sims))

            # F-score
            f_scores.append(f_score(p1, p2))

            # LPIPS
            lpips.append(
                np.mean([
                    lpips_score(ip, ig)
                    for ip, ig in zip(pred_imgs, gt_imgs)
                ])
            )

            shutil.rmtree(render_pred)
            shutil.rmtree(render_gt)

        except Exception as e:
            print(f"[WARN] skipped {uid}: {e}")

    results = {
        "num_objects": len(chamfers),
        "chamfer_distance": float(np.mean(chamfers)) if chamfers else None,
        "clip_score": float(np.mean(clip_scores)) if clip_scores else None,
        "clip_similarity": float(np.mean(clip_sims)) if clip_sims else None,
        "f_score": float(np.mean(f_scores)) if f_scores else None,
        "lpips": float(np.mean(lpips)) if lpips else None,
    }

    print("\n" + "─" * 50)
    print(f"Objects evaluated : {results['num_objects']}")
    print(f"Chamfer Distance  : {results['chamfer_distance']}")
    print(f"CLIP Score        : {results['clip_score']}")
    print(f"CLIP Similarity   : {results['clip_similarity']}")
    print(f"F-Score           : {results['f_score']}")
    print(f"LPIPS             : {results['lpips']}")
    print("─" * 50)

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    return results