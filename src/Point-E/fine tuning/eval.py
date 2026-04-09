import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import io
import json
import csv
import numpy as np
import trimesh
import objaverse
import torch
import clip
import lpips
import torchvision.transforms as T

from PIL import Image
from tqdm import tqdm
from scipy.spatial import cKDTree

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from point_e.util.plotting import plot_point_cloud
from point_e.util.point_cloud import PointCloud

# =========================
# PATHS
# =========================
JSONL_PATH = r"../data/splits/cap3d_test_100.jsonl"
PRED_DIR   = r"../outputs/cap3d_finetune/point_e_bestof3_test_100"
CSV_OUT    = r"../reports/point_e_bestof3_test_100_eval.csv"

NUM_GT_POINTS = 4096
FSCORE_TAU = 0.01
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =========================
# HELPERS
# =========================
def normalize_points(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    x = x - x.mean(axis=0, keepdims=True)
    scale = np.abs(x).max()
    if scale > 0:
        x = x / scale
    return x

def chamfer_distance(pred: np.ndarray, gt: np.ndarray) -> float:
    tree_pred = cKDTree(pred)
    tree_gt = cKDTree(gt)

    dist_gt_to_pred, _ = tree_pred.query(gt)
    dist_pred_to_gt, _ = tree_gt.query(pred)

    return float(np.mean(dist_gt_to_pred ** 2) + np.mean(dist_pred_to_gt ** 2))

def fscore(pred: np.ndarray, gt: np.ndarray, tau: float = 0.01) -> float:
    tree_pred = cKDTree(pred)
    tree_gt = cKDTree(gt)

    dist_gt_to_pred, _ = tree_pred.query(gt)
    dist_pred_to_gt, _ = tree_gt.query(pred)

    recall = np.mean(dist_gt_to_pred < tau)
    precision = np.mean(dist_pred_to_gt < tau)

    if precision + recall == 0:
        return 0.0

    return float(2 * precision * recall / (precision + recall))

def make_red_pointcloud(coords: np.ndarray) -> PointCloud:
    n = coords.shape[0]
    return PointCloud(
        coords=coords,
        channels={
            "R": np.ones(n, dtype=np.float32),
            "G": np.zeros(n, dtype=np.float32),
            "B": np.zeros(n, dtype=np.float32),
        }
    )

def npz_to_pointcloud(npz_path: str) -> PointCloud:
    data = np.load(npz_path)
    coords = normalize_points(data["coords"])

    if {"R", "G", "B"}.issubset(set(data.files)):
        pc = PointCloud(
            coords=coords,
            channels={
                "R": data["R"].astype(np.float32),
                "G": data["G"].astype(np.float32),
                "B": data["B"].astype(np.float32),
            }
        )
    else:
        pc = make_red_pointcloud(coords)

    return pc

def gt_mesh_to_pointcloud(gt_path: str, num_points: int = 4096) -> PointCloud:
    mesh = trimesh.load(gt_path, force="mesh")
    points = mesh.sample(num_points)
    points = normalize_points(points)
    return make_red_pointcloud(points)

def figure_to_pil(fig) -> Image.Image:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")

def render_pointcloud_to_pil(pc: PointCloud) -> Image.Image:
    fig = plot_point_cloud(
        pc,
        grid_size=3,
        fixed_bounds=((-0.75, -0.75, -0.75), (0.75, 0.75, 0.75))
    )
    return figure_to_pil(fig)

# =========================
# CLIP
# =========================
clip_model, clip_preprocess = clip.load("ViT-B/32", device=DEVICE)
clip_model.eval()

@torch.no_grad()
def clip_score_image_text(img: Image.Image, text: str) -> float:
    image_input = clip_preprocess(img).unsqueeze(0).to(DEVICE)
    text_input = clip.tokenize([text]).to(DEVICE)

    image_features = clip_model.encode_image(image_input)
    text_features = clip_model.encode_text(text_input)

    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    score = (image_features @ text_features.T).item()
    return float(score)

@torch.no_grad()
def clip_similarity_image_image(img1: Image.Image, img2: Image.Image) -> float:
    x1 = clip_preprocess(img1).unsqueeze(0).to(DEVICE)
    x2 = clip_preprocess(img2).unsqueeze(0).to(DEVICE)

    f1 = clip_model.encode_image(x1)
    f2 = clip_model.encode_image(x2)

    f1 = f1 / f1.norm(dim=-1, keepdim=True)
    f2 = f2 / f2.norm(dim=-1, keepdim=True)

    sim = (f1 @ f2.T).item()
    return float(sim)

# =========================
# LPIPS
# =========================
lpips_model = lpips.LPIPS(net="alex").to(DEVICE)
lpips_model.eval()

lpips_transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
])

@torch.no_grad()
def lpips_image_image(img1: Image.Image, img2: Image.Image) -> float:
    x1 = lpips_transform(img1).unsqueeze(0).to(DEVICE)
    x2 = lpips_transform(img2).unsqueeze(0).to(DEVICE)

    # LPIPS expects [-1, 1]
    x1 = x1 * 2.0 - 1.0
    x2 = x2 * 2.0 - 1.0

    val = lpips_model(x1, x2).item()
    return float(val)

# =========================
# MAIN
# =========================
def main():
    os.makedirs(os.path.dirname(CSV_OUT), exist_ok=True)

    with open(JSONL_PATH, "r", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]

    uids = [row["id"] for row in rows]
    gt_paths = objaverse.load_objects(uids=uids)

    results = []

    for row in tqdm(rows, desc="Evaluating"):
        uid = row["id"]
        caption = row["caption"]

        pred_path = os.path.join(PRED_DIR, f"{uid}.npz")
        gt_path = gt_paths.get(uid, None)

        if not os.path.exists(pred_path):
            print(f"Missing pred: {uid}")
            continue

        if gt_path is None or not os.path.exists(gt_path):
            print(f"Missing GT: {uid}")
            continue

        try:
            # -------------------------
            # Predicted point cloud
            # -------------------------
            pred_data = np.load(pred_path)
            pred_coords = normalize_points(pred_data["coords"])

            if {"R", "G", "B"}.issubset(set(pred_data.files)):
                pred_pc = PointCloud(
                    coords=pred_coords,
                    channels={
                        "R": pred_data["R"].astype(np.float32),
                        "G": pred_data["G"].astype(np.float32),
                        "B": pred_data["B"].astype(np.float32),
                    }
                )
            else:
                pred_pc = make_red_pointcloud(pred_coords)

            # -------------------------
            # Ground truth point cloud
            # -------------------------
            mesh = trimesh.load(gt_path, force="mesh")
            gt_coords = mesh.sample(NUM_GT_POINTS)
            gt_coords = normalize_points(gt_coords)
            gt_pc = make_red_pointcloud(gt_coords)

            # -------------------------
            # Geometric metrics
            # -------------------------
            cd = chamfer_distance(pred_coords, gt_coords)
            fs = fscore(pred_coords, gt_coords, tau=FSCORE_TAU)

            # -------------------------
            # Render to images
            # -------------------------
            pred_img = render_pointcloud_to_pil(pred_pc)
            gt_img = render_pointcloud_to_pil(gt_pc)

            # -------------------------
            # Semantic + perceptual
            # -------------------------
            clip_score = clip_score_image_text(pred_img, caption)
            clip_sim = clip_similarity_image_image(pred_img, gt_img)
            lpips_val = lpips_image_image(pred_img, gt_img)

            results.append({
                "uid": uid,
                "caption": caption,
                "chamfer": cd,
                "fscore": fs,
                "clip_score": clip_score,
                "clip_similarity": clip_sim,
                "lpips": lpips_val,
            })

            print(
                f"{uid} | "
                f"CD={cd:.6f} | "
                f"FScore={fs:.4f} | "
                f"CLIPScore={clip_score:.4f} | "
                f"CLIPSim={clip_sim:.4f} | "
                f"LPIPS={lpips_val:.4f}"
            )

        except Exception as e:
            print(f"Failed on {uid}: {e}")

    # -------------------------
    # Save CSV
    # -------------------------
    with open(CSV_OUT, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "uid",
                "caption",
                "chamfer",
                "fscore",
                "clip_score",
                "clip_similarity",
                "lpips",
            ]
        )
        writer.writeheader()
        writer.writerows(results)

    # -------------------------
    # Print averages
    # -------------------------
    if results:
        avg_cd = np.mean([r["chamfer"] for r in results])
        avg_fs = np.mean([r["fscore"] for r in results])
        avg_clip_score = np.mean([r["clip_score"] for r in results])
        avg_clip_sim = np.mean([r["clip_similarity"] for r in results])
        avg_lpips = np.mean([r["lpips"] for r in results])

        print("\n" + "=" * 60)
        print(f"Objects evaluated: {len(results)}")
        print(f"Avg Chamfer Dist : {avg_cd:.6f} (lower is better)")
        print(f"Avg F-Score      : {avg_fs:.4f} (higher is better)")
        print(f"Avg CLIP Score   : {avg_clip_score:.4f} (higher is better)")
        print(f"Avg CLIP Sim     : {avg_clip_sim:.4f} (higher is better)")
        print(f"Avg LPIPS        : {avg_lpips:.4f} (lower is better)")
        print("=" * 60)
        print(f"Saved CSV to: {CSV_OUT}")
    else:
        print("No valid results were computed.")

if __name__ == "__main__":
    main()