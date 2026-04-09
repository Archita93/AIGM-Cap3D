import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import io
import json
import numpy as np
import torch
import clip
from PIL import Image
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from point_e.diffusion.configs import DIFFUSION_CONFIGS, diffusion_from_config
from point_e.diffusion.sampler import PointCloudSampler
from point_e.models.download import load_checkpoint
from point_e.models.configs import MODEL_CONFIGS, model_from_config
from point_e.util.plotting import plot_point_cloud
from point_e.util.point_cloud import PointCloud

INPUT_JSONL = r"../data/splits/cap3d_test_100.jsonl"
OUTPUT_DIR = r"../outputs/cap3d_finetune/point_e_bestof3_test_100"
META_JSONL = r"../outputs/cap3d_finetune/point_e_bestof3_test_100/selection_log.jsonl"

K = 3  # number of samples per caption

os.makedirs(OUTPUT_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)

# -------------------------
# Load CLIP
# -------------------------
clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)
clip_model.eval()

@torch.no_grad()
def clip_score_image_text(img: Image.Image, text: str) -> float:
    image_input = clip_preprocess(img).unsqueeze(0).to(device)
    text_input = clip.tokenize([text]).to(device)

    image_features = clip_model.encode_image(image_input)
    text_features = clip_model.encode_text(text_input)

    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    return float((image_features @ text_features.T).item())

def normalize_points(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    x = x - x.mean(axis=0, keepdims=True)
    scale = np.abs(x).max()
    if scale > 0:
        x = x / scale
    return x

def pointcloud_from_np_arrays(coords, r, g, b) -> PointCloud:
    return PointCloud(
        coords=coords,
        channels={
            "R": r.astype(np.float32),
            "G": g.astype(np.float32),
            "B": b.astype(np.float32),
        }
    )

def render_pointcloud_to_pil(pc: PointCloud) -> Image.Image:
    fig = plot_point_cloud(
        pc,
        grid_size=3,
        fixed_bounds=((-0.75, -0.75, -0.75), (0.75, 0.75, 0.75))
    )
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")

def simplify_caption(c: str) -> str:
    c = c.strip().lower()

    # remove final period
    if c.endswith("."):
        c = c[:-1]

    # shorten very long captions, but keep useful shape cues
    replacements = [
        ("depicted in", ""),
        ("shown in", ""),
        ("featuring", "with"),
        ("characterized by", "with"),
        ("consisting of", "with"),
    ]
    for a, b in replacements:
        c = c.replace(a, b)

    # keep only the first two major clauses
    parts = [p.strip() for p in c.split(",") if p.strip()]
    if len(parts) >= 2:
        c = ", ".join(parts[:2])
    else:
        c = parts[0] if parts else c

    # trim overlong "with ..." chains
    if " with " in c:
        left, right = c.split(" with ", 1)
        right_parts = [p.strip() for p in right.split(" and ") if p.strip()]
        right = " and ".join(right_parts[:2])
        c = f"{left} with {right}"

    return c

# -------------------------
# Load Point-E
# -------------------------
base_name = "base40M-textvec"
upsampler_name = "upsample"

base_model = model_from_config(MODEL_CONFIGS[base_name], device)
base_model.eval()
base_model.load_state_dict(load_checkpoint(base_name, device))

upsampler_model = model_from_config(MODEL_CONFIGS[upsampler_name], device)
upsampler_model.eval()
upsampler_model.load_state_dict(load_checkpoint(upsampler_name, device))

base_diffusion = diffusion_from_config(DIFFUSION_CONFIGS[base_name])
upsampler_diffusion = diffusion_from_config(DIFFUSION_CONFIGS[upsampler_name])

sampler = PointCloudSampler(
    device=device,
    models=[base_model, upsampler_model],
    diffusions=[base_diffusion, upsampler_diffusion],
    num_points=[1024, 4096 - 1024],
    aux_channels=["R", "G", "B"],
    guidance_scale=[3.0, 0.0],
    model_kwargs_key_filter=("texts", ""),
)

# -------------------------
# Load captions
# -------------------------
rows = []
with open(INPUT_JSONL, "r", encoding="utf-8") as f:
    for line in f:
        rows.append(json.loads(line))

# -------------------------
# Run best-of-3
# -------------------------
with open(META_JSONL, "w", encoding="utf-8") as log_f:
    for row in tqdm(rows, desc="Best-of-3 Point-E"):
        uid = row["id"]
        caption = simplify_caption(row["caption"])

        best_score = -1e9
        best_data = None
        candidate_scores = []

        for sample_idx in range(K):
            samples = None
            for x in sampler.sample_batch_progressive(
                batch_size=1,
                model_kwargs=dict(texts=[caption])
            ):
                samples = x

            pc = sampler.output_to_point_clouds(samples)[0]

            coords = normalize_points(pc.coords)
            r = pc.channels["R"]
            g = pc.channels["G"]
            b = pc.channels["B"]

            pc_norm = pointcloud_from_np_arrays(coords, r, g, b)
            img = render_pointcloud_to_pil(pc_norm)
            score = clip_score_image_text(img, caption)

            candidate_scores.append(score)

            if score > best_score:
                best_score = score
                best_data = {
                    "coords": coords,
                    "R": r,
                    "G": g,
                    "B": b,
                    "sample_idx": sample_idx,
                }

        out_path = os.path.join(OUTPUT_DIR, f"{uid}.npz")
        np.savez(
            out_path,
            coords=best_data["coords"],
            R=best_data["R"],
            G=best_data["G"],
            B=best_data["B"],
        )

        log_row = {
            "id": uid,
            "caption": caption,
            "chosen_sample_idx": int(best_data["sample_idx"]),
            "best_clip_score": float(best_score),
            "all_candidate_scores": [float(x) for x in candidate_scores]
        }
        log_f.write(json.dumps(log_row, ensure_ascii=False) + "\n")

print(f"Saved best-of-{K} outputs to {OUTPUT_DIR}")
print(f"Saved selection log to {META_JSONL}")