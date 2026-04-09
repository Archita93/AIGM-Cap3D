import os
import json
import torch
from tqdm import tqdm

from point_e.diffusion.configs import DIFFUSION_CONFIGS, diffusion_from_config
from point_e.diffusion.sampler import PointCloudSampler
from point_e.models.download import load_checkpoint
from point_e.models.configs import MODEL_CONFIGS, model_from_config

INPUT_JSONL = r"../data/splits/cap3d_test_100.jsonl"
OUTPUT_DIR = r"../outputs/cap3d_zero_shot/point_e_test_100"

os.makedirs(OUTPUT_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)

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

rows = []
with open(INPUT_JSONL, "r", encoding="utf-8") as f:
    for line in f:
        rows.append(json.loads(line))

for row in tqdm(rows):
    uid = row["id"]
    caption = row["caption"]

    samples = None
    for x in sampler.sample_batch_progressive(
        batch_size=1,
        model_kwargs=dict(texts=[caption])
    ):
        samples = x

    pc = sampler.output_to_point_clouds(samples)[0]
    out_path = os.path.join(OUTPUT_DIR, f"{uid}.npz")
    pc.save(out_path)

print("Saved", len(rows), "point clouds to", OUTPUT_DIR)