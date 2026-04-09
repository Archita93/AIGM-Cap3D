import os
import torch
from shap_e.diffusion.sample import sample_latents
from shap_e.diffusion.gaussian_diffusion import diffusion_from_config
from shap_e.models.download import load_model, load_config
from shap_e.util.notebooks import decode_latent_mesh

import json
import ssl
import certifi
ssl._create_default_https_context = lambda: ssl.create_default_context(cafile=certifi.where())

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

OUT_DIR = "generated_guidance"
GUIDANCE_SCALE = 17.5       # higher = more faithful to prompt, less diverse
NUM_STEPS = 64              # more steps = better quality, slower

# 64, 15
# 64, 17.5
# 128, 20 

os.makedirs(OUT_DIR, exist_ok=True)

print("Loading models...")
xm = load_model("transmitter", device=DEVICE)
model = load_model("text300M", device=DEVICE)
diffusion = diffusion_from_config(load_config("diffusion"))

# Collect all captions
with open("downloaded_objects_split.json", "r") as f:
    data = json.load(f)

data = data[:100]

print(f"Found {len(data)} captions\n")

for item in data:
    stem = item["uid"]
    caption = item["caption"]
    out_path = os.path.join(OUT_DIR, f"{stem}.obj")

    if os.path.exists(out_path):
        print(f"Skipping {stem}, already exists")
        continue

    print(f"[{stem}] {caption[:80]}")
    # use_fp16 = DEVICE.type == "cuda"
    BATCH_SIZE=1

    latents = sample_latents(
        batch_size=BATCH_SIZE,
        model=model,
        diffusion=diffusion,
        guidance_scale=GUIDANCE_SCALE,
        model_kwargs=dict(texts=[caption] * BATCH_SIZE),
        progress=True,
        clip_denoised=True,
        use_fp16=True,
        use_karras=True,
        karras_steps=NUM_STEPS,
        sigma_min=1e-3,
        sigma_max=160,
        s_churn=0,
    )

    # Save as OBJ (can also do .obj or .glb)
    out_path = os.path.join(OUT_DIR, f"{stem}.obj")
    t = decode_latent_mesh(xm, latents[0]).tri_mesh()
    with open(out_path, "w") as f:
        t.write_obj(f)

    print(f"Saved: {out_path}\n")

print("Done!")
