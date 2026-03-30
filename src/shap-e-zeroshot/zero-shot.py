import json
import torch
import os
import time
import argparse
from tqdm import tqdm
 
from shap_e.diffusion.sample import sample_latents
from shap_e.diffusion.gaussian_diffusion import diffusion_from_config
from shap_e.models.download import load_model, load_config
from shap_e.util.notebooks import create_pan_cameras, decode_latent_images, decode_latent_mesh

parser = argparse.ArgumentParser()
parser.add_argument("--input",      default="cap3d_split.json", help="Path to cap3d_split.json")
parser.add_argument("--output_dir", default="zero_shot_output",  help="Where to save results")
parser.add_argument("--split",      default="train",             help="JSON split to use: train/test/val")
parser.add_argument("--n",          type=int, default=100,       help="Number of captions to run")
parser.add_argument("--guidance",   type=float, default=15.0,    help="Guidance scale")
parser.add_argument("--steps",      type=int, default=64,        help="Diffusion steps")
parser.add_argument("--save_gif",   action="store_true",         help="Save rotating GIF previews")
parser.add_argument("--save_obj",   action="store_true",         help="Save .obj mesh files")
parser.add_argument("--resume",     action="store_true",         help="Skip already-completed UIDs")
args = parser.parse_args()

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"Device: {device}")

os.makedirs(args.output_dir, exist_ok=True)
log_path = os.path.join(args.output_dir, "results.jsonl")

print("Loading Shap-E models...")
xm        = load_model("transmitter", device=device)
model     = load_model("text300M",    device=device)
diffusion = diffusion_from_config(load_config("diffusion"))
print("Models loaded.\n")

with open(args.input, "r") as f:
    data = json.load(f)
 
items = data[args.split][:args.n]
print(f"Running zero-shot inference on {len(items)} captions from '{args.split}' split.\n")

done_uids = set()
if args.resume and os.path.exists(log_path):
    with open(log_path, "r") as f:
        for line in f:
            entry = json.loads(line)
            done_uids.add(entry["uid"])
    print(f"Resuming — skipping {len(done_uids)} already completed.\n")

success, failed = 0, 0
 
with open(log_path, "a") as log_file:
    for item in tqdm(items, desc="Generating"):
        uid     = item["uid"]
        caption = item["caption"]
 
        if uid in done_uids:
            continue
 
        try:
            t0 = time.time()
 
            # Generate latents from caption
            latents = sample_latents(
                batch_size=1,
                model=model,
                diffusion=diffusion,
                guidance_scale=args.guidance,
                model_kwargs=dict(texts=[caption]),
                progress=False,
                clip_denoised=True,
                use_fp16=False,
                use_karras=True,
                karras_steps=args.steps,
                sigma_min=1e-3,
                sigma_max=160,
                s_churn=0,
            )
 
            uid_short = uid[:32]  # safe filename for both v1 and XL uids
 
            # Save GIF
            if args.save_gif:
                cameras   = create_pan_cameras(size=128, device=device)
                images    = decode_latent_images(xm, latents[0], cameras, rendering_mode="nerf")
                gif_path  = os.path.join(args.output_dir, f"{uid_short}.gif")
                images[0].save(gif_path, save_all=True, append_images=images[1:], duration=100, loop=0)
 
            # Save OBJ
            if args.save_obj:
                mesh     = decode_latent_mesh(xm, latents[0]).tri_mesh()
                obj_path = os.path.join(args.output_dir, f"{uid_short}.obj")
                with open(obj_path, "w") as f:
                    mesh.write_obj(f)
 
            elapsed = time.time() - t0
            success += 1
 
            # Log result
            log_file.write(json.dumps({
                "uid":     uid,
                "caption": caption,
                "status":  "success",
                "time_s":  round(elapsed, 2),
            }) + "\n")
            log_file.flush()
 
        except Exception as e:
            failed += 1
            print(f"\n  Failed [{uid[:16]}]: {e}")
            log_file.write(json.dumps({
                "uid":     uid,
                "caption": caption,
                "status":  "failed",
                "error":   str(e),
            }) + "\n")
            log_file.flush()

print(f"\nDone. Success: {success} | Failed: {failed}")
print(f"Results logged to: {log_path}")
 


