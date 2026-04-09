import os, json, glob, ssl, certifi
import torch, torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
from torch.utils.data import Dataset, DataLoader
from shap_e.models.download import load_model, load_config
from shap_e.models.configs import model_from_config
from shap_e.diffusion.gaussian_diffusion import diffusion_from_config
from torch.optim.lr_scheduler import LinearLR, SequentialLR

ssl._create_default_https_context = lambda: __import__('ssl').create_default_context(cafile=__import__('certifi').where())

DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 8
EPOCHS     = 15
LR         = 1e-5

with open("downloaded_objects_split.json") as f:
    data = json.load(f)

captions_df = pd.read_csv('Cap3D_automated_Objaverse_full.csv', header=None)
uid_to_caption = dict(zip(captions_df[0], captions_df[1]))
test_uids = set(item['uid'] for item in data[:100])

all_latents = [
    f.replace('.pt', '') for f in os.listdir('latents/') if f.endswith('.pt')
    and f.replace('.pt', '') not in test_uids
    and f.replace('.pt', '') in uid_to_caption
]

random.seed(42)
random.shuffle(all_latents)
split = int(0.8 * len(all_latents))
train_uids = all_latents[:split]
val_uids   = all_latents[split:]

print(f"Train: {len(train_uids)}, Val: {len(val_uids)}")

class ShapEDataset(Dataset):
    def __init__(self, uids):
        self.items = [
            (uid_to_caption[uid], f"latents/{uid}.pt")
            for uid in uids
        ]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        caption, latent_path = self.items[i]
        try:
            latent = torch.load(latent_path, map_location='cpu', weights_only=False).squeeze()

        except Exception as e:
            print(f"Skipping corrupted: {latent_path} — {e}")
            return self.__getitem__((i + 1) % len(self.items))

        return {"caption": caption, "latent": latent}

def train():
    diffusion = diffusion_from_config(load_config("diffusion"))
    os.makedirs("checkpoints_optim_3", exist_ok=True)
    latest = sorted(glob.glob("checkpoints_optim_3/epoch_*.pt"))

    if latest:
        ckpt = torch.load(latest[-1], map_location=DEVICE, weights_only=False)
        model = model_from_config(load_config("text300M"), device=DEVICE)
        model.load_state_dict(ckpt["model_state"])
        train_losses = ckpt["train_losses"]
        val_losses = ckpt["val_losses"]
        start_epoch = ckpt["epoch"] + 1
        best_val_loss = ckpt.get("best_val_loss", float('inf'))
        patience_counter = ckpt.get("patience_counter", 0)
        print(f"Resuming from epoch {start_epoch}")
    else:
        model = load_model("text300M", device=DEVICE)
        start_epoch = 0
        train_losses, val_losses = [], []
        best_val_loss = float('inf')
        patience_counter = 0            

    # Freeze everything
    for param in model.parameters():
        param.requires_grad = False

    for name, param in model.named_parameters():
        if any(x in name for x in [      
            "wrapped.backbone.resblocks.20",
            "wrapped.backbone.resblocks.21",
            "wrapped.backbone.resblocks.22",
            "wrapped.backbone.resblocks.23",
            "wrapped.ln_post",
            "wrapped.output_proj",
            "wrapped.clip_embed",    
            "wrapped.input_proj",    
            ]):
            param.requires_grad = True

    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)
    train_dataset = ShapEDataset(train_uids)
    val_dataset   = ShapEDataset(val_uids)

    print(f"Train size: {len(train_dataset)}, Val size: {len(val_dataset)}")

    remaining_epochs = EPOCHS - start_epoch   # ← only counts epochs still to run
    total_steps  = remaining_epochs * (len(train_dataset) // BATCH_SIZE)
    warmup = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=100)
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps-100)
    lr_scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[100])
    # lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              drop_last=True, num_workers=4, prefetch_factor=2)
    val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, drop_last=True,
                              num_workers=4, prefetch_factor=2)
    
    if latest:
        optimizer.load_state_dict(ckpt["optimizer_state"])
        lr_scheduler.load_state_dict(ckpt["scheduler_state"])

    # for early stopping
    patience = 7

    for epoch in range(start_epoch, EPOCHS):
        model.train()
        epoch_train_loss = []

        for i, batch in enumerate(train_loader):
            caption = batch["caption"]
            latent  = batch["latent"].to(DEVICE)
            # t = torch.randint(0, diffusion.num_timesteps, (BATCH_SIZE,), device=DEVICE)
            t = torch.full((BATCH_SIZE,), diffusion.num_timesteps // 2, device=DEVICE)

            optimizer.zero_grad()
            loss       = diffusion.training_losses(model, latent, t, model_kwargs=dict(texts=caption))
            final_loss = loss["loss"].mean()

            if torch.isnan(final_loss) or not torch.isfinite(final_loss):
                print(f"  Skipping step {i} — invalid loss")
                torch.cuda.empty_cache()
                continue

            final_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            lr_scheduler.step()
            epoch_train_loss.append(final_loss.item())
            print(f"Epoch {epoch} | Step {i} | Loss {final_loss.item():.4f}")

            if (i + 1) % 400 == 0:
                torch.save({
                    "epoch": epoch, "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "scheduler_state": lr_scheduler.state_dict(),
                    "train_losses": train_losses, "val_losses": val_losses,
                }, f"checkpoints_optim_3/epoch_{epoch}_step_{i+1}.pt")

        train_losses.append(np.mean(epoch_train_loss) if epoch_train_loss else 0)

        model.eval()
        epoch_val_loss = []
        with torch.no_grad():
            for batch in val_loader:
                caption = batch["caption"]
                latent  = batch["latent"].to(DEVICE)
                t = torch.full((BATCH_SIZE,), diffusion.num_timesteps // 2, device=DEVICE)  
                loss = diffusion.training_losses(model, latent, t, model_kwargs=dict(texts=caption))
                epoch_val_loss.append(loss["loss"].mean().item())
        val_losses.append(np.mean(epoch_val_loss) if epoch_val_loss else 0)

        if val_losses[-1] < best_val_loss:
            best_val_loss = val_losses[-1]
            patience_counter = 0
            torch.save(model.state_dict(), "best_model.pt")  # save best separately
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

        print(f"Epoch {epoch} | Train Loss {train_losses[-1]:.4f} | Val Loss {val_losses[-1]:.4f}")

        torch.save({
            "epoch": epoch, "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": lr_scheduler.state_dict(),
            "train_losses": train_losses, "val_losses": val_losses,
            "best_val_loss": best_val_loss,     
            "patience_counter": patience_counter, 
        }, f"checkpoints_optim_3/epoch_{epoch}.pt")
        
    plt.figure()
    plt.plot(train_losses, label="train")
    plt.plot(val_losses,   label="val")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig("training_curve_best.png")
    print("Training curve saved to training_curve.png")

    torch.save(model.state_dict(), "finetuned_model.pt")
    print("Model saved to finetuned_model.pt")

if __name__ == "__main__":
    train()
