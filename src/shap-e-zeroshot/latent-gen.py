import torch, os, pickle

with open('shapelatents/misc/ShapELatentCode_zips/compressed_files_info.pkl', 'rb') as f:
    info = pickle.load(f)

corrupted_uids = []
for f in os.listdir('latents/'):
    if not f.endswith('.pt'):
        continue
    try:
        torch.load(f'latents/{f}', map_location='cpu')
    except Exception:
        corrupted_uids.append(f.replace('.pt', ''))
        print(f"Corrupted: {f}")

print(f"\nTotal corrupted: {len(corrupted_uids)}")

for zip_name, uid_list in info.items():
    matches = set(corrupted_uids).intersection(set(uid_list))
    if matches:
        print(f"{zip_name}: {len(matches)} corrupted files")
