import json
from pathlib import Path
from tqdm import tqdm

from generate import generate_mesh
from evaluate import evaluation

from config import (
    FANTASIA3D_ZEROSHOT_MESH_DIR,
    PROCESSED_DIR,
    RESULTS_DIR
)

FANTASIA3D_ZEROSHOT_MESH_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def run():
    print("[INFO] Loading Cap3D split...")

    with open(PROCESSED_DIR / "downloaded_objects_split.json") as f:
        split = json.load(f)

    test_data = split[:1]
    print(f"[INFO] Running on {len(test_data)} samples...")

    print("[INFO] Generating 3D meshes...")

    for item in tqdm(test_data):
        uid = item["uid"]
        prompt = item["caption"]

        out_path = FANTASIA3D_ZEROSHOT_MESH_DIR / f"{uid}.obj"

        if out_path.exists():
            continue

        mesh = generate_mesh(prompt, uid, FANTASIA3D_ZEROSHOT_MESH_DIR)
        Path(mesh).rename(out_path)

    print("[INFO] Running evaluation...")

    evaluation(
        pred_dir=FANTASIA3D_ZEROSHOT_MESH_DIR,
        dataset=test_data,
        output_path=RESULTS_DIR / "fantasia3d_zeroshot.json"
    )


if __name__ == "__main__":
    run()