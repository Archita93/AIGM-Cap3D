import shutil
import subprocess
import json
from pathlib import Path

from config import (
    FANTASIA3D_MODEL_DIR
)


def generate_mesh(prompt, uid, out_dir,
                  num_steps=500):
    
    exp_dir = out_dir / "tmp" / uid
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    config_path = exp_dir / "config.json"
    config = {
        "text": prompt,
        "iter": num_steps
    }
    
    with open(config_path, "w") as f:
        json.dump(config, f)
        
    subprocess.run([
        sys.executable, "train.py",
        "--config", str(config_path)
    ], cwd=FANTASIA3D_MODEL_DIR, check=True)
    
    obj_files = sorted(
        Path(FANTASIA3D_MODEL_DIR).rglob("dmtet_mesh/*.obj"),
        key=lambda x: x.stat().st_mtime,
        reverse=True
    )

    if not obj_files:
        raise RuntimeError("No mesh generated")
    
    final_path = out_dir / f"{uid}.obj"
    shutil.copy(obj_files[0], final_path)
    
    shutil.rmtree(exp_dir)
    
    return final_path