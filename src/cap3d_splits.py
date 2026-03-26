import pandas as pd
import json
import random
from pathlib import Path

from config import (
    RAW_DATA_DIR,
    PROCESSED_DIR,
    RANDOM_SEED,
    TRAIN_RATIO,
    VAL_RATIO
)


def load_dataset(file_path: str) -> pd.DataFrame:
    """
    Load dataset into pandas DataFrame
    """
    try:
        df = pd.read_csv(file_path, header=None)
    except Exception as e:
        print(f"[ERROR] Failed to load dataset: {e}")
        raise

    return df

def extract_data(df: pd.DataFrame) -> list:
    """
    Extract (uid, caption) pairs
    """
    data = [
        {"uid": row[0], "caption": row[1]}
        for row in df.itertuples(index=False)
    ]
    return data

def create_split(data, train_ratio=TRAIN_RATIO, val_ratio=VAL_RATIO, seed=RANDOM_SEED):
    random.seed(seed)
    
    # Remove duplicates based on uid
    unique_data = {item["uid"]: item for item in data}
    data_list = list(unique_data.values())
    random.shuffle(data_list)

    n = len(data_list)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)

    return {
        "train": data_list[:train_end],
        "valid": data_list[train_end:val_end],
        "test": data_list[val_end:],
        "meta": {
            "total_samples": n,
            "train_count": train_end,
            "val_count": val_end - train_end,
            "test_count": n - val_end,
            "seed": seed
        }
    }

def save_json(split, path):
    with open(path, "w") as f:
        json.dump(split, f, indent=2)

def main():
    print("[INFO] Loading dataset...")
    df = load_dataset(RAW_DATA_DIR / "Cap3D_automated_Objaverse_full.csv")
    print(f"[INFO] Dataset size: {len(df)}")

    data = extract_data(df)
    split = create_split(data)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    save_json(split, PROCESSED_DIR / "cap3d_split.json")
    print("[✓] Done! Split saved.")

if __name__ == "__main__":
    main()