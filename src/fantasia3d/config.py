from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR = PROJECT_ROOT / "data" / "outputs"
RESULTS_DIR = PROJECT_ROOT / "results"

FANTASIA3D_MODEL_DIR = PROJECT_ROOT / "Fantasia3D"
FANTASIA3D_OUT_DIR = OUTPUT_DIR / "fantasia3d"
FANTASIA3D_ZEROSHOT_MESH_DIR = FANTASIA3D_OUT_DIR / "zeroshot"