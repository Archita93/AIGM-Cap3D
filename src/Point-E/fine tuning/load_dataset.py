import json

INPUT_JSON = r"../data/splits/cap3d_split.json"
OUTPUT_JSONL = r"../data/splits/cap3d_test_10.jsonl"

with open(INPUT_JSON, "r", encoding="utf-8") as f:
    data = json.load(f)

test_items = data["test"][:10]

with open(OUTPUT_JSONL, "w", encoding="utf-8") as f:
    for item in test_items:
        row = {
            "id": item["uid"],
            "caption": item["caption"]
        }
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

print(f"Saved {len(test_items)} samples to {OUTPUT_JSONL}")