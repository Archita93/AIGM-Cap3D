import json
import ssl
import certifi
import objaverse
import objaverse.xl as oxl

ssl._create_default_https_context = lambda: ssl.create_default_context(
    cafile=certifi.where()
)

def main():
    # Load data
    with open("../cap3d_split.json", "r") as f:
        data = json.load(f)

    items = data["train"]
    uids = [item["uid"] for item in items]

    # caption mapping
    uid_to_caption = {item["uid"]: item["caption"] for item in items}

    # slicing the objaverse - 600 images
    objaverse_uids = [uid for uid in uids if len(uid) <= 32][:600]
    
    print(f"Objaverse 1.0: {len(objaverse_uids)}")

    objects = {}
    count = len(objaverse_uids)

    if objaverse_uids:
        v1_objects = objaverse.load_objects(
            uids=objaverse_uids,
            download_processes=8
        )
        objects.update(v1_objects)


    output = []
    for uid, filepath in objects.items():
        if uid in uid_to_caption:
            output.append({"uid": uid, "caption": uid_to_caption[uid], "filepath": filepath})

    with open("downloaded_objects_split.json", "w") as f:
        json.dump(output, f, indent=2)

    print(f"Saved {len(output)} entries to downloaded_objects.json")


if __name__ == "__main__":
    main()