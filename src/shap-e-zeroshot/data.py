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

    # Get first 100
    items = data["train"][:100]
    uids = [item["uid"] for item in items]

    # Split by type
    objaverse_uids = [uid for uid in uids if len(uid) <= 32]
    objaverse_xl_uids = [uid for uid in uids if len(uid) == 64]

    print(f"Objaverse 1.0: {len(objaverse_uids)}")
    print(f"Objaverse XL:  {len(objaverse_xl_uids)}")

    objects = {}

    if objaverse_uids:
        v1_objects = objaverse.load_objects(
            uids=objaverse_uids,
            download_processes=4
        )
        objects.update(v1_objects)

    if objaverse_xl_uids:
        annotations = oxl.get_annotations()
        subset = annotations[annotations["sha256"].isin(objaverse_xl_uids)]

        xl_objects = oxl.download_objects(
            objects=subset,
            download_processes=4
        )

        if xl_objects:
            objects.update(xl_objects)

    print(f"Total downloaded: {len(objects)}")
    print(objects)


if __name__ == "__main__":
    main()