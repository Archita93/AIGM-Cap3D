import objaverse

uid = "a3db27de00424d78a3f5a6d93b967f5d"  # your first test sample

# optional: metadata
ann = objaverse.load_annotations([uid])
print("annotation keys:", ann[uid].keys() if uid in ann else "missing")

# download GT object
objects = objaverse.load_objects(uids=[uid])
print(objects)