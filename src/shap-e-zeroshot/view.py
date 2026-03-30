import trimesh
import pyrender
from PIL import Image

count = 0
for uid, path in objects.items():
    print(f"Viewing: {path}")
    mesh = trimesh.load(path)
    mesh.show()   # opens viewer window

    count += 1
    if count == 6:
        break

count = 0
for uid, path in objects.items():
    print(f"Rendering: {path}")

    mesh = trimesh.load(path)
    scene = pyrender.Scene.from_trimesh_scene(mesh)

    viewer = pyrender.OffscreenRenderer(512, 512)
    color, _ = viewer.render(scene)

    img = Image.fromarray(color)
    img.save(f"preview_{count}.png")

    count += 1
    if count == 6:
        break