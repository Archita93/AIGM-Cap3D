import trimesh
import numpy as np
import matplotlib.pyplot as plt

GT_PATH = r"..\data\.objaverse\hf-objaverse-v1\glbs/000-011\a3db27de00424d78a3f5a6d93b967f5d.glb"

mesh = trimesh.load(GT_PATH, force="mesh")
points = mesh.sample(5000)

# normalize to centered unit scale
center = points.mean(axis=0)
points = points - center
scale = np.abs(points).max()
points = points / scale

fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(111, projection="3d")
ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=1)
ax.set_xlim(-0.75, 0.75)
ax.set_ylim(-0.75, 0.75)
ax.set_zlim(-0.75, 0.75)
plt.show()