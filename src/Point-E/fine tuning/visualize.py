import numpy as np
from point_e.util.plotting import plot_point_cloud
from point_e.util.point_cloud import PointCloud

NPZ_PATH = r"../outputs/cap3d_finetune/point_e_bestof3_test_100/2e28e3401008416d156ba907ac02fa1b67dd082bee3ea012ceb38d99cf79ec7e.npz"

data = np.load(NPZ_PATH)

pc = PointCloud(
    coords=data["coords"],
    channels={
        "R": data["R"],
        "G": data["G"],
        "B": data["B"],
    }
)

fig = plot_point_cloud(
    pc,
    grid_size=3,
    fixed_bounds=((-0.75, -0.75, -0.75), (0.75, 0.75, 0.75))
)

import matplotlib.pyplot as plt
plt.show()