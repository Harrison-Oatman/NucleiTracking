from pathlib import Path
from scipy.spatial import KDTree
from sklearn.decomposition import PCA

import napari
import numpy as np

from nucleitracking.utils.load_hdf5_data import load_embryo

spots_path = Path(r"D:\Tracking\NucleiTracking\data\processed\lightsheet\spots")
embryo = r"lightsheet_20250131_spots.h5"
spots_df = load_embryo(spots_path / embryo)[2]


def main():
    viewer = napari.Viewer()

    points = spots_df[["frame", "z", "y", "x"]].values
    points_center = np.mean(points[:, 1:], axis=0)
    tree_points = points * np.array([250, 1, 1, 1])
    tree = KDTree(tree_points)

    points_layer = viewer.add_points(
        points,
        size=spots_df["radius"].values * 2.1,
        face_color=[[0.5, 0.5, 0.5, 0.125] for _ in spots_df["radius"]],
        border_color="k",
        border_width=0.2,
        name="nuclei",
    )

    def center_on_nucleus(nuc_id):
        nucleus_loc = points[nuc_id]

        _dis, neighbors = tree.query(tree_points[nuc_id], 25)

        # get normal axis for camera orientation
        pca = PCA(3)
        pca.fit(points[neighbors][:, 1:])
        normal_axis = pca.components_[-1]

        print(normal_axis, points_center, nucleus_loc)

        centered_loc = nucleus_loc[1:] - points_center
        if np.linalg.norm(centered_loc + normal_axis) > np.linalg.norm(centered_loc):
            normal_axis = -normal_axis

        # set loc for camera center
        viewer.dims.set_current_step(0, nucleus_loc[0])
        viewer.camera.center = nucleus_loc

        viewer.camera.set_view_direction(normal_axis)

        points_layer.face_color[neighbors] = np.array([1.0, 0.0, 1.0, 1.0])

    @points_layer.bind_key("Enter")
    def choose_nucleus(layer):
        nuc = np.random.randint(0, len(spots_df.index))
        print(points[nuc, 0])

        center_on_nucleus(nuc)

        layer.face_color[nuc] = np.array([0.0, 1.0, 0.0, 1.0])
        layer.refresh()

    napari.run()


if __name__ == "__main__":
    main()
