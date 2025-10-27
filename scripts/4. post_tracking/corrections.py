from pathlib import Path

import napari

from nucleitracking.utils.load_hdf5_data import load_embryo

spots_path = Path(r"D:\Tracking\NucleiTracking\data\processed\lightsheet\spots")
embryo = r"lightsheet_20250131_spots.h5"
spots_df = load_embryo(spots_path / embryo)[2]


def main():
    viewer = napari.Viewer()
    points_layer = viewer.add_points(
        spots_df[["t", "x", "y", "z"]].values,
        size=2,
        face_color="red",
        name="nuclei",
        ndim=3,
    )
    tracks_layer = viewer.add_tracks(
        spots_df[["t", "track_id", "x", "y", "z"]].values,
        name="tracks",
    )
    napari.run()


if __name__ == "__main__":
    main()
