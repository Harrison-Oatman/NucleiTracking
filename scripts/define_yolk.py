import numpy as np
import matplotlib.pyplot as plt
import tifffile
import cv2
import time
import napari
import json
import pandas as pd
from abc import ABC
from scipy.spatial.distance import cdist
from scipy.spatial import ConvexHull, Delaunay
from scipy.interpolate import CubicSpline
from tqdm import tqdm
from pathlib import Path
from skimage.filters import difference_of_gaussians
from skimage.feature import peak_local_max

BASE_PATH = Path(r"D:\Tracking\NucleiTracking\data\interim\lightsheet")
SOURCE = 181


def main():
    img = tifffile.imread(BASE_PATH / f"2024_11_04\\recon2\\recon_fused_tp_{SOURCE}_ch_0.tif")
    print(img.shape)

    viewer = napari.Viewer()
    img_layer = viewer.add_image(img)

    v = difference_of_gaussians(img, 2, 8)

    pts = peak_local_max(v, min_distance=4, threshold_abs=25)
    vals = v[pts[:, 0], pts[:, 1], pts[:, 2]]

    # remove points at the very edge of the image
    d = 25
    all_axes = []
    for axis in (0, 1, 2):
        close = np.logical_or(pts[:, axis] < d, pts[:, axis] > img.shape[axis] - d)
        all_axes.append(close)
    pts = pts[np.sum(all_axes, axis=0) < 2]

    # remove lone points
    dists = cdist(pts, pts)
    close = dists < 50
    pts = pts[np.sum(close, axis=1) > 1]

    viewer.add_image(v)
    viewer.add_points(pts, size=10, out_of_slice_display=True)

    hull = ConvexHull(pts)
    viewer.add_shapes([np.array([hull.points[p] for p in simplex]) for simplex in hull.simplices], shape_type='polygon', edge_color='red')

    d = Delaunay(pts)
    viewer.add_points(pts, text=[str(np.min(p)) for p in d.plane_distance(pts)], size=5, face_color='red')

    napari.run()


if __name__ == "__main__":
    main()
