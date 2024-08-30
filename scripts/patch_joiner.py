import numpy as np
import matplotlib.pyplot as plt
import tifffile
import cv2
import time
import napari
import json
from abc import ABC
from scipy.spatial.distance import cdist
from scipy.spatial import ConvexHull, Delaunay
from scipy.interpolate import CubicSpline
from tqdm import tqdm
from pathlib import Path
from collections import defaultdict


BASE_PATH = Path(r"D:\Tracking\NucleiTracking\data\interim\lightsheet")


def test_point_in_bounds(coords, bounds):
    return all([0 <= c <= b for c, b in zip(coords, bounds)])


# def merge_overlapping_segmentations(masks, patches):
#     ix = np.nonzero(masks[0])
#     a = masks[0][ix]


def main():
    img = tifffile.imread(BASE_PATH / r"raw\Recon_fused_tp_240_ch_0_normalized.tif")
    print(img.shape)

    viewer = napari.Viewer()
    img_layer = viewer.add_image(img)
    label_layer = viewer.add_labels(np.zeros_like(img, dtype=np.uint16))

    with open(BASE_PATH / "patch_test_out/patches.json", "r") as f:
        patches = json.load(f)

    for patch in patches:
        file = BASE_PATH / f"patch_test_out/patch_{patch}_cyto3masks.tif"
        mask = tifffile.imread(file)[0]
        c = [[int(val) for val in axis] for axis in patches[patch]]
        label_layer.data[c[0][0]:c[0][1], c[1][0]:c[1][1], c[2][0]:c[2][1]] += mask

        ix = np.nonzero(mask)
        a = mask[ix]
        ix = np.stack(ix)
        print(ix.shape)


    label_layer.refresh()

    napari.run()


if __name__ == "__main__":
    main()