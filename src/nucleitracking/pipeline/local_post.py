import re
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
from scipy.ndimage import distance_transform_edt
from scipy.spatial.distance import cdist
from skimage.measure import regionprops_table
from tqdm import tqdm

from nucleitracking.pipeline.config import PipelineConfig


def find_centroids_2d(masks, locs, vals, area, argv):
    centroids = []
    dis = distance_transform_edt(1 - np.isnan(locs[0, ..., 0]))

    for t, (maskslice, locslice, valslice, argslice) in tqdm(
        enumerate(zip(masks, locs, vals, argv, strict=False)),
        total=len(masks),
        desc="  Extracting 2D Centroids",
    ):
        intensity_img = np.concatenate(
            [
                locslice,
                np.expand_dims(dis, -1),
                valslice,
                np.expand_dims(argslice, -1),
                np.expand_dims(area, -1),
            ],
            axis=-1,
        )

        props = regionprops_table(
            maskslice, intensity_img, properties=("centroid", "intensity_mean", "area")
        )
        props = pd.DataFrame(props)
        if len(props) == 0:
            continue

        mapper = {
            "centroid-0": "uv_v",
            "centroid-1": "uv_u",
            "area": "uv_area",
            "intensity_mean-0": "px_z",
            "intensity_mean-1": "px_y",
            "intensity_mean-2": "px_x",
            "intensity_mean-3": "uv_distance_from_edge",
            "intensity_mean-4": "intensity_mean",
            "intensity_mean-5": "uv_z",
            "intensity_mean-6": "area_distortion",
        }

        props = props.rename(columns=mapper)
        props["timepoint"] = t
        props["px_area"] = props["uv_area"] * props["area_distortion"]
        centroids.append(props)

    return pd.concat(centroids, ignore_index=True)


def merge_centroids(current_centroids, centroids_to_add, max_distance=25):
    """
    Merges centroids by rejecting overlaps, keeping ones furthest from the edge.
    """
    axes = ["px_z", "px_y", "px_x"]
    merged_centroids = current_centroids.copy()
    add_to_drop = set()
    merged_to_drop = set()

    for timepoint in set(current_centroids["timepoint"]).intersection(
        set(centroids_to_add["timepoint"])
    ):
        current = current_centroids[current_centroids["timepoint"] == timepoint]
        to_add = centroids_to_add[centroids_to_add["timepoint"] == timepoint]
        if len(current) == 0 or len(to_add) == 0:
            continue

        x_current = current[axes].values
        x_add = to_add[axes].values

        distances = cdist(x_current, x_add)
        close_pairs = np.argwhere(distances < max_distance)

        for i, j in close_pairs:
            # keep the centroids farthest from the edge
            if (
                to_add.iloc[j]["uv_distance_from_edge"]
                < current.iloc[i]["uv_distance_from_edge"]
            ):
                add_to_drop.add(to_add.index[j])
            else:
                merged_to_drop.add(current.index[i])

    out = pd.concat(
        [merged_centroids.drop(merged_to_drop), centroids_to_add.drop(add_to_drop)],
        ignore_index=True,
    )
    return out


def run_reconstruct_3d(dataset: Path, config: PipelineConfig):
    """
    Reads the Cellpose segmentations, computes centroids, projects back to 3D.
    Heavily adapted from batched/find_centroids.py and reconstruct_2.ipynb
    """
    print(f"[{dataset.name}] Running 3D Reconstruction...")

    out_dir = dataset / f"tracking_{config.param_set_name}"
    uv_unwrap_dir = out_dir / "uv_unwrap"

    if not uv_unwrap_dir.exists():
        print(f"  [Error] uv_unwrap dir missing at {uv_unwrap_dir}")
        return

    meshes = set()
    locs_pattern = re.compile(r"(?P<mesh_name>.+)_all_locs")

    for locs_file in uv_unwrap_dir.glob("*.tif"):
        match = locs_pattern.match(locs_file.stem)
        if match:
            meshes.add(match.group("mesh_name"))

    all_props = []

    # Step A: Find Centroids for each Mesh
    for mesh_name in meshes:
        print(f"  Generating properties for mesh: {mesh_name}")
        cellpose_stack_path = uv_unwrap_dir / mesh_name / "cellpose_stack" / "cellpose"
        if not cellpose_stack_path.exists():
            continue

        masks_files = list(cellpose_stack_path.glob("*_masks.tif"))
        if not masks_files:
            continue

        masks_files.sort()
        masks = np.stack([tifffile.imread(m) for m in masks_files], axis=0)

        locs = tifffile.imread(uv_unwrap_dir / f"{mesh_name}_all_locs.tif")
        vals = tifffile.imread(uv_unwrap_dir / f"{mesh_name}_all_rawvals.tif")
        args = tifffile.imread(uv_unwrap_dir / f"{mesh_name}_all_vals_max_project.tif")
        area = tifffile.imread(uv_unwrap_dir / f"{mesh_name}_area_distortion.tif")

        props = find_centroids_2d(masks, locs, vals, area, args)
        props["mesh_name"] = mesh_name
        all_props.append(props)

    if not all_props:
        print("  [Error] No centroids found.")
        return

    centroids = pd.concat(all_props, ignore_index=True)
    centroids["px_area"] = centroids["px_area"].abs()

    # Filter suspiciously large areas
    centroids = centroids[centroids["px_area"] < 300]

    # Step B: Merge Centroids
    new_centroids = None
    for mesh_name in meshes:
        print(f"  Merging {mesh_name} into global cloud")
        to_add = centroids[centroids["mesh_name"] == mesh_name]

        if new_centroids is None:
            new_centroids = to_add
            continue

        new_centroids = merge_centroids(
            new_centroids,
            to_add,
            max_distance=config.local_post.reconstruct_3d.max_distance,
        )

    new_centroids = new_centroids.dropna()
    new_centroids = new_centroids.reset_index(drop=True)
    new_centroids["id_prev"] = new_centroids.index.astype(np.uint32)

    out_file = out_dir / "new_centroids.csv"
    new_centroids.to_csv(out_file, index=False)
    print(f"  Saved 3D reconstructed centroids to {out_file}")
