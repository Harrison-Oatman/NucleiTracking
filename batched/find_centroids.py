import tifffile
import numpy as np
from skimage.measure import regionprops_table
from pathlib import Path
from tqdm import tqdm
import re
import pandas as pd
from scipy.ndimage import distance_transform_edt

def find_centroids_3d(masks, locs):

    props = regionprops_table(masks, locs, properties=("centroid", "intensity_mean", "area"))
    props = pd.DataFrame(props)

    mapper = {
        "centroid-0": "uv_z",
        "centroid-1": "uv_v",
        "centroid-2": "uv_u",
        "intensity_mean-0": "px_z",
        "intensity_mean-1": "px_y",
        "intensity_mean-2": "px_x",
        "intensity_mean-3": "uv_distance_from_edge",
    }

    props = props.rename(columns=mapper)

    return props


def find_centroids_2d(masks, locs, vals, argv):

    centroids = []

    dis = distance_transform_edt(1 - np.isnan(locs[0, ..., 0]))

    for t, (maskslice, locslice, valslice, argslice) in tqdm(enumerate(zip(masks, locs, vals, argv))):

        intensity_img = np.concatenate([locslice,
                                        np.expand_dims(dis, -1),
                                        np.expand_dims(valslice, -1),
                                        np.expand_dims(argslice, -1)], axis=-1)

        props = regionprops_table(maskslice, intensity_img, properties=("centroid", "intensity_mean", "area"))
        props = pd.DataFrame(props)

        mapper = {
            "centroid-0": "uv_v",
            "centroid-1": "uv_u",
            "intensity_mean-0": "px_z",
            "intensity_mean-1": "px_y",
            "intensity_mean-2": "px_x",
            "intensity_mean-3": "uv_distance_from_edge",
            "intensity_mean-4": "intensity_mean",
            "intensity_mean-5": "uv_z",
        }

        props = props.rename(columns=mapper)
        props["timepoint"] = t

        centroids.append(props)

    return pd.concat(centroids, ignore_index=True)


def main():
    args = process_cli()

    base = Path(args.base)

    masks_path = base / "cellpose_output"

    """
    2D centroids
    """
    #
    # masks_files = list(masks_path.glob("*.tif"))
    #
    # pattern_2d = re.compile(r"(?P<mesh_name>.+)_all_vals_cp_masks")
    # pattern_3d = re.compile(r"(?P<mesh_name>.+)_recon_fused_tp_(?P<timepoint>\d+)_ch_0_unwrap_cp_masks")
    #
    # all_files_2d = []
    # all_files_3d = []
    #
    # mesh_names = set()
    #
    # for file in masks_files:
    #     match = pattern_2d.match(file.stem)
    #
    #     if match:
    #         mesh_name = match.group("mesh_name")
    #         mesh_names.add(mesh_name)
    #         all_files_2d.append((mesh_name, file))
    #
    #     match = pattern_3d.match(file.stem)
    #
    #     if match:
    #         mesh_name = match.group("mesh_name")
    #         timepoint = match.group("timepoint")
    #
    #         mesh_names.add(mesh_name)
    #         all_files_3d.append((mesh_name, timepoint, file))
    #
    # all_props = []
    #
    # for mesh, mask_file in all_files_2d:
    #
    #     masks = tifffile.imread(mask_file)
    #     locs = tifffile.imread(base / f"{mesh}_all_locs.tif")
    #     vals = tifffile.imread(base / f"{mesh}_all_vals.tif")
    #     args = tifffile.imread(base / f"{mesh}_all_vals_max_project.tif")
    #
    #     props = find_centroids_2d(masks, locs, vals, args)
    #     props["mesh_name"] = mesh
    #     all_props.append(props)
    #
    # if all_props:
    #     all_props_df = pd.concat(all_props, ignore_index=True)
    #     output_file = base / "centroids_2d.csv"
    #     all_props_df.to_csv(output_file, index=False)
    #     print(f"2D centroids saved to {output_file}")
    #

    """
    2D centroids
    """

    #
    # locs = {}
    #
    # for mesh_name in mesh_names:
    #     locs_file = base / mesh_name / "locs" / f"{mesh_name}_full_locs.tif"
    #     if locs_file.exists():
    #         full_locs = tifffile.imread(locs_file)
    #         dis = distance_transform_edt(1 - np.isnan(full_locs[..., 0]))
    #
    #         full_locs = np.concatenate([full_locs, np.expand_dims(dis, -1)], axis=-1)
    #
    #         locs[mesh_name] = full_locs
    #
    #     else:
    #         print(f"Locs file not found for {mesh_name}: {locs_file}")
    #
    # all_props = []
    #
    # for mesh_name, timepoint, masks_file in tqdm(all_files_3d):
    #     if mesh_name not in locs:
    #         print(f"Skipping {mesh_name} as locs file is missing.")
    #         continue
    #
    #     masks = tifffile.imread(masks_file)
    #     locs_data = locs[mesh_name]
    #
    #     props = find_centroids_3d(masks, locs_data)
    #
    #     props["mesh_name"] = mesh_name
    #     props["timepoint"] = timepoint
    #
    #     all_props.append(props)
    #
    # if all_props:
    #     all_props_df = pd.concat(all_props, ignore_index=True)
    #     output_file = base / "centroids_3d.csv"
    #     all_props_df.to_csv(output_file, index=False)
    #     print(f"Centroids saved to {output_file}")

    # alternative approach, look for 2d stack as slices in subfolder
    meshes = set()

    locs_pattern = re.compile(r"(?P<mesh_name>.+)_full_locs.tif")

    for locs_file in base.glob("*full_locs.tif"):

        match = locs_pattern.match(locs_file.stem)
        if match:
            mesh_name = match.group("mesh_name")
            meshes.add(mesh_name)

    all_props = []

    for mesh_name in meshes:

        cellpose_stack_path = base / mesh_name / "cellpose_stack" / "cellpose"
        if not cellpose_stack_path.exists():
            print(f"Cellpose stack path does not exist for {mesh_name}: {cellpose_stack_path}")
            continue

        masks_files = list(cellpose_stack_path.glob("*.tif"))

        if not masks_files:
            print(f"No masks files found for {mesh_name} in {cellpose_stack_path}")
            continue

        masks_files.sort()  # Ensure files are sorted by timepoint
        all_mask_slices = []

        for masks_file in masks_files:
            masks = tifffile.imread(masks_file)
            all_mask_slices.append(masks)

        if not all_mask_slices:
            print(f"No valid masks found for {mesh_name} in {cellpose_stack_path}")
            continue

        masks = np.stack(all_mask_slices, axis=0)  # Shape: (timepoints, height, width)

        locs = tifffile.imread(base / f"{mesh_name}_all_locs.tif")
        vals = tifffile.imread(base / f"{mesh_name}_all_vals.tif")
        args = tifffile.imread(base / f"{mesh_name}_all_vals_max_project.tif")

        props = find_centroids_2d(masks, locs, vals, args)
        props["mesh_name"] = mesh_name
        all_props.append(props)

    if all_props:
        all_props_df = pd.concat(all_props, ignore_index=True)
        output_file = base / "centroids_2d_from_stack.csv"
        all_props_df.to_csv(output_file, index=False)
        print(f"Centroids from stack saved to {output_file}")




def process_cli():
    import argparse

    parser = argparse.ArgumentParser(description="Find centroids in masks and locs.")
    parser.add_argument("--base", type=str, required=True, help="Base uv_unwrap directory.")

    args = parser.parse_args()

    return args

if __name__ == "__main__":
    main()
