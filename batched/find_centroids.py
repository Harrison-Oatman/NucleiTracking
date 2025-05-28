import tifffile
import numpy as np
from skimage.measure import regionprops_table
from pathlib import Path
from tqdm import tqdm
import re
import pandas as pd

def find_centroids(masks, locs):

    props = regionprops_table(masks, locs, properties=("centroid", "intensity_mean", "area"))

    return pd.DataFrame(props)

def main():
    args = process_cli()

    base = Path(args.base)

    masks_path = base / "cellpose_output"

    # regex get mesh name and timepoint
    # sample filename: bottom_recon_fused_tp_366_ch_0_unwrap_cp_masks.tif
    pattern = re.compile(r"(?P<mesh_name>.+)_recon_fused_tp_(?P<timepoint>\d+)_ch_0_unwrap_cp_masks\.tif")
    masks_files = list(masks_path.glob("*.tif"))

    all_files = []

    # infer mesh names from directory names
    mesh_names = set()

    for file in masks_files:
        print(file.stem)
        match = pattern.match(file.stem)
        if not match:
            continue

        mesh_name = match.group("mesh_name")
        timepoint = match.group("timepoint")

        mesh_names.add(mesh_name)
        all_files.append((mesh_name, timepoint, file))

    locs = {}

    for mesh_name in mesh_names:
        locs_file = base / mesh_name / "locs" / f"{mesh_name}_all_locs.tif"
        if locs_file.exists():
            locs[mesh_name] = tifffile.imread(locs_file)
        else:
            print(f"Locs file not found for {mesh_name}: {locs_file}")

    all_props = []

    for mesh_name, timepoint, masks_file in tqdm(all_files):
        if mesh_name not in locs:
            print(f"Skipping {mesh_name} as locs file is missing.")
            continue

        masks = tifffile.imread(masks_file)
        locs_data = locs[mesh_name]

        props = find_centroids(masks, locs_data)

        props["mesh_name"] = mesh_name
        props["timepoint"] = timepoint

        all_props.append(props)

    if all_props:
        all_props_df = pd.concat(all_props, ignore_index=True)
        output_file = base / "centroids.csv"
        all_props_df.to_csv(output_file, index=False)
        print(f"Centroids saved to {output_file}")


def process_cli():
    import argparse

    parser = argparse.ArgumentParser(description="Find centroids in masks and locs.")
    parser.add_argument("--base", type=str, required=True, help="Base uv_unwrap directory.")

    args = parser.parse_args()

    return args

if __name__ == "__main__":
    main()
