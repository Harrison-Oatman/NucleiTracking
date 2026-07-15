import argparse
import json
import logging
from pathlib import Path

import natsort
import pandas as pd
import tifffile
from skimage.measure import regionprops_table
from tqdm import tqdm


def main():
    args = process_cli()
    logging.basicConfig(level=args.level)

    outpath = Path(args.output)
    assert outpath.exists(), f"output directory not found: {outpath}"

    with open(outpath / "bbox.json") as f:
        bbox = json.load(f)

    cellpose_dir = outpath / "cellpose"
    crops_dir = outpath / "crops"

    assert cellpose_dir.exists(), f"cellpose directory not found: {cellpose_dir}"
    assert crops_dir.exists(), f"crops directory not found: {crops_dir}"

    mask_files = natsort.natsorted(list(cellpose_dir.glob("*_cp_masks.tif")))
    crop_files = natsort.natsorted(list(crops_dir.glob("*.tif")))

    assert len(mask_files) > 0, "no *_cp_masks.tif files found in cellpose directory"
    assert len(mask_files) == len(crop_files), (
        f"mask count ({len(mask_files)}) != crop count ({len(crop_files)})"
    )

    logging.info(f"processing {len(mask_files)} timepoints")

    all_props = []
    for t, (mask_file, crop_file) in tqdm(
        enumerate(zip(mask_files, crop_files, strict=True))
    ):
        masks = tifffile.imread(str(mask_file))
        intensity = tifffile.imread(str(crop_file))

        if masks.max() == 0:
            logging.warning(f"timepoint {t}: no masks found in {mask_file.name}")
            continue

        props = regionprops_table(
            masks,
            intensity_image=intensity,
            properties=("centroid", "intensity_mean", "intensity_std", "area"),
        )
        df = pd.DataFrame(props)
        df = df.rename(
            columns={
                "centroid-0": "crop_z",
                "centroid-1": "crop_y",
                "centroid-2": "crop_x",
            }
        )
        df["global_z"] = df["crop_z"] + bbox["z_lo"]
        df["global_y"] = df["crop_y"] + bbox["y_lo"]
        df["global_x"] = df["crop_x"] + bbox["x_lo"]
        df["timepoint"] = t
        df["bbox_name"] = bbox["name"]
        all_props.append(df)

    if not all_props:
        logging.error("no valid timepoints with masks found")
        return

    result = pd.concat(all_props, ignore_index=True)
    out_csv = outpath / "centroids_3d.csv"
    result.to_csv(out_csv, index=False)
    logging.info(f"saved {len(result)} nuclei to {out_csv}")


def process_cli():
    parser = argparse.ArgumentParser(
        description="Run regionprops on 3D cellpose masks to produce centroid table."
    )
    parser.add_argument(
        "-o",
        "--output",
        required=True,
        help="bbox output directory (contains bbox.json, crops/, cellpose/)",
    )
    parser.add_argument("-l", "--level", default="INFO")
    return parser.parse_args()


if __name__ == "__main__":
    main()
