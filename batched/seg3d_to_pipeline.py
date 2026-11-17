"""Feed 3D-segmentation centroids (segment_embryo.py --merge output) into the existing pipeline's
LAP tracking + division mapping + export, in a new tracking_<param_set_name><suffix> folder.

Writes:
  <dataset>/tracking_<name><suffix>/new_centroids.csv   in the run_merge_centroids schema
  <dataset>/tracking_<name><suffix>/pipeline_config.yml copy of the base config with
                                                        param_set_name = <name><suffix>,
                                                        merge_3d_tracking disabled
Then run the post phase (run_merge_centroids skips because new_centroids.csv exists):
  uv run python -m nucleitracking.run_pipeline -c <that yml> --phase local_post

Column mapping (same convention run_merge_3d_tracking uses for 3D detections):
  frame, px_z, px_y, px_x        from the centroid table (frame = recon file index)
  px_area = pi * r^2, r = radius of the sphere with the mask's volume
  intensity_mean/std             raw intensities inside the mask
  mesh_name = "seg3d"; uv_distance_from_edge = 0; uv_u/uv_v/uv_z/uv_area/area_distortion = NaN
  -> distance_from_surface (uv_z - 12) is NaN in the export; nothing else in tracking,
     division mapping, cylindrical coords or the export needs the UV columns.

Usage:
  python seg3d_to_pipeline.py --centroids .../centroids_3d_20250131.csv \
      --dataset .../final_processing/20250131 --base-config .../tracking_20250131/pipeline_config.yml
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


def main():
    a = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    a.add_argument("--centroids", type=Path, required=True)
    a.add_argument("--dataset", type=Path, required=True, help="dataset dir (holds tracking_<name>/)")
    a.add_argument("--base-config", type=Path, required=True)
    a.add_argument("--suffix", default="_seg3d")
    a.add_argument("--min-volume", type=int, default=30)
    a.add_argument("--frames", nargs=2, type=int, default=None, help="keep frames lo..hi (inclusive)")
    args = a.parse_args()

    cfg = yaml.safe_load(args.base_config.read_text())
    name = cfg["param_set_name"] + args.suffix
    cfg["param_set_name"] = name
    cfg["dataset"] = str(args.dataset)
    cfg.setdefault("local_post", {}).setdefault("merge_3d_tracking", {})["enabled"] = False
    out_dir = args.dataset / f"tracking_{name}"
    out_dir.mkdir(parents=True, exist_ok=True)

    c = pd.read_csv(args.centroids)
    c = c[c.volume >= args.min_volume]
    if args.frames:
        c = c[(c.frame >= args.frames[0]) & (c.frame <= args.frames[1])]
    r = (c.volume * 3 / (4 * np.pi)) ** (1 / 3)
    nc = pd.DataFrame({
        "frame": c.frame.values,
        "px_z": c.px_z.values, "px_y": c.px_y.values, "px_x": c.px_x.values,
        "px_area": (np.pi * r**2).values,
        "intensity_mean": c.mean_intensity.values,
        "intensity_std": c.std_intensity.values if "std_intensity" in c else np.nan,
        "mesh_name": "seg3d",
        "uv_distance_from_edge": 0.0,
        "uv_v": np.nan, "uv_u": np.nan, "uv_z": np.nan, "uv_area": np.nan, "area_distortion": np.nan,
        "seg_label": c.label.values, "volume": c.volume.values,
    }).sort_values(["frame", "px_z", "px_y", "px_x"]).reset_index(drop=True)
    nc["id_prev"] = nc.index.astype(np.uint32)
    nc.to_csv(out_dir / "new_centroids.csv", index=True, index_label="ix")
    cfg_path = out_dir / "pipeline_config.yml"
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    print(f"{len(nc)} centroids, frames {nc.frame.min()}-{nc.frame.max()} -> {out_dir / 'new_centroids.csv'}")
    print(f"config -> {cfg_path}")
    print(f"next: uv run python -m nucleitracking.run_pipeline -c \"{cfg_path}\" --phase local_post")


if __name__ == "__main__":
    main()
