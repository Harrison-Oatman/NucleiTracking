"""Merge surface-projection (UV) detections with 3D-segmentation detections, then write a
tracking_<name>_merged folder for the pipeline's post steps (run_seg3d_tracking.py).

Base = the UV pipeline's new_centroids.csv (tracking_<name>/). A seg3d centroid is added when it
is more than --dist voxels from every UV centroid of the same frame (pole cells, fallen-out
nuclei, nuclei the UV map missed). UV rows are kept as they are; added rows use the seg3d
schema of seg3d_to_pipeline.py (mesh_name "seg3d", UV columns NaN).

Usage: python merge_detections.py --embryo 20250704 --base tracking_20250704_motion [--dist 5]
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.spatial import cKDTree

F = Path(r"C:\Tracking\NucleiTracking\data\interim\lightsheet\final_processing")


def main():
    a = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    a.add_argument("--embryo", required=True)
    a.add_argument("--base", required=True, help="existing UV tracking folder name")
    a.add_argument("--seg", default="seg3d/cpsam_v2_spots_v1")
    a.add_argument("--dist", type=float, default=5.0)
    a.add_argument("--suffix", default="_merged")
    args = a.parse_args()

    ds = F / args.embryo
    uv = pd.read_csv(ds / args.base / "new_centroids.csv", index_col="ix")
    seg_tab = pd.read_csv(ds / args.seg / f"centroids_3d_{args.embryo}.csv")
    seg_tab = seg_tab.dropna(subset=["frame", "px_z", "px_y", "px_x", "volume"])
    seg_tab = seg_tab.astype({"frame": int, "volume": int})
    seg_tab = seg_tab[seg_tab.volume >= 30]

    axes = ["px_z", "px_y", "px_x"]
    keep = []
    for f, g in seg_tab.groupby("frame"):
        u = uv[uv.frame == f]
        if not len(u):
            keep.append(g)
            continue
        d = cKDTree(u[axes].to_numpy()).query(g[axes].to_numpy())[0]
        keep.append(g[d > args.dist])
    add = pd.concat(keep)
    r = (add.volume * 3 / (4 * np.pi)) ** (1 / 3)
    add_rows = pd.DataFrame(
        {
            "frame": add.frame.values,
            "px_z": add.px_z.values,
            "px_y": add.px_y.values,
            "px_x": add.px_x.values,
            "px_area": (np.pi * r**2).values,
            "intensity_mean": add.mean_intensity.values,
            "intensity_std": add.std_intensity.values
            if "std_intensity" in add
            else np.nan,
            "mesh_name": "seg3d",
            "uv_distance_from_edge": 0.0,
            "uv_v": np.nan,
            "uv_u": np.nan,
            "uv_z": np.nan,
            "uv_area": np.nan,
            "area_distortion": np.nan,
        }
    )
    merged = pd.concat(
        [uv.drop(columns=["id_prev"], errors="ignore"), add_rows], ignore_index=True
    )
    merged = merged.sort_values(["frame", "px_z", "px_y", "px_x"]).reset_index(
        drop=True
    )
    merged["id_prev"] = merged.index.astype(np.uint32)

    cfg = yaml.safe_load((ds / args.base / "pipeline_config.yml").read_text())
    name = cfg["param_set_name"] + args.suffix
    cfg["param_set_name"] = name
    cfg["dataset"] = str(ds)
    cfg.setdefault("local_post", {}).setdefault("merge_3d_tracking", {})["enabled"] = (
        False
    )
    out = ds / f"tracking_{name}"
    out.mkdir(exist_ok=True)
    merged.to_csv(out / "new_centroids.csv", index=True, index_label="ix")
    (out / "pipeline_config.yml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    per = add_rows.groupby("frame").size()
    print(
        f"{args.embryo}: UV {len(uv)} + seg3d-only {len(add_rows)} (median {per.median():.0f}/frame, "
        f"max {per.max()} at frame {per.idxmax()}) -> {out}",
        flush=True,
    )


if __name__ == "__main__":
    main()
