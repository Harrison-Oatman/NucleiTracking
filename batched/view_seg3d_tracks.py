"""Open one napari window per embryo with the seg3d tracking as 4D points (t = pipeline frame)
coloured by track_id (glasbey), sized by nucleus radius, shown in 3D.

Usage: uv run python batched/view_seg3d_tracks.py [--embryos 20250131 ...] [--old]
  --old shows the current (non-seg3d) tracking instead, for comparison.
"""

import argparse
from pathlib import Path

import colorcet as cc
import napari
import numpy as np
import pandas as pd

F = Path(r"C:\Tracking\NucleiTracking\data\interim\lightsheet\final_processing")
EMB = {
    "20250131": "20250131",
    "20250625": "20250625_motion",
    "20250702": "20250702",
    "20250704": "20250704_motion",
}


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--embryos", nargs="+", default=list(EMB))
    a.add_argument(
        "--old", action="store_true", help="current tracking (same as --suffix '')"
    )
    a.add_argument(
        "--suffix", default="_seg3d", help="_seg3d, _merged, or '' for current"
    )
    args = a.parse_args()
    suffix = "" if args.old else args.suffix
    pal = np.array(
        [[int(c[i : i + 2], 16) / 255 for i in (1, 3, 5)] + [1.0] for c in cc.glasbey]
    )
    viewers = []
    for e in args.embryos:
        n = EMB[e]
        name = f"{n}{suffix}"
        d = pd.read_hdf(F / e / f"tracking_{name}" / f"{name}_spots.h5", "lineages")
        v = napari.Viewer(
            title=f"{e} tracks ({suffix.strip('_') or 'current'}) - {d.track_id.nunique()} tracks"
        )
        v.add_points(
            d[["frame", "px_z", "px_y", "px_x"]].to_numpy(),
            name=f"{e} tracks",
            size=np.clip(2 * d["radius"].to_numpy() / 0.58275, 3, 20)
            if "radius" in d
            else 8,
            face_color=pal[d.track_id.to_numpy() % len(pal)],
            border_width=0,
            features=d[["track_id", "tracklet_id", "cycle", "AP"]],
            shading="spherical",
        )
        if suffix == "_merged" and "mesh_name" in d:
            s = d[d.mesh_name == "seg3d"]
            v.add_points(
                s[["frame", "px_z", "px_y", "px_x"]].to_numpy(),
                name=f"{e} added by 3D seg (rings)",
                size=14,
                face_color="transparent",
                border_color="white",
                border_width=0.15,
                border_width_is_relative=True,
            )
        v.dims.axis_labels = ("frame", "z", "y", "x")
        v.dims.ndisplay = 3
        v.dims.set_point(0, int(d.frame.min()))
        viewers.append(v)
        print(
            f"{e}: {len(d)} spots, {d.track_id.nunique()} tracks, frames {d.frame.min()}-{d.frame.max()}",
            flush=True,
        )
    napari.run()


if __name__ == "__main__":
    main()
