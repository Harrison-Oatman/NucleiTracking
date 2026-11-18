"""Quick point annotation of one test frame for the next training round.

Shows the raw volume, a model's masks, the corrected spots, and candidate errors from
match_score.py (1:1 spot-mask matching, run rescore_eval_matched.py first):
  missed (red ring)    spot with no mask nearby      -> likely real miss
  extra  (cyan)        mask with no spot nearby      -> fallen-out nucleus, pole cell, or false mask
  split  (orange, off) 1 spot / 2 masks              -> usually a spot error (gray area)
  merge  (yellow, off) 2 spots / 1 mask              -> e.g. dividing pair kept together (gray area)
Two editable points layers: "ADD nucleus" (click each real nucleus that has no mask / no spot,
e.g. fallen-out nuclei, both halves of a dividing pair) and "WRONG mask" (click inside masks
that are not a nucleus). Press S in the viewer to save both to
  seg3d_benchmark/annotations/<embryo>_t<N>_<layer>.csv   (also saved when the window closes).

Usage: uv run python batched/annotate_frame.py --embryo 20250702 --n 275 [--model cpsam_v2_spots_v3]
"""

import argparse
import sys
from pathlib import Path

import napari
import numpy as np
import pandas as pd
import tifffile

sys.path.insert(0, str(Path(__file__).parent))
from build_training_set_v2 import SETS, SPOTS, vol_path

EVAL = Path(
    r"C:\Tracking\NucleiTracking\data\interim\lightsheet\seg3d_benchmark\eval_multi"
)
ANN = Path(
    r"C:\Tracking\NucleiTracking\data\interim\lightsheet\seg3d_benchmark\annotations"
)


def main():
    a = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    a.add_argument("--embryo", required=True)
    a.add_argument("--n", type=int, required=True)
    a.add_argument("--model", default="cpsam_v2_spots_v3")
    args = a.parse_args()
    e, n = args.embryo, args.n
    ANN.mkdir(parents=True, exist_ok=True)
    md = EVAL / args.model

    vol = tifffile.imread(vol_path(e, n))
    masks = tifffile.imread(md / f"{e}_t{n:03d}_masks.tif")
    s = pd.read_hdf(SPOTS / f"{e}_spots.h5", "lineages")
    s = s[(~s.is_interpolated.astype(bool)) & (s.frame == n - SETS[e]["offset"])]
    P = s[["px_z", "px_y", "px_x"]].to_numpy()
    ids = masks[tuple(np.clip(np.round(P).astype(int), 0, np.array(masks.shape) - 1).T)]
    shared = pd.Series(ids[ids > 0]).value_counts()
    shared_pts = P[np.isin(ids, shared[shared > 1].index)]

    v = napari.Viewer(title=f"annotate {e} frame {n} ({args.model}) - press S to save")
    lo, hi = np.percentile(vol[::2, ::2, ::2], [1, 99.95])
    v.add_image(vol, name="raw", contrast_limits=(float(lo), float(hi)))
    v.add_labels(masks, name=f"{args.model} masks", opacity=0.35)
    v.add_points(
        P,
        name="spots (corrected)",
        size=3,
        face_color="#bbbbbb",
        border_width=0,
        opacity=0.6,
    )
    cs, cmk = md / f"{e}_t{n:03d}_cat_spots.csv", md / f"{e}_t{n:03d}_cat_masks.csv"
    if cs.exists() and cmk.exists():
        # categories from match_score.py (1:1 matching); gray areas hidden by default
        sp, mk = pd.read_csv(cs), pd.read_csv(cmk)
        for cat, src, col, ring, vis in (
            ("missed", sp, "red", True, True),
            ("extra", mk, "cyan", False, True),
            ("split", mk, "orange", False, False),
            ("merge", sp, "yellow", True, False),
        ):
            d = src[src.category == cat]
            xyz = d[["px_z", "px_y", "px_x"]] if "px_z" in d else d[["z", "y", "x"]]
            desc = {
                "missed": "spot, no mask nearby",
                "extra": "mask, no spot nearby",
                "split": "1 spot / 2 masks",
                "merge": "2 spots / 1 mask",
            }[cat]
            kw = (
                {
                    "face_color": "transparent",
                    "border_color": col,
                    "border_width": 0.2,
                    "border_width_is_relative": True,
                }
                if ring
                else {"face_color": col, "border_width": 0}
            )
            lay = v.add_points(
                xyz.to_numpy(),
                name=f"{cat}: {desc} ({len(d)})",
                size=10 if ring else 7,
                visible=vis,
                **kw,
            )
            if "label" in d:
                lay.features = d[["label"]].reset_index(drop=True)
    else:
        v.add_points(
            shared_pts,
            name=f"spots sharing a mask ({len(shared_pts)})",
            size=7,
            face_color="yellow",
            border_width=0,
        )
    add = v.add_points(
        np.zeros((0, 3)),
        name="ADD nucleus",
        size=8,
        face_color="lime",
        border_width=0,
        ndim=3,
    )
    bad = v.add_points(
        np.zeros((0, 3)),
        name="WRONG mask",
        size=8,
        face_color="magenta",
        border_width=0,
        ndim=3,
    )
    add.mode = "add"

    def save(*_):
        for lay, tag in ((add, "add"), (bad, "wrong")):
            out = ANN / f"{e}_t{n:03d}_{tag}.csv"
            pd.DataFrame(lay.data, columns=["z", "y", "x"]).to_csv(out, index=False)
        print(f"saved {len(add.data)} add / {len(bad.data)} wrong -> {ANN}", flush=True)

    v.bind_key("s", save, overwrite=True)
    v.window._qt_window.destroyed.connect(save)
    napari.run()
    save()


if __name__ == "__main__":
    main()
