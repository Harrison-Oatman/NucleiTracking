"""Score 3D instance masks (e.g. Cellpose-SAM --do_3d output) against tracked spots.

Per frame: hit = spot inside a mask that holds only that spot; merge = mask holding >1 spot;
miss = spot on background; extra = mask with no spot inside or within --tol voxels.
Spots are treated as near-complete reference detections (checked on 20250131 against
independent DoG peaks), so "extra" masks are candidates for false positives OR missed spots;
look at a few before trusting either.

Usage:
  python NucleiTracking/batched/score_masks_vs_spots.py \
      --masks <dir with frame_XXXX*_masks.tif>  --spots <pseudolabels>/labels \
      --out scores.csv
Mask files must be named frame_XXXX..._masks.tif (or _cp_masks.tif) and cover the same crop as
volumes/frame_XXXX.tif, so the crop_z/y/x columns of labels/frame_XXXX_spots.csv apply.
"""

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
from scipy import ndimage as ndi


def score(masks, P, tol=3.0):
    p = np.clip(np.round(P).astype(int), 0, np.array(masks.shape) - 1)
    ids = masks[p[:, 0], p[:, 1], p[:, 2]]
    n_per = pd.Series(ids[ids > 0]).value_counts()
    hit = np.array([i > 0 and n_per[i] == 1 for i in ids])
    merged = np.array([i > 0 and n_per[i] > 1 for i in ids])
    labels = np.unique(masks)
    labels = labels[labels > 0]
    cents = (
        np.array(ndi.center_of_mass(masks > 0, masks, labels))
        if len(labels)
        else np.zeros((0, 3))
    )
    from scipy.spatial import cKDTree

    d, _ = cKDTree(P).query(cents) if len(cents) else (np.zeros(0), None)
    no_spot = np.setdiff1d(labels, ids)
    extra = int(sum(d[np.searchsorted(labels, no_spot)] > tol)) if len(no_spot) else 0
    return {
        "n_spots": len(P),
        "n_masks": len(labels),
        "hit": int(hit.sum()),
        "merged_spots": int(merged.sum()),
        "missed": int((ids == 0).sum()),
        "extra_masks": extra,
        "recall": float(hit.mean()),
    }


def main():
    a = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    a.add_argument("--masks", required=True, type=Path)
    a.add_argument(
        "--spots",
        required=True,
        type=Path,
        help="labels/ dir from spot_pseudolabels.py",
    )
    a.add_argument("--out", required=True, type=Path)
    a.add_argument("--tol", type=float, default=3.0)
    args = a.parse_args()
    rows = []
    for mf in sorted(args.masks.glob("frame_*masks.tif")):
        fr = int(re.match(r"frame_(\d+)", mf.name).group(1))
        s = pd.read_csv(args.spots / f"frame_{fr:04d}_spots.csv")
        rows.append(
            dict(
                frame=fr,
                **score(
                    tifffile.imread(mf),
                    s[["crop_z", "crop_y", "crop_x"]].to_numpy(),
                    args.tol,
                ),
            )
        )
    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    print(df.to_string())


if __name__ == "__main__":
    main()
