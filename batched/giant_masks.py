"""Count very large masks per model x tp: volume > --factor x the median pseudo-mask volume of
that frame (build_training_set.py label_volumes/tp_XXX_pseudo.tif). Also reports the largest
mask and the share of all mask voxels held by giant masks.

Usage: uv run python batched/giant_masks.py --tps 96 148 220 307 [--models cpsam_v2 ...]
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile

OUT = Path(r"C:\Tracking\NucleiTracking\data\interim\lightsheet\seg3d_benchmark\20250131")
PSEUDO = Path(r"C:\Tracking\NucleiTracking\data\interim\lightsheet\seg3d_benchmark\training\spots_v1\label_volumes")


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--tps", nargs="+", type=int, required=True)
    a.add_argument("--models", nargs="+", default=None, help="default: every model dir with masks")
    a.add_argument("--factor", type=float, default=5.0)
    a.add_argument("--out", type=Path, default=OUT)
    args = a.parse_args()
    models = args.models or sorted(d.name for d in args.out.iterdir() if d.is_dir())
    rows = []
    for tp in args.tps:
        ps = np.bincount(tifffile.imread(PSEUDO / f"tp_{tp:03d}_pseudo.tif").ravel())[1:]
        med = float(np.median(ps[ps > 0]))
        for m in models:
            f = args.out / m / f"tp_{tp:03d}_masks.tif"
            if not f.exists():
                continue
            v = np.bincount(tifffile.imread(f).ravel())[1:]
            v = v[v > 0]
            g = v > args.factor * med
            rows.append(dict(model=m, tp=tp, median_pseudo_vox=med, n_masks=len(v), n_giant=int(g.sum()),
                             giant_vox_frac=round(float(v[g].sum() / v.sum()), 3) if len(v) else 0,
                             max_vol_x_median=round(float(v.max() / med), 1) if len(v) else 0))
            print(rows[-1], flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(args.out / f"giant_masks_x{args.factor:g}.csv", index=False)
    print(df.pivot(index="model", columns="tp", values="n_giant").to_string())


if __name__ == "__main__":
    main()
