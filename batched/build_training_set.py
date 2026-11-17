"""Build a 2D Cellpose training set from spot pseudo-masks (20250131, local full volumes).

Per frame: seeded-watershed pseudo-masks grown from that frame's spots (spot_pseudolabels.
pseudo_masks, half-max edge) on the FULL volume, then whole planes along z, y and x exported
as (image, masks) pairs. Planes are drawn uniformly over the embryo's extent (not only planes
dense in nuclei), so sparse regions, yolk and outside-embryo background are included; every
voxel without a pseudo-mask is background, so nuclei that lack a spot become false background
(spots are near-complete in interphase; mid-division frames are not used).

Outputs (--out):
  train/<stem>_<axis><idx>.tif + _masks.tif, test/ likewise
  label_volumes/tp_XXX_pseudo.tif       full 3D pseudo-masks (uint16, zlib)
  frames.csv                            per frame: split, n spots, alignment ratio, empty masks,
                                        median mask volume (voxels), planes per axis

Usage: uv run python batched/build_training_set.py
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile

sys.path.insert(0, str(Path(__file__).parent))
from benchmark_models_vs_spots import SPOTS, TP_OFFSET, find_volumes, load_spots  # noqa: E402
from spot_pseudolabels import check_alignment, pseudo_masks  # noqa: E402

OUT = Path(r"C:\Tracking\NucleiTracking\data\interim\lightsheet\seg3d_benchmark\training\spots_v1")
# mid-division frames (spot count rising) excluded: 142, 148(test), 190, 196, 198, 266
TRAIN_TPS = [87, 115, 130, 136, 154, 160, 166, 172, 178, 184, 202, 208, 214, 229, 286, 294]
TEST_TPS = [96, 148, 220, 307]  # quick-look frames; 93 (next to 96) is left out of both


def main():
    a = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    a.add_argument("--out", type=Path, default=OUT)
    a.add_argument("--train-tps", nargs="+", type=int, default=TRAIN_TPS)
    a.add_argument("--test-tps", nargs="+", type=int, default=TEST_TPS)
    a.add_argument("--planes", type=int, default=12, help="planes per axis per train frame")
    a.add_argument("--early-mult", type=int, default=2, help="plane multiplier for NC10-11 train frames")
    a.add_argument("--test-planes", type=int, default=4)
    a.add_argument("--pad", type=int, default=12)
    a.add_argument("--seed", type=int, default=0)
    args = a.parse_args()

    vols, _ = find_volumes()
    spots = load_spots(SPOTS)
    rng = np.random.default_rng(args.seed)
    for d in ("train", "test", "label_volumes"):
        (args.out / d).mkdir(parents=True, exist_ok=True)
    rows = []
    for split, tps in (("test", args.test_tps), ("train", args.train_tps)):
        for tp in tps:
            s = spots[spots.frame == tp - TP_OFFSET]
            P = s[["px_z", "px_y", "px_x"]].to_numpy()
            cyc = int(s.cycle.mode().iloc[0])
            vol = tifffile.imread(vols[tp]).astype(np.float32)
            ratio = check_alignment(vol, P)
            if ratio < 1.0:  # low-SNR early frames can sit below 1 even when aligned
                print(f"WARNING tp {tp}: spot/p99 = {ratio:.2f} < 1; check alignment", flush=True)
            lp = args.out / "label_volumes" / f"tp_{tp:03d}_pseudo.tif"
            if lp.exists():
                lab = tifffile.imread(lp)
            else:
                # pseudo-masks inside the spots' bbox (+pad) for speed; zero outside
                lo = np.maximum(np.floor(P.min(0)).astype(int) - args.pad, 0)
                hi = np.minimum(np.ceil(P.max(0)).astype(int) + args.pad + 1, vol.shape)
                lab = np.zeros(vol.shape, np.uint16)
                lab[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]] = pseudo_masks(
                    vol[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]], P - lo)
                tifffile.imwrite(lp, lab, compression="zlib")
            sizes = np.bincount(lab.ravel(), minlength=len(P) + 1)[1:]

            n = args.test_planes if split == "test" else args.planes * (args.early_mult if cyc <= 11 else 1)
            lo = np.maximum(np.floor(P.min(0)).astype(int) - args.pad, 0)
            hi = np.minimum(np.ceil(P.max(0)).astype(int) + args.pad + 1, vol.shape)
            for ax, name in enumerate("zyx"):
                ks = rng.choice(np.arange(lo[ax], hi[ax]), size=min(n, hi[ax] - lo[ax]), replace=False)
                for k in sorted(ks):
                    stem = f"tp{tp:03d}_{name}{k:03d}"
                    tifffile.imwrite(args.out / split / f"{stem}.tif", np.take(vol, k, axis=ax))
                    tifffile.imwrite(args.out / split / f"{stem}_masks.tif", np.take(lab, k, axis=ax))
            row = dict(split=split, tp=tp, cycle=cyc, n_spots=len(P), check_ratio=round(ratio, 2),
                       empty_masks=int((sizes < 5).sum()), median_vol_vox=float(np.median(sizes)),
                       planes_per_axis=n)
            rows.append(row)
            print(row, flush=True)
            pd.DataFrame(rows).to_csv(args.out / "frames.csv", index=False)


if __name__ == "__main__":
    main()
