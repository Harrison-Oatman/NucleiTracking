"""Training set v2: spot pseudo-masks from the corrected spots of all four WT embryos.

Changes from v1 (build_training_set.py, 20250131 only):
  - embryos 20250625 / 20250702 / 20250704 added (frames chosen where the v1 model failed,
    one interphase frame per cycle, and mid-division frames); 20250131 adds its mid-division
    frames (105, 142, 190, 196, 198, 266) to the v1 training frames
  - pseudo-masks smoothed per label: binary closing (radius 1) + 3D and per-plane hole filling,
    to remove the half-max cups/crescents the v1 model learned
Frame <-> spots: 20250131 files are named by tp (h5 frame = tp - 61); the other embryos' files
are named by pipeline frame N (h5 frame = N - 1). Corrected spots: BlastodermAnalysis/data/spots.

Outputs (--out): train/, test/ (2D planes + _masks), label_volumes/<E>_tXXX_pseudo.tif, frames.csv
Usage: uv run python batched/build_training_set_v2.py
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
from scipy import ndimage as ndi

sys.path.insert(0, str(Path(__file__).parent))
from spot_pseudolabels import check_alignment, pseudo_masks

FP = Path(r"C:\Tracking\NucleiTracking\data\interim\lightsheet\final_processing")
PREV = Path(
    r"C:\Tracking\NucleiTracking\data\interim\lightsheet\previous\2025_01_31\recon"
)
SPOTS = Path(r"C:\Tracking\BlastodermAnalysis\data\spots")
OUT = Path(
    r"C:\Tracking\NucleiTracking\data\interim\lightsheet\seg3d_benchmark\training\spots_v2"
)

SETS = {
    "20250131": {
        "offset": 61,
        "test": [96, 148, 220, 307],
        "train": [
            87,
            115,
            130,
            136,
            154,
            160,
            166,
            172,
            178,
            184,
            202,
            208,
            214,
            229,
            286,
            294,
            88,
            90,
            112,
            119,
            122,
            125,
            105,
            142,
            190,
            196,
            198,
            266,
        ],
    },
    "20250625": {
        "offset": 1,
        "test": [130, 246, 255],
        "train": [50, 63, 85, 110, 140, 160, 200, 242, 249, 252, 300],
    },
    "20250702": {
        "offset": 1,
        "test": [76, 83, 130, 275],
        "train": [62, 68, 72, 80, 86, 100, 115, 127, 133, 160, 183, 230, 271, 278, 300],
    },
    "20250704": {
        "offset": 1,
        "test": [93, 107, 287],
        "train": [80, 86, 90, 96, 101, 104, 125, 147, 170, 197, 240, 283, 310],
    },
}


def vol_path(e, n):
    p = FP / e / "recon" / f"recon_fused_tp_{n}_ch_0.tif"
    if not p.exists() and e == "20250131":
        p = PREV / f"recon_fused_tp_{n}_ch_0.tif"
    return p


def smooth_labels(lab):
    out = lab.copy()
    st = ndi.generate_binary_structure(3, 1)
    for i, sl in enumerate(ndi.find_objects(lab), 1):
        if sl is None:
            continue
        sl = tuple(slice(max(s.start - 2, 0), s.stop + 2) for s in sl)
        sub = out[sl]
        m = sub == i
        m2 = ndi.binary_closing(m, st, iterations=1)
        m2 = ndi.binary_fill_holes(m2)
        for ax in range(3):
            m2 |= np.moveaxis(
                np.stack([ndi.binary_fill_holes(p) for p in np.moveaxis(m2, ax, 0)]),
                0,
                ax,
            )
        sub[m2 & (sub == 0)] = i  # only take background, never another nucleus
    return out


def main():
    a = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    a.add_argument("--out", type=Path, default=OUT)
    a.add_argument("--planes", type=int, default=12)
    a.add_argument("--early-mult", type=int, default=2)
    a.add_argument("--test-planes", type=int, default=4)
    a.add_argument("--pad", type=int, default=12)
    a.add_argument("--embryos", nargs="+", default=list(SETS))
    a.add_argument("--seed", type=int, default=0)
    args = a.parse_args()
    rng = np.random.default_rng(args.seed)
    for d in ("train", "test", "label_volumes"):
        (args.out / d).mkdir(parents=True, exist_ok=True)
    csv = (
        args.out / f"frames_{'_'.join(args.embryos)}.csv"
    )  # one file per process (parallel runs)
    rows = pd.read_csv(csv).to_dict("records") if csv.exists() else []
    done = {(str(r["embryo"]), int(r["n"])) for r in rows}

    for e in args.embryos:
        cfg = SETS[e]
        s_all = pd.read_hdf(SPOTS / f"{e}_spots.h5", "lineages")
        s_all = s_all[~s_all.is_interpolated.astype(bool)]
        for split in ("test", "train"):
            for n in cfg[split]:
                if (e, n) in done:
                    continue
                s = s_all[s_all.frame == n - cfg["offset"]]
                if not len(s):
                    print(
                        f"WARNING {e} {n}: no spots at h5 frame {n - cfg['offset']}",
                        flush=True,
                    )
                    continue
                P = s[["px_z", "px_y", "px_x"]].to_numpy()
                vol = tifffile.imread(vol_path(e, n)).astype(np.float32)
                if (P.max(0) >= np.array(vol.shape)).any():
                    msg = f"{e} {n}: spots outside volume {vol.shape}"
                    raise SystemExit(msg)
                ratio = check_alignment(vol, P)
                p = np.round(P).astype(int)
                on = vol[tuple(p.T)].mean()
                off = vol[
                    tuple(np.clip([*p, 0, 8, 0], 0, np.array(vol.shape) - 1).T)
                ].mean()
                if on / off < 1.1:
                    print(
                        f"WARNING {e} {n}: spots barely brighter than 8 vox away ({on / off:.2f})",
                        flush=True,
                    )
                lp = args.out / "label_volumes" / f"{e}_t{n:03d}_pseudo.tif"
                if lp.exists():
                    lab = tifffile.imread(lp)
                else:
                    lo = np.maximum(np.floor(P.min(0)).astype(int) - args.pad, 0)
                    hi = np.minimum(
                        np.ceil(P.max(0)).astype(int) + args.pad + 1, vol.shape
                    )
                    lab = np.zeros(vol.shape, np.uint16)
                    crop = pseudo_masks(
                        vol[lo[0] : hi[0], lo[1] : hi[1], lo[2] : hi[2]], P - lo
                    )
                    lab[lo[0] : hi[0], lo[1] : hi[1], lo[2] : hi[2]] = smooth_labels(
                        crop
                    )
                    tifffile.imwrite(lp, lab, compression="zlib")
                sizes = np.bincount(lab.ravel(), minlength=len(P) + 1)[1:]
                cyc = int(s.cycle.mode().iloc[0])
                k = (
                    args.test_planes
                    if split == "test"
                    else args.planes * (args.early_mult if cyc <= 11 else 1)
                )
                lo = np.maximum(np.floor(P.min(0)).astype(int) - args.pad, 0)
                hi = np.minimum(np.ceil(P.max(0)).astype(int) + args.pad + 1, vol.shape)
                for ax, name in enumerate("zyx"):
                    for i in sorted(
                        rng.choice(
                            np.arange(lo[ax], hi[ax]),
                            size=min(k, hi[ax] - lo[ax]),
                            replace=False,
                        )
                    ):
                        stem = f"{e}_t{n:03d}_{name}{i:03d}"
                        tifffile.imwrite(
                            args.out / split / f"{stem}.tif", np.take(vol, i, axis=ax)
                        )
                        tifffile.imwrite(
                            args.out / split / f"{stem}_masks.tif",
                            np.take(lab, i, axis=ax),
                        )
                rows.append(
                    {
                        "embryo": e,
                        "n": n,
                        "split": split,
                        "cycle": cyc,
                        "n_spots": len(P),
                        "shape": str(vol.shape),
                        "check_ratio": round(ratio, 2),
                        "on_off": round(float(on / off), 2),
                        "empty_masks": int((sizes < 5).sum()),
                        "median_vol_vox": float(np.median(sizes)),
                        "planes_per_axis": k,
                    }
                )
                print(rows[-1], flush=True)
                pd.DataFrame(rows).to_csv(csv, index=False)


if __name__ == "__main__":
    main()
