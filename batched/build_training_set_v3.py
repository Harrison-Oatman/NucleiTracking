"""Training set v3: hybrid labels = spot pseudo-masks (v2, smoothed) completed with the v1 model's
masks (the cluster run, seg3d/cpsam_v2_spots_v1/masks) where the spots are unreliable.

Per frame, starting from the v2 pseudo-mask volume:
  1. spots whose pseudo-mask is empty or tiny (< --min-vox): replace with the v1 mask that contains
     the spot, or the nearest v1 mask within --near vox (e.g. spots averaged between a fallen-out
     nucleus and its surface neighbour, condensed mitotic chromatin)
  2. v1 masks with no spot inside and no spot within --far vox (fallen-out nuclei, pole cells,
     nuclei the tracker missed) are added as labels
  v1 voxels only ever fill background; spot masks keep priority where they exist.
Frame numbering of the v1 masks: pipeline frame = recon index (20250131: tp - 60; others: N).

Outputs like build_training_set_v2.py in --out (train/, test/, label_volumes/, frames_*.csv), with
counts of replaced / added labels per frame.
Usage: uv run python batched/build_training_set_v3.py [--embryos ...]
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
from scipy import ndimage as ndi
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).parent))
from build_training_set_v2 import FP, SETS, SPOTS, vol_path

V2 = Path(
    r"C:\Tracking\NucleiTracking\data\interim\lightsheet\seg3d_benchmark\training\spots_v2\label_volumes"
)
OUT = Path(
    r"C:\Tracking\NucleiTracking\data\interim\lightsheet\seg3d_benchmark\training\spots_v3"
)


def v1_masks(e, n):
    f = n - 60 if e == "20250131" else n
    return tifffile.imread(
        FP / e / "seg3d" / "cpsam_v2_spots_v1" / "masks" / f"frame_{f:04d}_masks.tif"
    )


def hybrid(lab, m1, P, min_vox, near, far):
    lab = lab.copy()
    nxt = int(lab.max()) + 1
    sizes = np.bincount(lab.ravel(), minlength=len(P) + 1)
    p = np.clip(np.round(P).astype(int), 0, np.array(lab.shape) - 1)
    ids1 = m1[tuple(p.T)]
    labels1 = np.unique(m1)
    labels1 = labels1[labels1 > 0]
    c1 = (
        np.array(ndi.center_of_mass(m1 > 0, m1, labels1))
        if len(labels1)
        else np.zeros((0, 3))
    )
    tree1 = cKDTree(c1) if len(c1) else None
    replaced = 0
    for i in range(len(P)):
        if sizes[i + 1] >= min_vox:
            continue
        j = ids1[i]
        if j == 0 and tree1 is not None:
            d, k = tree1.query(P[i])
            j = labels1[k] if d <= near else 0
        if j:
            sel = (m1 == j) & (lab == 0)
            lab[sel] = i + 1
            replaced += 1
    # v1 masks with no spot inside and far from every spot -> new labels
    with_spot = set(ids1[ids1 > 0].tolist())
    d = cKDTree(P).query(c1)[0] if len(c1) else np.zeros(0)
    added = 0
    for j, dist in zip(labels1, d):
        if j in with_spot or dist <= far:
            continue
        sel = (m1 == j) & (lab == 0)
        if sel.sum() >= min_vox:
            lab[sel] = nxt
            nxt += 1
            added += 1
    return lab, replaced, added


def main():
    a = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    a.add_argument("--out", type=Path, default=OUT)
    a.add_argument("--embryos", nargs="+", default=list(SETS))
    a.add_argument("--planes", type=int, default=12)
    a.add_argument("--early-mult", type=int, default=2)
    a.add_argument("--test-planes", type=int, default=4)
    a.add_argument("--pad", type=int, default=12)
    a.add_argument("--min-vox", type=int, default=20)
    a.add_argument("--near", type=float, default=4.0)
    a.add_argument("--far", type=float, default=4.0)
    a.add_argument("--seed", type=int, default=0)
    args = a.parse_args()
    rng = np.random.default_rng(args.seed)
    for d in ("train", "test", "label_volumes"):
        (args.out / d).mkdir(parents=True, exist_ok=True)
    csv = args.out / f"frames_{'_'.join(args.embryos)}.csv"
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
                P = s[["px_z", "px_y", "px_x"]].to_numpy()
                base = tifffile.imread(V2 / f"{e}_t{n:03d}_pseudo.tif")
                lab, rep, add = hybrid(
                    base, v1_masks(e, n), P, args.min_vox, args.near, args.far
                )
                tifffile.imwrite(
                    args.out / "label_volumes" / f"{e}_t{n:03d}_hybrid.tif",
                    lab,
                    compression="zlib",
                )
                vol = tifffile.imread(vol_path(e, n)).astype(np.float32)
                sizes = np.bincount(lab.ravel(), minlength=len(P) + 1)[1 : len(P) + 1]
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
                        "empty_before": int(
                            (
                                np.bincount(base.ravel(), minlength=len(P) + 1)[1:]
                                < args.min_vox
                            ).sum()
                        ),
                        "empty_after": int((sizes < args.min_vox).sum()),
                        "replaced_from_v1": rep,
                        "added_from_v1": add,
                        "planes_per_axis": k,
                    }
                )
                print(rows[-1], flush=True)
                pd.DataFrame(rows).to_csv(csv, index=False)


if __name__ == "__main__":
    main()
