"""Compare first-pass tracking on 3D-seg detections with the current tracking and the corrected spots.

Per embryo:
  detections: per-frame counts (seg3d vs current final_lineages vs corrected spots), and spatial
              agreement with the corrected spots (match within --tol voxels, per frame, Hungarian-free
              nearest-neighbour both ways) summarised per cycle
  tracks:     track / tracklet counts, divisions (spots with n_children == 2) per cycle,
              share of tracklets flagged problematic, median tracklet length (frames) per cycle
The corrected-spots frame offset is found by testing -2..2 on a mid frame (pipeline frame vs h5).
Writes <out>/seg3d_tracking_comparison.csv and prints a summary.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

F = Path(r"C:\Tracking\NucleiTracking\data\interim\lightsheet\final_processing")
SPOTS = Path(r"C:\Tracking\BlastodermAnalysis\data\spots")
EMB = {
    "20250131": "tracking_20250131",
    "20250625": "tracking_20250625_motion",
    "20250702": "tracking_20250702",
    "20250704": "tracking_20250704_motion",
}


def h5(path):
    d = pd.read_hdf(path, "lineages")
    if "is_interpolated" in d:
        d = d[~d.is_interpolated.astype(bool)]
    return d


def best_offset(a, b, frame):
    """offset o such that a.frame == f matches b.frame == f + o best (median NN distance)."""
    pa = a[a.frame == frame][["px_z", "px_y", "px_x"]].to_numpy()
    res = {}
    for o in range(-2, 3):
        pb = b[b.frame == frame + o][["px_z", "px_y", "px_x"]].to_numpy()
        if len(pa) and len(pb):
            res[o] = float(np.median(cKDTree(pb).query(pa)[0]))
    return min(res, key=res.get), res


def agree(a, b, off, tol):
    rows = []
    for f, ga in a.groupby("frame"):
        gb = b[b.frame == f + off]
        if not len(gb):
            continue
        pa, pb = (
            ga[["px_z", "px_y", "px_x"]].to_numpy(),
            gb[["px_z", "px_y", "px_x"]].to_numpy(),
        )
        da = cKDTree(pb).query(pa)[0]
        db = cKDTree(pa).query(pb)[0]
        rows.append(
            {
                "frame": f,
                "cycle": int(gb.cycle.mode().iloc[0]),
                "n_new": len(pa),
                "n_ref": len(pb),
                "ref_found": float((db <= tol).mean()),
                "new_matched": float((da <= tol).mean()),
            }
        )
    return pd.DataFrame(rows)


def track_stats(d, label):
    d = d.copy()
    out = {}
    out[f"{label}_tracks"] = d.track_id.nunique()
    out[f"{label}_tracklets"] = d.tracklet_id.nunique()
    if "n_children" in d:
        div = d[d.n_children == 2]
        out[f"{label}_divisions"] = len(div)
    if "problematic" in d:
        tl = d.groupby("tracklet_id").problematic.first()
        out[f"{label}_problematic_frac"] = round(float(tl.mean()), 3)
    return out


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--tol", type=float, default=5.0)
    a.add_argument("--embryos", nargs="+", default=list(EMB))
    a.add_argument("--suffix", default="_seg3d")
    args = a.parse_args()
    summary, per_cycle = [], []
    for e in args.embryos:
        base = EMB[e]
        new_dir = F / e / f"{base}{args.suffix}"
        new = h5(new_dir / f"{base.replace('tracking_', '')}{args.suffix}_spots.h5")
        old = h5(F / e / base / f"{base.replace('tracking_', '')}_spots.h5")
        ref = h5(SPOTS / f"{e}_spots.h5")
        mid = int(np.median(new.frame))
        off, res = best_offset(new, ref, mid)
        offo, _ = best_offset(new, old, mid)
        ag = agree(new, ref, off, args.tol).assign(embryo=e)
        ago = agree(old, ref, best_offset(old, ref, mid)[0], args.tol).assign(embryo=e)
        pc = ag.groupby("cycle").agg(
            n_new=("n_new", "median"),
            n_ref=("n_ref", "median"),
            ref_found=("ref_found", "mean"),
            new_matched=("new_matched", "mean"),
        )
        pco = ago.groupby("cycle").agg(
            n_old=("n_new", "median"),
            old_ref_found=("ref_found", "mean"),
            old_matched=("new_matched", "mean"),
        )
        pc = pc.join(pco).assign(embryo=e).reset_index()
        per_cycle.append(pc)
        s = {
            "embryo": e,
            "base": base,
            "offset_vs_corrected": off,
            "offset_vs_old": offo,
            "frames_new": f"{new.frame.min()}-{new.frame.max()}",
            "frames_ref": f"{ref.frame.min()}-{ref.frame.max()}",
        }
        s.update(track_stats(new, "new"))
        s.update(track_stats(old, "old"))
        s.update(track_stats(ref, "corrected"))
        for name, d in (("new", new), ("old", old), ("corrected", ref)):
            if "cycle" in d and "n_children" in d:
                dv = d[d.n_children == 2].groupby("cycle").size()
                s.update({f"{name}_div_nc{c}": int(n) for c, n in dv.items()})
        summary.append(s)
        print(e, "offset vs corrected", off, res, flush=True)
        print(pc.round(3).to_string(index=False), flush=True)
    S = pd.DataFrame(summary)
    P = pd.concat(per_cycle)
    S.to_csv(F / f"tracking{args.suffix}_summary.csv", index=False)
    P.to_csv(F / f"tracking{args.suffix}_per_cycle.csv", index=False)
    print(S.T.to_string())


if __name__ == "__main__":
    main()
