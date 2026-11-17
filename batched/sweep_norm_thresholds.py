"""Sweep Cellpose 3D settings on one timepoint: normalization (network input) x mask settings
(cellprob_threshold, flow3D_smooth; recomputed from the same flows, no extra network run).

Scores every combination against the spots (benchmark_models_vs_spots.score) and counts giant
masks (> 5x the frame's median pseudo-mask volume). Writes <out>/sweep_tpXXX.csv and the masks
as <out>/<model>/<norm>_cp<c>_s<s>_tpXXX_masks.tif.

Usage: uv run python batched/sweep_norm_thresholds.py --tp 96 --models cpsam_v2 <path/to/finetuned>
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
from scipy.ndimage import gaussian_filter

sys.path.insert(0, str(Path(__file__).parent))
from benchmark_models_vs_spots import SPOTS, TP_OFFSET, find_volumes, load_spots, score  # noqa: E402

OUT = Path(r"C:\Tracking\NucleiTracking\data\interim\lightsheet\seg3d_benchmark\sweep_20250131")
PSEUDO = Path(r"C:\Tracking\NucleiTracking\data\interim\lightsheet\seg3d_benchmark\training\spots_v1\label_volumes")
NORMS = {
    "p1-99": {"percentile": [1.0, 99.0]},
    "p1-99.9": {"percentile": [1.0, 99.9]},
    "p1-99.99": {"percentile": [1.0, 99.99]},
    "tile128": {"tile_norm_blocksize": 128},
}
MASK_SETTINGS = [(0.0, 0), (1.0, 0), (2.0, 0), (0.0, 1)]  # (cellprob_threshold, flow3D_smooth)


def main():
    a = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    a.add_argument("--tp", type=int, default=96)
    a.add_argument("--models", nargs="+", required=True)
    a.add_argument("--norms", nargs="+", default=list(NORMS))
    a.add_argument("--out", type=Path, default=OUT)
    a.add_argument("--factor", type=float, default=5.0)
    args = a.parse_args()

    import torch
    from cellpose import dynamics, models

    vols, _ = find_volumes()
    vol = tifffile.imread(vols[args.tp]).astype(np.float32)
    s = load_spots(SPOTS)
    P = s[s.frame == args.tp - TP_OFFSET][["px_z", "px_y", "px_x"]].to_numpy()
    ps = np.bincount(tifffile.imread(PSEUDO / f"tp_{args.tp:03d}_pseudo.tif").ravel())[1:]
    med = float(np.median(ps[ps > 0]))
    args.out.mkdir(parents=True, exist_ok=True)
    csv = args.out / f"sweep_tp{args.tp:03d}.csv"
    rows = pd.read_csv(csv).to_dict("records") if csv.exists() else []
    done = {(r["model"], r["norm"], r["cellprob"], r["smooth"]) for r in rows}

    for mref in args.models:
        mname = Path(mref).name if Path(mref).is_file() else mref
        model = models.CellposeModel(gpu=True, pretrained_model=mref)
        (args.out / mname).mkdir(exist_ok=True)
        for norm in args.norms:
            if all((mname, norm, c, sm) in done for c, sm in MASK_SETTINGS):
                continue
            t0 = time.time()
            _, flows, _ = model.eval(vol, do_3D=True, z_axis=0, compute_masks=False,
                                     normalize={"normalize": True, **NORMS[norm]})
            t_net = time.time() - t0
            dP, cellprob = flows[1], flows[2]
            for c, sm in MASK_SETTINGS:
                if (mname, norm, c, sm) in done:
                    continue
                t1 = time.time()
                d = gaussian_filter(dP, [0, sm, sm, sm]) if sm else dP
                masks = dynamics.resize_and_compute_masks(
                    d, cellprob, niter=200, cellprob_threshold=c, do_3D=True, min_size=15,
                    max_size_fraction=0.4, device=model.device).astype(np.uint16)
                tifffile.imwrite(args.out / mname / f"{norm}_cp{c:g}_s{sm}_tp{args.tp:03d}_masks.tif",
                                 masks, compression="zlib")
                st, _, _ = score(masks, P, P)
                v = np.bincount(masks.ravel())[1:]
                v = v[v > 0]
                g = v > args.factor * med
                rows.append(dict(model=mname, norm=norm, cellprob=c, smooth=sm, tp=args.tp, **st,
                                 n_giant=int(g.sum()),
                                 giant_vox_frac=round(float(v[g].sum() / max(v.sum(), 1)), 3),
                                 max_vol_x_median=round(float(v.max() / med), 1) if len(v) else 0,
                                 net_s=round(t_net, 1), mask_s=round(time.time() - t1, 1)))
                print(rows[-1], flush=True)
                pd.DataFrame(rows).to_csv(csv, index=False)
            del dP, cellprob, flows
            torch.cuda.empty_cache()
        del model
        torch.cuda.empty_cache()
    df = pd.DataFrame(rows)
    print(df[["model", "norm", "cellprob", "smooth", "recall", "merged", "missed", "new", "n_giant",
              "giant_vox_frac"]].to_string(index=False))


if __name__ == "__main__":
    main()
