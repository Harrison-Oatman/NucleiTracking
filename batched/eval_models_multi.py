"""Evaluate fine-tuned models on the held-out test frames of all four WT embryos.

Per model x (embryo, frame): Cellpose 3D at defaults + mask_postprocess, scored against the
corrected spots (hit / merged / missed / new, as benchmark_models_vs_spots.score), giant masks
(> 5x the frame's median pseudo-mask volume, from training/spots_v2/label_volumes), and the
"new" masks (no spot within 3 vox) written out so models can be compared on them: for each pair
of models, how many of model A's new masks model B also finds (within 5 vox).

Outputs: <out>/<model>/<E>_t<N>_masks.tif, <E>_t<N>_new.csv, <E>_t<N>_missed.csv, eval.csv
Usage: uv run python batched/eval_models_multi.py --models <path_v1> <path_v2>
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).parent))
from benchmark_models_vs_spots import score
from build_training_set_v2 import SETS, SPOTS, vol_path
from mask_postprocess import postprocess

OUT = Path(
    r"C:\Tracking\NucleiTracking\data\interim\lightsheet\seg3d_benchmark\eval_multi"
)
PSEUDO = Path(
    r"C:\Tracking\NucleiTracking\data\interim\lightsheet\seg3d_benchmark\training\spots_v2\label_volumes"
)


def main():
    a = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    a.add_argument("--models", nargs="+", required=True)
    a.add_argument("--out", type=Path, default=OUT)
    a.add_argument("--factor", type=float, default=5.0)
    a.add_argument(
        "--cluster-v1", action="store_true", help="score v1 from the cluster-run masks"
    )
    args = a.parse_args()

    import torch
    from cellpose import models

    args.out.mkdir(parents=True, exist_ok=True)
    csv = args.out / "eval.csv"
    rows = pd.read_csv(csv).to_dict("records") if csv.exists() else []
    done = {(r["model"], str(r["embryo"]), int(r["n"])) for r in rows}
    spots = {e: pd.read_hdf(SPOTS / f"{e}_spots.h5", "lineages") for e in SETS}
    spots = {e: s[~s.is_interpolated.astype(bool)] for e, s in spots.items()}

    for mref in args.models:
        mname = Path(mref).name
        model = None
        (args.out / mname).mkdir(exist_ok=True)
        for e, cfg in SETS.items():
            for n in cfg["test"]:
                if (mname, e, n) in done:
                    continue
                s = spots[e][spots[e].frame == n - cfg["offset"]]
                P = s[["px_z", "px_y", "px_x"]].to_numpy()
                if mname == "cpsam_v2_spots_v1" and args.cluster_v1:
                    # same model + post-processing, already run on the cluster for every frame
                    from build_training_set_v3 import v1_masks

                    masks, rt = v1_masks(e, n), np.nan
                else:
                    if model is None:
                        model = models.CellposeModel(gpu=True, pretrained_model=mref)
                    vol = tifffile.imread(vol_path(e, n)).astype(np.float32)
                    t0 = time.time()
                    masks = model.eval(vol, do_3D=True, z_axis=0)[0].astype(np.uint16)
                    masks, _ = postprocess(masks)
                    rt = time.time() - t0
                tifffile.imwrite(
                    args.out / mname / f"{e}_t{n:03d}_masks.tif",
                    masks,
                    compression="zlib",
                )
                st, missed, new_c = score(masks, P, P)
                s[missed][
                    ["frame", "track_id", "px_z", "px_y", "px_x", "cycle"]
                ].to_csv(args.out / mname / f"{e}_t{n:03d}_missed.csv", index=False)
                pd.DataFrame(new_c, columns=["z", "y", "x"]).to_csv(
                    args.out / mname / f"{e}_t{n:03d}_new.csv", index=False
                )
                ps = np.bincount(
                    tifffile.imread(PSEUDO / f"{e}_t{n:03d}_pseudo.tif").ravel()
                )[1:]
                med = float(np.median(ps[ps > 0]))
                v = np.bincount(masks.ravel())[1:]
                v = v[v > 0]
                g = v > args.factor * med
                rows.append(
                    dict(
                        model=mname,
                        embryo=e,
                        n=n,
                        cycle=int(s.cycle.mode().iloc[0]),
                        runtime_s=round(rt, 1),
                        **st,
                        n_giant=int(g.sum()),
                        giant_vox_frac=round(float(v[g].sum() / max(v.sum(), 1)), 3),
                    )
                )
                print(rows[-1], flush=True)
                pd.DataFrame(rows).to_csv(csv, index=False)
        del model
        torch.cuda.empty_cache()

    df = pd.DataFrame(rows)
    print(
        df.pivot_table(
            index=["embryo", "n", "cycle"],
            columns="model",
            values=["recall", "missed", "merged", "new", "n_giant"],
        ).to_string()
    )
    # overlap of "new" masks between models
    names = [Path(m).name for m in args.models]
    ov = []
    for e, cfg in SETS.items():
        for n in cfg["test"]:
            for i, A in enumerate(names):
                for B in names[i + 1 :]:
                    fa, fb = (
                        args.out / A / f"{e}_t{n:03d}_new.csv",
                        args.out / B / f"{e}_t{n:03d}_new.csv",
                    )
                    if not (fa.exists() and fb.exists()):
                        continue
                    na, nb = pd.read_csv(fa).to_numpy(), pd.read_csv(fb).to_numpy()
                    a_in_b = (
                        int((cKDTree(nb).query(na)[0] <= 5).sum())
                        if len(na) and len(nb)
                        else 0
                    )
                    ov.append(
                        {
                            "embryo": e,
                            "n": n,
                            "A": A,
                            "B": B,
                            "new_A": len(na),
                            "new_B": len(nb),
                            "A_new_also_in_B": a_in_b,
                        }
                    )
    if ov:
        o = pd.DataFrame(ov)
        o.to_csv(args.out / "new_overlap.csv", index=False)
        print(o.to_string(index=False))


if __name__ == "__main__":
    main()
