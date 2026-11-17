"""Benchmark Cellpose models (3D) against the tracked spots of one embryo.

For each model x timepoint: run Cellpose with do_3D=True on the volume (optionally a y-slab),
save masks, and score them against the spots of that frame:
  hit    = spot inside a mask that holds only that spot
  merged = spot inside a mask shared with other spots
  missed = spot on background
  new    = mask with no spot inside whose centroid is > tol voxels from every spot
Per (model, tp) it writes <out>/<model>/tp_XXX_{masks.tif,row.csv,missed.csv,new.csv}
(coordinates in full-volume voxels). --summarize merges the row CSVs into
per_frame_all.csv, <model>/per_frame.csv and summary.csv (per model x cycle, and all).

Timepoint <-> spots: h5 frame = tp - offset (default 61; spot frames 26-246 = tp 87-307).

Usage:
  # local (Windows defaults): every model x every tp found in --recon-dir/--extra-recon
  uv run python batched/benchmark_models_vs_spots.py
  # one task (cluster array): index into models x tps, model-major
  python benchmark_models_vs_spots.py --recon-dir R --spots S --out O --tps 87 93 ... --task-index $SLURM_ARRAY_TASK_ID
  python benchmark_models_vs_spots.py --model cpdino --tp 307 --recon-dir R --spots S --out O
  # merge
  python benchmark_models_vs_spots.py --out O --summarize
"""

import argparse
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
from scipy import ndimage as ndi
from scipy.spatial import cKDTree

# Local defaults (this PC); override with CLI args elsewhere.
RECON = Path(
    r"C:\Tracking\NucleiTracking\data\interim\lightsheet\final_processing\20250131\recon"
)
EXTRA_RECON = Path(
    r"C:\Tracking\NucleiTracking\data\interim\lightsheet\previous\2025_01_31\recon"
)
SPOTS = Path(r"C:\Tracking\BlastodermAnalysis\data\spots\20250131_spots.h5")
OUT = Path(
    r"C:\Tracking\NucleiTracking\data\interim\lightsheet\seg3d_benchmark\20250131"
)
MODELS = ["cpsam", "cpsam_v2", "cpdino", "cpdino-vitb"]
SHAPE = (361, 901, 378)
TP_OFFSET = 61  # h5 frame = tp - 61
NAME_RE = re.compile(r"recon_fused_tp_(\d+)_ch_0\.tif")
SUM_COLS = ["n_spots", "hit", "merged", "missed", "new", "n_masks", "runtime_s"]


def find_volumes(dirs=(RECON, EXTRA_RECON), shape=SHAPE):
    """tp -> path over dirs (earlier dirs win). Locally, EXTRA_RECON is the same reconstruction
    (tp 307 byte-identical; spots sit on nuclei in every volume). shape=None skips the check."""
    vols, skipped = {}, []
    for d in dirs:
        if d is None or not Path(d).exists():
            continue
        for f in sorted(Path(d).glob("recon_fused_tp_*_ch_0.tif")):
            m = NAME_RE.fullmatch(f.name)
            if not m or int(m.group(1)) in vols:
                continue
            if shape is not None:
                with tifffile.TiffFile(f) as t:
                    s = tuple(t.series[0].shape)
                if s != tuple(shape):
                    skipped.append((f.name, f"shape {s}"))
                    continue
            vols[int(m.group(1))] = f
    return vols, skipped


def score(masks, P, P_all, tol=3.0):
    """P: spots to score (hit/merged/missed); P_all: all spots used for mask sharing and 'new'."""

    def clip(X):
        return np.clip(np.round(X).astype(int), 0, np.array(masks.shape) - 1)

    ids_all = masks[tuple(clip(P_all).T)] if len(P_all) else np.zeros(0, int)
    n_per = pd.Series(ids_all[ids_all > 0]).value_counts()
    ids = masks[tuple(clip(P).T)] if len(P) else np.zeros(0, int)
    hit = np.array([i > 0 and n_per[i] == 1 for i in ids], bool)
    merged = np.array([i > 0 and n_per[i] > 1 for i in ids], bool)
    missed = ids == 0
    labels = np.unique(masks)
    labels = labels[labels > 0]
    cents = (
        np.array(ndi.center_of_mass(masks > 0, masks, labels))
        if len(labels)
        else np.zeros((0, 3))
    )
    no_spot = ~np.isin(labels, ids_all)
    d = (
        cKDTree(P_all).query(cents)[0]
        if len(cents) and len(P_all)
        else np.full(len(cents), np.inf)
    )
    new = no_spot & (d > tol)
    stats = {
        "n_spots": len(P),
        "n_masks": len(labels),
        "hit": int(hit.sum()),
        "merged": int(merged.sum()),
        "missed": int(missed.sum()),
        "new": int(new.sum()),
        "recall": float(hit.mean()) if len(P) else np.nan,
    }
    return stats, missed, cents[new]


def load_spots(path):
    s = pd.read_hdf(path, "lineages")
    return s[~s.is_interpolated.astype(bool)]


def run_one(model, model_name, tp, vol_path, spots, args):
    """Segment + score one (model, tp); returns the row dict. Skips if row CSV exists."""
    import torch

    mdir = args.out / model_name
    mdir.mkdir(parents=True, exist_ok=True)
    row_f = mdir / f"tp_{tp:03d}_row.csv"
    if row_f.exists() and not args.overwrite:
        print(f"skip {model_name} tp {tp}: {row_f} exists")
        return pd.read_csv(row_f).iloc[0].to_dict()

    full = tifffile.imread(vol_path)
    y0, y1 = args.slab if args.slab else (0, full.shape[1])
    vol = full[:, y0:y1].astype(np.float32)
    del full
    s = spots[spots.frame == tp - args.offset]
    cyc = int(s.cycle.mode().iloc[0]) if len(s) else -1
    P_full = s[["px_z", "px_y", "px_x"]].to_numpy()
    in_slab = (P_full[:, 1] >= y0) & (P_full[:, 1] < y1)
    interior = (
        (P_full[:, 1] >= y0 + args.tol) & (P_full[:, 1] < y1 - args.tol)
        if args.slab
        else in_slab
    )
    off = np.array([0, y0, 0])

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.time()
    masks = model.eval(vol, do_3D=True, z_axis=0)[0]
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    rt = time.time() - t0
    masks = masks.astype(np.uint16)
    if not args.no_save_masks:
        tifffile.imwrite(mdir / f"tp_{tp:03d}_masks.tif", masks, compression="zlib")

    st, missed, new_c = score(
        masks, P_full[interior] - off, P_full[in_slab] - off, args.tol
    )
    s[interior][missed][["frame", "track_id", "px_z", "px_y", "px_x", "cycle"]].assign(
        tp=tp
    ).to_csv(mdir / f"tp_{tp:03d}_missed.csv", index=False)
    pd.DataFrame(new_c + off, columns=["z", "y", "x"]).assign(tp=tp).to_csv(
        mdir / f"tp_{tp:03d}_new.csv", index=False
    )
    row = dict(
        model=model_name,
        tp=tp,
        frame=tp - args.offset,
        cycle=cyc,
        runtime_s=round(rt, 1),
        **st,
    )
    pd.DataFrame([row]).to_csv(row_f, index=False)  # written last = task complete
    print(row, flush=True)
    return row


def summarize(out):
    rows = [pd.read_csv(f) for f in sorted(out.glob("*/tp_*_row.csv"))]
    df = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    # fall back to per_frame.csv written by the first (pre-refactor) local run
    for f in sorted(out.glob("*/per_frame.csv")):
        old = pd.read_csv(f)
        if len(df):
            old = old[
                ~old.set_index(["model", "tp"]).index.isin(
                    df.set_index(["model", "tp"]).index
                )
            ]
        df = pd.concat([df, old], ignore_index=True)
    if not len(df):
        print("no results in", out)
        return
    df = df.sort_values(["model", "tp"])
    df.to_csv(out / "per_frame_all.csv", index=False)
    for m, g in df.groupby("model"):
        g.to_csv(out / m / "per_frame.csv", index=False)
    g = df.groupby(["model", "cycle"])[SUM_COLS].sum()
    g["n_frames"] = df.groupby(["model", "cycle"]).size()
    tot = df.groupby("model")[SUM_COLS].sum()
    tot["n_frames"] = df.groupby("model").size()
    tot = tot.assign(cycle="all").set_index("cycle", append=True)
    summary = pd.concat([g, tot])
    summary.index = summary.index.set_levels(
        summary.index.levels[1].astype(str), level=1
    )
    summary = summary.sort_index()
    summary["recall"] = summary.hit / summary.n_spots
    summary["missed_frac"] = summary.missed / summary.n_spots
    summary["merged_frac"] = summary.merged / summary.n_spots
    summary.to_csv(out / "summary.csv")
    print(summary.to_string())


def main():
    a = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    a.add_argument("--recon-dir", type=Path, default=RECON)
    a.add_argument(
        "--extra-recon",
        type=Path,
        default=None,
        help="second recon dir for tps missing from --recon-dir (local default: previous/2025_01_31/recon)",
    )
    a.add_argument("--spots", type=Path, default=SPOTS)
    a.add_argument("--out", type=Path, default=OUT)
    a.add_argument(
        "--offset", type=int, default=TP_OFFSET, help="h5 frame = tp - offset"
    )
    a.add_argument("--models", nargs="+", default=MODELS)
    a.add_argument(
        "--tps",
        nargs="+",
        type=int,
        default=None,
        help="default: all tps with a volume and spots",
    )
    a.add_argument("--model", default=None, help="run only this model (with --tp)")
    a.add_argument("--tp", type=int, default=None, help="run only this tp")
    a.add_argument(
        "--task-index",
        type=int,
        default=None,
        help="index into models x tps (model-major): model = models[i // n_tps], tp = tps[i %% n_tps]",
    )
    a.add_argument(
        "--check-shape",
        action="store_true",
        help=f"skip volumes whose shape != {SHAPE}",
    )
    a.add_argument(
        "--slab", nargs=2, type=int, default=None, help="y0 y1 crop (all z, all x)"
    )
    a.add_argument("--tol", type=float, default=3.0)
    a.add_argument("--no-save-masks", action="store_true")
    a.add_argument("--overwrite", action="store_true")
    a.add_argument(
        "--summarize", action="store_true", help="only merge existing results"
    )
    args = a.parse_args()

    if args.summarize:
        summarize(args.out)
        return

    local = args.recon_dir == RECON
    extra = args.extra_recon or (EXTRA_RECON if local else None)
    vols, skipped = find_volumes(
        (args.recon_dir, extra), SHAPE if (local or args.check_shape) else None
    )
    spots = load_spots(args.spots)
    fmin, fmax = int(spots.frame.min()), int(spots.frame.max())

    def in_range(t):
        return fmin <= t - args.offset <= fmax

    if args.tps is not None:
        tps = list(args.tps)
        for t in tps:
            if t not in vols:
                msg = f"tp {t}: no recon_fused_tp_{t}_ch_0.tif in {args.recon_dir}"
                raise SystemExit(msg)
            if not in_range(t):
                msg = (
                    f"tp {t}: frame {t - args.offset} outside spot frames {fmin}-{fmax}"
                )
                raise SystemExit(msg)
    else:
        tps = sorted(vols)
        for t in list(tps):
            if not in_range(t):
                skipped.append(
                    (
                        vols[t].name,
                        f"tp {t} outside spot frames ({fmin + args.offset}-{fmax + args.offset})",
                    )
                )
                tps.remove(t)

    if args.task_index is not None:
        i = args.task_index
        if not 0 <= i < len(args.models) * len(tps):
            msg = f"task index {i} out of range 0-{len(args.models) * len(tps) - 1}"
            raise SystemExit(msg)
        tasks = [(args.models[i // len(tps)], tps[i % len(tps)])]
    else:
        models = [args.model] if args.model else args.models
        tps = [args.tp] if args.tp is not None else tps
        tasks = [(m, t) for m in models for t in tps]

    args.out.mkdir(parents=True, exist_ok=True)
    if args.task_index is None:
        pd.DataFrame(skipped, columns=["file", "reason"]).to_csv(
            args.out / "skipped_frames.csv", index=False
        )
        print("skipped:", skipped)
    print("tasks:", tasks, flush=True)

    import torch
    from cellpose import models as cpm

    loaded = None
    for model_ref, tp in tasks:
        # built-in name, or path to a trained model file (results labelled by its file name)
        model_name = Path(model_ref).name if Path(model_ref).is_file() else model_ref
        if (
            not args.overwrite
            and (args.out / model_name / f"tp_{tp:03d}_row.csv").exists()
        ):
            print(f"skip {model_name} tp {tp}: done")
            continue
        if loaded is None or loaded[0] != model_ref:
            loaded = None
            torch.cuda.empty_cache()
            loaded = (
                model_ref,
                cpm.CellposeModel(gpu=True, pretrained_model=model_ref),
            )
        run_one(loaded[1], model_name, tp, vols[tp], spots, args)

    if args.task_index is None and args.model is None and args.tp is None:
        summarize(args.out)


if __name__ == "__main__":
    main()
