"""Open napari on the Cellpose benchmark (batched/benchmark_models_vs_spots.py output).

Layers: raw 4D (t = index into the chosen tps, lazy, each frame scaled to its own
1st-99.8th percentile), all spots (grey), per model: masks (labels; cpsam hidden),
"new" (filled dots) and "missed" (hollow rings) points, one colour per model.

Usage: uv run python batched/view_benchmark_napari.py [--tps 96 148 220 307] [--out DIR]
"""

import argparse
import sys
from pathlib import Path

import dask.array as da
import numpy as np
import pandas as pd
import tifffile
from dask import delayed

sys.path.insert(0, str(Path(__file__).parent))
import napari
from benchmark_models_vs_spots import (
    MODELS,
    OUT,
    SHAPE,
    SPOTS,
    TP_OFFSET,
    find_volumes,
)

COLORS = {
    "cpsam": "#e6194b",
    "cpsam_v2": "#f58231",
    "cpdino": "#3cb44b",
    "cpdino-vitb": "#4363d8",
}
HIDDEN_MASKS = {"cpsam"}


PCT_HI = 99.99  # per-frame scale: 1st pct -> 0, PCT_HI pct -> 1; not clipped (widen limits freely)


def _norm(p):
    v = tifffile.imread(p).astype(np.float32)
    lo, hi = np.percentile(v[::2, ::2, ::2], [1, PCT_HI])
    return (v - lo) / (hi - lo + 1e-6)


def _read_or_zeros(p):
    return tifffile.imread(p) if Path(p).exists() else np.zeros(SHAPE, np.uint16)


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--out", type=Path, default=OUT)
    a.add_argument("--tps", nargs="+", type=int, default=None, help="default: all scored tps")
    a.add_argument("--normalize", action="store_true", help="scale each frame to its own percentiles")
    args = a.parse_args()
    out = args.out

    rows = pd.concat([pd.read_csv(f) for f in out.glob("*/tp_*_row.csv")])
    tps = sorted(args.tps or rows.tp.unique())
    vols, _ = find_volumes()
    t_of = {tp: i for i, tp in enumerate(tps)}
    print("time index -> tp:", dict(enumerate(tps)))

    v = napari.Viewer(title="20250131 Cellpose benchmark  t index -> tp: "
                      + ", ".join(f"{i}:{t}" for i, t in enumerate(tps)))
    if args.normalize:
        raw = da.stack([da.from_delayed(delayed(_norm)(vols[t]), SHAPE, np.float32) for t in tps])
        v.add_image(raw, name=f"raw (per-frame 1-{PCT_HI} pct)", contrast_limits=(0, 1), metadata={"tps": tps})
    else:
        # unchanged values; initial limits = 1st pct .. max over the chosen frames (subsampled)
        lo, hi = np.inf, -np.inf
        for t in tps:
            sub = tifffile.imread(vols[t])[::2, ::2, ::2]
            lo, hi = min(lo, float(np.percentile(sub, 1))), max(hi, float(sub.max()))
        raw = da.stack([da.from_delayed(delayed(tifffile.imread)(vols[t]), SHAPE, np.float32) for t in tps])
        v.add_image(raw, name="raw", contrast_limits=(lo, hi), metadata={"tps": tps})
        v.layers["raw"].contrast_limits_range = (lo, hi)
        print(f"raw contrast limits {lo:.1f}-{hi:.1f}")

    s = pd.read_hdf(SPOTS, "lineages")
    s = s[~s.is_interpolated.astype(bool)]
    s = s.assign(tp=s.frame + TP_OFFSET)
    s = s[s.tp.isin(tps)]
    v.add_points(np.c_[s.tp.map(t_of), s[["px_z", "px_y", "px_x"]]], name="spots (all)",
                 size=4, face_color="#bbbbbb", border_width=0, opacity=0.6)

    extra = sorted(d.name for d in out.iterdir() if d.is_dir() and d.name not in MODELS and any(d.glob("tp_*_row.csv")))
    models = [m for m in MODELS if (out / m).exists()] + extra
    for m in models:
        mdir = out / m
        paths = [mdir / f"tp_{t:03d}_masks.tif" for t in tps]
        if not any(p.exists() for p in paths):
            continue
        lab = da.stack([da.from_delayed(delayed(_read_or_zeros)(p), SHAPE, np.uint16) for p in paths])
        v.add_labels(lab, name=f"{m} masks", visible=m not in HIDDEN_MASKS, opacity=0.4)
    for m in models:
        mdir = out / m
        miss = [pd.read_csv(mdir / f"tp_{t:03d}_missed.csv") for t in tps if (mdir / f"tp_{t:03d}_missed.csv").exists()]
        new = [pd.read_csv(mdir / f"tp_{t:03d}_new.csv") for t in tps if (mdir / f"tp_{t:03d}_new.csv").exists()]
        miss = pd.concat(miss) if miss else pd.DataFrame(columns=["tp", "px_z", "px_y", "px_x"])
        new = pd.concat(new) if new else pd.DataFrame(columns=["tp", "z", "y", "x"])
        c = COLORS.get(m, "#f032e6" if "spots" in m else "white")
        v.add_points(np.c_[miss.tp.map(t_of), miss[["px_z", "px_y", "px_x"]]].astype(float) if len(miss) else np.zeros((0, 4)),
                     name=f"{m} missed (hollow)", size=12, face_color="transparent", border_color=c,
                     border_width=0.25, border_width_is_relative=True)
        v.add_points(np.c_[new.tp.map(t_of), new[["z", "y", "x"]]].astype(float) if len(new) else np.zeros((0, 4)),
                     name=f"{m} new (filled)", size=8, face_color=c, border_width=0)
    v.dims.axis_labels = ("t", "z", "y", "x")
    v.dims.set_point(0, 0)
    v.dims.set_point(1, SHAPE[0] // 2)
    napari.run()


if __name__ == "__main__":
    main()
