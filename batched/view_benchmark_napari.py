"""Open napari on the Cellpose benchmark (batched/benchmark_models_vs_spots.py output).

Layers: raw 4D (t sorted by tp, lazy), all spots (grey), per model "new" (filled) and
"missed" (hollow ring) points, and per model masks (hidden, lazy). Time axis = index into
the sorted timepoint list; tp of each index is printed and shown in the layer metadata.
"""

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


def lazy_stack(paths, dtype):
    return da.stack(
        [da.from_delayed(delayed(tifffile.imread)(p), SHAPE, dtype) for p in paths]
    )


def main():
    df = pd.read_csv(OUT / "per_frame_all.csv")
    tps = sorted(df.tp.unique())
    vols, _ = find_volumes()
    t_of = {tp: i for i, tp in enumerate(tps)}
    print("time index -> tp:", dict(enumerate(tps)))

    v = napari.Viewer(
        title="20250131 Cellpose benchmark (t index -> tp: "
        + ", ".join(f"{i}:{t}" for i, t in enumerate(tps))
        + ")"
    )
    raw = lazy_stack([vols[t] for t in tps], np.float32)
    v.add_image(raw, name="raw", contrast_limits=(0, 300), metadata={"tps": tps})

    s = pd.read_hdf(SPOTS, "lineages")
    s = s[~s.is_interpolated.astype(bool)]
    s = s.assign(tp=s.frame + TP_OFFSET)
    s = s[s.tp.isin(tps)]
    v.add_points(
        np.c_[s.tp.map(t_of), s[["px_z", "px_y", "px_x"]]],
        name="spots (all)",
        size=4,
        face_color="#bbbbbb",
        border_width=0,
        opacity=0.5,
    )

    models = [m for m in MODELS if (OUT / m).exists()]
    for m in models:
        mdir = OUT / m
        mtps = [t for t in tps if (mdir / f"tp_{t:03d}_masks.tif").exists()]
        if mtps and len(mtps) == len(tps):
            v.add_labels(
                lazy_stack([mdir / f"tp_{t:03d}_masks.tif" for t in tps], np.uint16),
                name=f"{m} masks",
                visible=False,
            )
    for m in models:
        mdir = OUT / m
        miss = pd.concat([pd.read_csv(f) for f in mdir.glob("tp_*_missed.csv")])
        new = pd.concat([pd.read_csv(f) for f in mdir.glob("tp_*_new.csv")])
        c = COLORS.get(m, "white")
        v.add_points(
            np.c_[miss.tp.map(t_of), miss[["px_z", "px_y", "px_x"]]]
            if len(miss)
            else np.zeros((0, 4)),
            name=f"{m} missed (hollow)",
            size=12,
            face_color="transparent",
            border_color=c,
            border_width=0.25,
            border_width_is_relative=True,
        )
        v.add_points(
            np.c_[new.tp.map(t_of), new[["z", "y", "x"]]]
            if len(new)
            else np.zeros((0, 4)),
            name=f"{m} new (filled)",
            size=8,
            face_color=c,
            border_width=0,
        )
    v.dims.axis_labels = ("t", "z", "y", "x")
    napari.run()


if __name__ == "__main__":
    main()
