"""Make 3D pseudo-masks of nuclei from tracked spots, for training/evaluating 3D segmentation.

For each requested frame: read the fused volume (recon_fused_tp_*.tif), take that frame's spots
(px_z, px_y, px_x in voxels of the fused volume), and grow one mask per spot by seeded watershed
on the smoothed image, clipped at each nucleus's own half maximum above local background.

Frame <-> file: file index (in the naturally sorted recon dir) = spots frame + --frame-offset.
For the exported <embryo>_spots.h5 the offset is 1 (checked on 20250131: h5 frame f matches the
pipeline's final_lineages.csv frame f+1 exactly, and csv frame = tp - 60, so h5 frame f is
recon_fused_tp_{f+61}). On the first frame the script tests offsets -2..+2 by matching spots to
DoG peaks of the image and warns if another offset fits better.

Outputs in --out:
  volumes/frame_XXXX.tif          image (float32, z y x), cropped to the shell bbox
  labels/frame_XXXX_masks.tif     pseudo-masks (uint16), label i = i-th spot of that frame
  labels/frame_XXXX_spots.csv     spots used, with their label id
  cellpose_slices/                2D xy/xz/yz slices + _masks for Cellpose training (optional)
  qc.csv                          per frame: n spots, empty masks, median volume, check value

Usage (cluster):
  python NucleiTracking/batched/spot_pseudolabels.py \
      --recon /mnt/home/hoatman/ceph/lightsheet_20250131/raw_image/downscaled/recon \
      --spots /mnt/home/hoatman/ceph/spots/20250131_spots.h5 \
      --out   /mnt/home/hoatman/ceph/lightsheet_20250131/seg3d/pseudolabels \
      --frames 30 60 85 110 170 225 245 --slices 6
"""

import argparse
import logging
from pathlib import Path

import natsort
import numpy as np
import pandas as pd
import tifffile
from scipy import ndimage as ndi
from skimage.segmentation import watershed

UM_PER_PX = 0.588


def load_spots(path: Path) -> pd.DataFrame:
    if path.suffix in (".h5", ".hdf5"):
        df = pd.read_hdf(path, key="lineages")
    else:
        df = pd.read_csv(path)
    if "is_interpolated" in df:
        df = df[~df["is_interpolated"].astype(bool)]
    return df


def pseudo_masks(
    vol, seeds, smooth=1.0, bg_sigma=15.0, max_dist=10.0, half=0.5, fg_frac=0.15
):
    """Seeded watershed. vol: (z,y,x) float; seeds: (n,3) voxel coords. Returns uint16 labels."""
    vs = ndi.gaussian_filter(vol.astype(np.float32), smooth)
    bg = ndi.gaussian_filter(vs, bg_sigma)
    p = np.clip(np.round(seeds).astype(int), 0, np.array(vol.shape) - 1)
    markers = np.zeros(vol.shape, np.int32)
    markers[p[:, 0], p[:, 1], p[:, 2]] = np.arange(1, len(p) + 1)
    peak = vs[p[:, 0], p[:, 1], p[:, 2]]
    b = bg[p[:, 0], p[:, 1], p[:, 2]]
    bg0 = np.median(b)
    fg = vs > bg0 + fg_frac * (np.median(peak) - bg0)
    near = ndi.distance_transform_edt(markers == 0) < max_dist
    lab = watershed(-vs, markers, mask=fg & near)
    thr = np.zeros(len(p) + 1, np.float32)
    thr[1:] = b + half * (peak - b)
    lab[vs < thr[lab]] = 0
    return lab.astype(np.uint16)


def offset_table(vol, spots, fr, offset, sigma=2.2):
    """Median spot-to-DoG-peak distance when the volume is paired with spots of nearby frames."""
    from scipy.spatial import cKDTree
    from skimage.feature import peak_local_max

    v = ndi.zoom(vol, 0.5, order=1)  # half resolution keeps this cheap
    dog = ndi.gaussian_filter(v, sigma / 2) - ndi.gaussian_filter(v, sigma)
    pk = (
        peak_local_max(dog, min_distance=2, threshold_abs=np.percentile(dog, 99.5))
        * 2.0
    )
    tree = cKDTree(pk)
    rows = []
    for d in range(-2, 3):
        s = spots[spots["frame"] == fr + d]
        if len(s):
            dist, _ = tree.query(s[["px_z", "px_y", "px_x"]].to_numpy())
            rows.append(
                {"offset": offset - d, "median_dist_px": float(np.median(dist))}
            )
    return pd.DataFrame(rows)


def check_alignment(vol, seeds):
    """Mean intensity at spot voxels relative to the volume's 99th percentile (expect > 1)."""
    p = np.clip(np.round(seeds).astype(int), 0, np.array(vol.shape) - 1)
    return float(vol[p[:, 0], p[:, 1], p[:, 2]].mean() / np.percentile(vol, 99))


def export_slices(img, lab, outdir, stem, n, rng):
    """Write n planes per axis through labelled voxels, as Cellpose 2D training pairs."""
    outdir.mkdir(parents=True, exist_ok=True)
    for ax, name in enumerate("zyx"):
        occupied = np.flatnonzero(
            (lab > 0).sum(axis=tuple(a for a in range(3) if a != ax)) > 50
        )
        if len(occupied) == 0:
            continue
        for k in rng.choice(occupied, size=min(n, len(occupied)), replace=False):
            im2 = np.take(img, k, axis=ax)
            lb2 = np.take(lab, k, axis=ax)
            tifffile.imwrite(
                outdir / f"{stem}_{name}{k:03d}.tif", im2.astype(np.float32)
            )
            tifffile.imwrite(outdir / f"{stem}_{name}{k:03d}_masks.tif", lb2)


def main():
    a = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    a.add_argument(
        "--recon", required=True, type=Path, help="dir of recon_fused_tp_*.tif"
    )
    a.add_argument(
        "--spots",
        required=True,
        type=Path,
        help="<embryo>_spots.h5 (key 'lineages') or csv",
    )
    a.add_argument("--out", required=True, type=Path)
    a.add_argument("--frames", nargs="+", type=int, required=True)
    a.add_argument(
        "--slices",
        type=int,
        default=0,
        help="planes per axis per frame for Cellpose training (0 = none)",
    )
    a.add_argument(
        "--pad", type=int, default=12, help="voxels kept around the spots' bbox"
    )
    a.add_argument(
        "--half",
        type=float,
        default=0.5,
        help="mask edge as fraction of peak above background",
    )
    a.add_argument("--seed", type=int, default=0)
    a.add_argument(
        "--frame-offset",
        type=int,
        default=1,
        help="recon file index = spots frame + offset (1 for exported h5)",
    )
    args = a.parse_args()
    logging.basicConfig(level="INFO")

    files = natsort.natsorted(args.recon.glob("*.tif"))
    logging.info(f"{len(files)} recon files; frame 0 = {files[0].name}")
    spots = load_spots(args.spots)
    rng = np.random.default_rng(args.seed)
    (args.out / "volumes").mkdir(parents=True, exist_ok=True)
    (args.out / "labels").mkdir(parents=True, exist_ok=True)
    qc = []

    for i, fr in enumerate(args.frames):
        s = spots[spots["frame"] == fr].reset_index(drop=True)
        f = files[fr + args.frame_offset]
        vol = tifffile.imread(f).astype(np.float32)
        if i == 0:
            t = offset_table(vol, spots, fr, args.frame_offset)
            logging.info(
                f"frame-offset check (lower is better):\n{t.to_string(index=False)}"
            )
            best = int(t.loc[t.median_dist_px.idxmin(), "offset"])
            if best != args.frame_offset:
                logging.warning(
                    f"offset {best} fits better than --frame-offset {args.frame_offset}; check before using labels"
                )
        P = s[["px_z", "px_y", "px_x"]].to_numpy()
        ratio = check_alignment(vol, P)
        logging.info(
            f"frame {fr} ({f.name}): {len(s)} spots, spot/p99 intensity = {ratio:.2f}"
        )
        if ratio < 1.0:
            msg = f"frame {fr}: spots do not sit on nuclei (ratio {ratio:.2f}); check frame mapping"
            raise SystemExit(msg)

        lo = np.maximum(np.floor(P.min(0)).astype(int) - args.pad, 0)
        hi = np.minimum(np.ceil(P.max(0)).astype(int) + args.pad + 1, vol.shape)
        crop = vol[lo[0] : hi[0], lo[1] : hi[1], lo[2] : hi[2]]
        lab = pseudo_masks(crop, P - lo, half=args.half)

        stem = f"frame_{fr:04d}"
        tifffile.imwrite(
            args.out / "volumes" / f"{stem}.tif",
            crop,
            imagej=True,
            resolution=(1 / UM_PER_PX, 1 / UM_PER_PX),
            metadata={"spacing": UM_PER_PX, "unit": "um"},
        )
        tifffile.imwrite(args.out / "labels" / f"{stem}_masks.tif", lab)
        s.assign(
            label=np.arange(1, len(s) + 1),
            crop_z=P[:, 0] - lo[0],
            crop_y=P[:, 1] - lo[1],
            crop_x=P[:, 2] - lo[2],
        ).to_csv(args.out / "labels" / f"{stem}_spots.csv", index=False)

        vols = np.bincount(lab.ravel(), minlength=len(s) + 1)[1:]
        qc.append(
            {
                "frame": fr,
                "file": f.name,
                "n_spots": len(s),
                "check_ratio": ratio,
                "empty_masks": int((vols < 5).sum()),
                "median_vol_um3": float(np.median(vols) * UM_PER_PX**3),
                "bbox_lo": lo.tolist(),
                "bbox_hi": hi.tolist(),
            }
        )
        if args.slices:
            export_slices(
                crop, lab, args.out / "cellpose_slices", stem, args.slices, rng
            )

    pd.DataFrame(qc).to_csv(args.out / "qc.csv", index=False)
    logging.info(pd.DataFrame(qc).to_string())


if __name__ == "__main__":
    main()
