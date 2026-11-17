"""3D-segment every recon frame of one embryo with a (fine-tuned) Cellpose model.

Frame numbering: frame = index of the file in the naturally sorted recon dir
(recon_fused_tp_*_ch_0.tif), i.e. the numbering of the pipeline's final_lineages.csv.
The exported <embryo>_spots.h5 uses frame - 1.

Per frame (skipped if <out>/centroids/frame_XXXX.csv exists):
  masks/frame_XXXX_masks.tif   uint16 labels (zlib), after mask_postprocess.postprocess
  centroids/frame_XXXX.csv     label, frame, tp, px_z, px_y, px_x, volume, mean_intensity,
                               std_intensity, max_intensity, bbox_z0..bbox_x1 (end exclusive),
                               runtime_s
--merge writes <out>/centroids_3d_<embryo>.csv (all frames) and runtimes.

Settings: Cellpose defaults with do_3D=True, z_axis=0 (the tp 96 sweep found nothing better for
the fine-tuned model), then largest-component + hole filling (3D and per-plane) + min volume.

Usage:
  python segment_embryo.py --recon R --out O --model M --index $SLURM_ARRAY_TASK_ID
  python segment_embryo.py --recon R --out O --frames 0 10 20 --model M
  python segment_embryo.py --recon R --out O --merge --embryo 20250131
"""

import argparse
import re
import sys
import time
from pathlib import Path

import natsort
import numpy as np
import pandas as pd
import tifffile
from scipy import ndimage as ndi

sys.path.insert(0, str(Path(__file__).parent))
from mask_postprocess import postprocess  # noqa: E402

TP_RE = re.compile(r"recon_fused_tp_(\d+)_ch_0\.tif$")


def recon_files(recon):
    return natsort.natsorted(f for f in Path(recon).glob("recon_fused_tp_*_ch_0.tif") if TP_RE.search(f.name))


def centroid_table(masks, vol, frame, tp):
    labels = np.unique(masks)
    labels = labels[labels > 0]
    if not len(labels):
        return pd.DataFrame()
    cz, cy, cx = np.array(ndi.center_of_mass(masks > 0, masks, labels)).T
    volume = ndi.sum_labels(np.ones_like(masks, np.uint8), masks, labels)
    mean_i = ndi.mean(vol, masks, labels)
    std_i = ndi.standard_deviation(vol, masks, labels)
    max_i = ndi.maximum(vol, masks, labels)
    objs = ndi.find_objects(masks)
    bb = np.array([[s.start for s in objs[l - 1]] + [s.stop for s in objs[l - 1]] for l in labels])
    return pd.DataFrame(dict(
        label=labels, frame=frame, tp=tp, px_z=cz, px_y=cy, px_x=cx, volume=volume.astype(int),
        mean_intensity=mean_i, std_intensity=std_i, max_intensity=max_i,
        bbox_z0=bb[:, 0], bbox_y0=bb[:, 1], bbox_x0=bb[:, 2], bbox_z1=bb[:, 3], bbox_y1=bb[:, 4], bbox_x1=bb[:, 5]))


def merge(out, embryo):
    fs = sorted((out / "centroids").glob("frame_*.csv"))
    df = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)
    df = df.sort_values(["frame", "label"])
    dst = out / f"centroids_3d_{embryo}.csv"
    df.to_csv(dst, index=False)
    per = df.groupby("frame").agg(tp=("tp", "first"), n=("label", "size"), median_vol=("volume", "median"),
                                  runtime_s=("runtime_s", "first"))
    per.to_csv(out / f"per_frame_{embryo}.csv")
    print(f"{len(fs)} frames, {len(df)} nuclei -> {dst}")
    print(per.describe().round(1).to_string())


def main():
    a = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    a.add_argument("--recon", type=Path, required=True)
    a.add_argument("--out", type=Path, required=True)
    a.add_argument("--model", default="/mnt/home/hoatman/ceph/models/cpsam_v2_spots_v1")
    a.add_argument("--index", type=int, default=None, help="one frame (array task id)")
    a.add_argument("--frames", nargs="+", type=int, default=None)
    a.add_argument("--min-vol", type=int, default=30)
    a.add_argument("--merge", action="store_true")
    a.add_argument("--embryo", default="embryo")
    a.add_argument("--overwrite", action="store_true")
    args = a.parse_args()

    if args.merge:
        merge(args.out, args.embryo)
        return

    files = recon_files(args.recon)
    if args.index is not None:
        if args.index >= len(files):
            print(f"index {args.index} >= {len(files)} frames; nothing to do")
            return
        frames = [args.index]
    else:
        frames = args.frames if args.frames is not None else list(range(len(files)))
    (args.out / "masks").mkdir(parents=True, exist_ok=True)
    (args.out / "centroids").mkdir(parents=True, exist_ok=True)
    todo = [f for f in frames if args.overwrite or not (args.out / "centroids" / f"frame_{f:04d}.csv").exists()]
    print(f"{len(files)} recon files; frames to do: {todo}", flush=True)
    if not todo:
        return

    import torch
    from cellpose import models

    model = models.CellposeModel(gpu=True, pretrained_model=args.model)
    for fr in todo:
        f = files[fr]
        tp = int(TP_RE.search(f.name).group(1))
        vol = tifffile.imread(f).astype(np.float32)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.time()
        # eval normalizes its input in place, so pass a copy and keep vol raw for intensities
        masks = model.eval(vol.copy(), do_3D=True, z_axis=0)[0].astype(np.uint16)
        masks, info = postprocess(masks, min_vol=args.min_vol)
        rt = time.time() - t0
        tifffile.imwrite(args.out / "masks" / f"frame_{fr:04d}_masks.tif", masks, compression="zlib")
        tab = centroid_table(masks, vol, fr, tp).assign(runtime_s=round(rt, 1))
        tab.to_csv(args.out / "centroids" / f"frame_{fr:04d}.csv", index=False)  # written last = done
        print(f"frame {fr} ({f.name}): {len(tab)} nuclei, {rt:.0f} s, {info}", flush=True)


if __name__ == "__main__":
    main()
