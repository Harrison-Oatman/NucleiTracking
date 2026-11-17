"""Fine-tune a Cellpose model on the 2D spot pseudo-label planes (build_training_set.py output).

Saves <save-root>/models/<name> and <save-root>/models/<name>_losses.csv (train/test loss per
epoch; test loss is only computed every 10 epochs by Cellpose, NaN otherwise).

Usage: uv run python batched/train_spots_model.py [--base cpsam_v2] [--name cpsam_v2_spots_v1]
"""

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile

DATA = Path(r"C:\Tracking\NucleiTracking\data\interim\lightsheet\seg3d_benchmark\training\spots_v1")
SAVE_ROOT = Path(r"C:\Tracking\NucleiTracking\data\interim\lightsheet\seg3d_benchmark")


def load_pairs(d):
    ims, labs = [], []
    for f in sorted(d.glob("*.tif")):
        if f.name.endswith("_masks.tif"):
            continue
        ims.append(tifffile.imread(f).astype(np.float32))
        labs.append(tifffile.imread(f.with_name(f.stem + "_masks.tif")).astype(np.uint16))
    return ims, labs


def main():
    a = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    a.add_argument("--data", type=Path, default=DATA)
    a.add_argument("--base", default="cpsam_v2")
    a.add_argument("--name", default="cpsam_v2_spots_v1")
    a.add_argument("--epochs", type=int, default=100)
    a.add_argument("--nimg-per-epoch", type=int, default=None, help="default: all train planes")
    a.add_argument("--batch-size", type=int, default=4)
    a.add_argument("--lr", type=float, default=1e-5)
    a.add_argument("--weight-decay", type=float, default=0.1)
    a.add_argument("--min-train-masks", type=int, default=1)
    args = a.parse_args()

    from cellpose import io, models, train

    io.logger_setup()
    tr_x, tr_y = load_pairs(args.data / "train")
    te_x, te_y = load_pairs(args.data / "test")
    print(f"train planes {len(tr_x)}, test planes {len(te_x)}", flush=True)

    model = models.CellposeModel(gpu=True, pretrained_model=args.base)
    t0 = time.time()
    path, tr_loss, te_loss = train.train_seg(
        model.net, train_data=tr_x, train_labels=tr_y, test_data=te_x, test_labels=te_y,
        batch_size=args.batch_size, learning_rate=args.lr, weight_decay=args.weight_decay,
        n_epochs=args.epochs, nimg_per_epoch=args.nimg_per_epoch, min_train_masks=args.min_train_masks,
        save_path=SAVE_ROOT, model_name=args.name, normalize=True,
    )
    dt = time.time() - t0
    te = np.asarray(te_loss, float)
    te[te == 0] = np.nan
    pd.DataFrame({"epoch": np.arange(len(tr_loss)), "train_loss": tr_loss, "test_loss": te}).to_csv(
        SAVE_ROOT / "models" / f"{args.name}_losses.csv", index=False)
    print(f"saved {path} in {dt / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
