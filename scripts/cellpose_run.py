import logging
import argparse
import tifffile
import numpy as np

from math import floor
from skimage.measure import regionprops
from pathlib import Path

from cellpose import models

"""
This script is used to do cellpose inference on a tif movie. 
"""


def main():
    argparser = argparse.ArgumentParser(description="script to process raw data from tif")

    argparser.add_argument("-i", "--input", help="path to raw file to process", default=None)
    argparser.add_argument("-id", "--input_dir", help="process all tifs in directory", default=None)
    argparser.add_argument("-o", "--output", help="results directory", default=None)

    argparser.add_argument_group("cellpose keywords")
    argparser.add_argument("--use_gpu", action="store_true")
    argparser.add_argument("--do_3d", action="store_true")
    argparser.add_argument("--model", default="nuclei")
    argparser.add_argument("--diam", default=9., type=float)
    argparser.add_argument("-c", "--cellprob_thresh", default=0.0, type=float)
    argparser.add_argument("-f", "--flow_thresh", default=0.4, type=float)
    argparser.add_argument("-t", "--top_percentile", default=99.99, type=float)
    argparser.add_argument("--channels", default=None, type=int, nargs="+")

    argparser.add_argument_group("other")
    argparser.add_argument("-l", "--level", default="INFO")
    argparser.add_argument("--detect_channel_axis", default=True)
    argparser.add_argument("--axes", default="tzyxc")

    args = argparser.parse_args()

    logging.basicConfig(level=args.level)
    logging.info("loading tifffile")

    outpath = args.output

    if args.input is not None:
        infile = args.input
        infile = Path(infile)
        assert infile.exists(), f"file not found: {infile}"
        outpath = Path(outpath) if outpath is not None else infile.parent
        cellpose_process_file(infile, outpath, args)

    elif args.input_dir is not None:
        input_dir = args.input_dir
        input_dir = Path(input_dir)
        assert input_dir.exists(), f"directory not found: {input_dir}"
        files = [f for f in input_dir.iterdir() if f.suffix == '.tif']
        for infile in files:
            cellpose_process_file(infile, outpath, args)

    else:
        raise ValueError("must specify input or input_dir")


def handle_axes(raw, args):
    """
    Manipulates the axes of the raw data to match the expected axes of the model.
    """
    axes = {char: i for i, char in enumerate(args.axes)}
    missing_axes = set("tzcyx") - set(axes.keys())

    assert len(axes) == len(raw.shape), f"axes {axes} do not match shape {raw.shape}"
    n = len(axes)

    for j, missing_ax in enumerate(missing_axes):
        axes[missing_ax] = n + j
        raw = np.expand_dims(raw, axis=axes[missing_ax])

    # reorder the axes
    raw = np.moveaxis(raw, [axes[ax] for ax in "tzcyx"], list(range(5)))

    axes = "tzcyx"

    if not args.do_3d:
        raw = raw.squeeze(-4)
        axes = "tcyx"

    print(f"reordered input shape: {raw.shape} (axes: {axes})")

    return raw, axes


def cellpose_process_file(infile, outpath, args):
    model = models.CellposeModel(gpu=args.use_gpu, model_type=args.model, diam_mean=30.)

    outtif = Path(outpath) / f"{infile.stem}_{args.model}masks.tif"

    raw = tifffile.imread(infile)
    raw, axes = handle_axes(raw, args)

    # normalize to percentile
    raw = raw / np.quantile(raw, args.top_percentile / 100.0)
    raw = np.clip(raw, 0, 1)

    if args.channels is None:
        args.channels = [0, 0]

    logging.info(f"shape of cellpose input: {raw.shape}")

    results = model.eval([v for v in raw],
                         channels=args.channels,
                         channel_axis=-3,
                         diameter=args.diam,
                         cellprob_threshold=args.cellprob_thresh,
                         flow_threshold=args.flow_thresh,
                         do_3D=args.do_3d,
                         normalize={"percentile": [1, 100]})

    out = np.array(results[0])

    tifffile.imwrite(outtif, out)

if __name__ == "__main__":
    main()