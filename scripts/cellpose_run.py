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

    argparser.add_argument("-i", "--input", dest="input", help="path to raw file to process", default=None)
    argparser.add_argument("-id", "--input_dir", dest="input_dir", help="process all tifs in directory", default=None)
    argparser.add_argument("-o", "--output", dest="output", help="results directory", default=None)

    argparser.add_argument_group("cellpose keywords")
    argparser.add_argument("--use_gpu", dest="use_gpu", default=True)
    argparser.add_argument("--do_3d", dest="do_3d", default=False)
    argparser.add_argument("--model", dest="model", default="nuclei")
    argparser.add_argument("--diam", dest="diam", default=9., type=float)
    argparser.add_argument("-c", "--cellprob_thresh", dest="cellprob_thresh", default=0.0, type=float)
    argparser.add_argument("-f", "--flow_thresh", dest="flow_thresh", default=0.4, type=float)

    argparser.add_argument_group("other")
    argparser.add_argument("-l", "--level", dest="level", default="INFO")
    argparser.add_argument("--detect_channel_axis", dest="detect_channel_axis", default=True)

    args = argparser.parse_args()

    logging.basicConfig(level=args.level)
    logging.info("loading tifffile")

    outpath = args.output

    if args.input is not None:
        infile = args.input
        infile = Path(infile)
        assert infile.exists(), f"file not found: {infile}"
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


def cellpose_process_file(infile, outpath, args):

    outfile = Path(outpath) / f"{infile.stem}_masks.npy"
    outtif = Path(outpath) / f"{infile.stem}_masks.tif"
    binout = Path(outpath) / f"{infile.stem}_binary_masks.tif"

    raw = tifffile.imread(infile)

    model = models.CellposeModel(gpu=args.use_gpu, model_type=args.model, diam_mean=30.)

    # find smallest dimension axis and move to last
    if args.detect_channel_axis:
        chan_axis = np.argmin(raw.shape)
        if raw.shape[chan_axis] < 4:  # assume channel axis has less than 4 channels
            # move channel axis to last
            if chan_axis != -1:
                raw = np.moveaxis(raw, chan_axis, -1)
        else:
            # create channel axis if none exists
            np.expand_dims(raw, -1)

    logging.info(f"shape of cellpose input: {raw.shape}")

    results = model.eval([v for v in raw],
                         channels=[0, 0],
                         channel_axis=-1,
                         diameter=args.diam,
                         cellprob_threshold=args.cellprob_thresh,
                         flow_threshold=args.flow_thresh,
                         do_3D=args.do_3d,)

    out = np.array(results[0])

    np.save(str(outfile), out)
    tifffile.imwrite(outtif, out)
    tifffile.imwrite(binout, (1.0 * (out > 0)).astype(np.uint8))

if __name__ == "__main__":
    main()