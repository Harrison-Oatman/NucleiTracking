import argparse
import logging
import tifffile
import numpy as np
from cellpose import models
from tqdm import tqdm
import torch
import time

from pathlib import Path


"""
--Tasks--
- accept toml config file
- accept multiple input files
"""

# DUMMY_MPI_NPROCS = 100
# DUMMY_MPI_RANK = 99


def main():
    args = process_cli()

    logging.basicConfig(level=args.level)
    logging.info("loading tifffile")

    infile = args.input
    infile = Path(infile)
    if not infile.exists():  # try relative path
        infile = Path().cwd() / args.input

    assert infile.exists(), f"file not found: {infile}"

    outpath = args.output
    outpath = Path(outpath) if outpath is not None else infile.parent

    outpath = outpath / f"{infile.stem}_m{args.model}_d{round(args.diam)}_masks"
    outpath.mkdir(exist_ok=True)

    raw = tifffile.imread(infile)
    logging.info(f"shape of input: {raw.shape}")

    raw, axes = handle_axes(raw, args)

    # split the raw data into chunks
    T = raw.shape[0]
    nprocs = args.nprocs
    rank = args.rank
    chunk = raw[(rank * T) // nprocs: ((rank + 1) * T) // nprocs]

    start = time.time()
    print(f"starting process {rank} of {nprocs} on {torch.cuda.current_device()}")

    # process the chunk
    masks = process_chunk(chunk, args)

    end = time.time()

    # save the results
    outfile = outpath / f"{infile.stem}_masks_{rank}.tif"
    tifffile.imwrite(outfile, masks, imagej=True, metadata={"axes": axes})

    print(f"evaluation took {(end - start) / 60} minutes")
    print(f"out: {masks.shape}")


def process_cli() -> argparse.Namespace:
    argparser = argparse.ArgumentParser(description="script to process raw data from tif")

    argparser.add_argument("-i", "--input", dest="input", help="path to raw file to process", default=None)
    argparser.add_argument("-o", "--output", dest="output", help="results directory", default=None)

    argparser.add_argument_group("cellpose keywords")
    argparser.add_argument("--use_gpu", action="store_true")
    argparser.add_argument("--do_3d", action="store_true")
    argparser.add_argument("--model", default="nuclei")
    argparser.add_argument("--diam", default=9., type=float)
    argparser.add_argument("-c", "--cellprob_thresh", default=0.0, type=float)
    argparser.add_argument("-f", "--flow_thresh", default=0.4, type=float)
    argparser.add_argument("-t", "--top_percentile", default=99.99, type=float)
    argparser.add_argument("--stitch_threshold", default=0.0, type=float)
    argparser.add_argument("--batch_size", default=8, type=int)

    argparser.add_argument_group("multiprocessing")
    argparser.add_argument("--nprocs", default=1, type=int)
    argparser.add_argument("--rank", default=0, type=int)

    argparser.add_argument_group("other")
    argparser.add_argument("-l", "--level", default="INFO")
    # argparser.add_argument("--detect_channel_axis", dest="detect_channel_axis", default=True)
    argparser.add_argument("--axes", default="tyx")

    return argparser.parse_args()


def handle_axes(raw, args):
    """
    Manipulates the axes of the raw data to match the expected axes of the model.
    """
    axes = {char: i for i, char in enumerate(args.axes)}
    missing_axes = set("tzcyx") - set(axes.keys())

    for j, missing_ax in enumerate(missing_axes):
        axes[missing_ax] = len(axes) + j
        raw = np.expand_dims(raw, axis=axes[missing_ax])

    # reorder the axes
    raw = np.moveaxis(raw, [axes[ax] for ax in "tzcyx"], list(range(5)))

    axes = "tzyx"

    if (not args.do_3d) and (args.stitch_threshold == 0):
        raw = raw.squeeze(-4)
        axes = "tyx"

    print(f"reordered input shape: {raw.shape} (axes: {axes})")

    return raw, axes


def process_chunk(chunk, args):
    model = models.CellposeModel(gpu=args.use_gpu, model_type=args.model, diam_mean=30.)

    print(chunk.shape)

    out = []

    for c in tqdm(chunk):
        results = model.eval(c,
                             channels=[0, 0],
                             channel_axis=-3,
                             batch_size=args.batch_size,
                             diameter=args.diam,
                             cellprob_threshold=args.cellprob_thresh,
                             flow_threshold=args.flow_thresh,
                             do_3D=args.do_3d,
                             stitch_threshold=args.stitch_threshold,
                             normalize={"percentile": [1, args.top_percentile]},)

        out.append(results[0])

    return np.array(out)


if __name__ == "__main__":

    main()
