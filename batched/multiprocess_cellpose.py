import argparse
import logging
import tifffile
import numpy as np
from cellpose import models
from tqdm import tqdm
import torch
import time
import multiprocessing
from pathlib import Path


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
    nprocs = args.nprocs

    chunks = np.array_split(raw, nprocs)

    start = time.time()

    with multiprocessing.Pool(processes=nprocs) as pool:
        jobs = []
        for i, chunk in enumerate(chunks):
            job = pool.apply_async(process_chunk, (i, chunk, args, outpath, infile, axes))
            jobs.append(job)

        # Wait for all jobs to finish
        for job in jobs:
            job.get()

    end = time.time()

    print(f"evaluation took {(end - start) / 60} minutes")


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

    argparser.add_argument_group("other")
    argparser.add_argument("-l", "--level", default="INFO")
    # argparser.add_argument("--detect_channel_axis", dest="detect_channel_axis", default=True)
    argparser.add_argument("--axes", default="tyx")
    argparser.add_argument("--channels", nargs="+", default=[0, 0], type=int)

    return argparser.parse_args()


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

    if (not args.do_3d) and (args.stitch_threshold == 0):
        raw = raw.squeeze(-4)
        axes = "tcyx"

    print(f"reordered input shape: {raw.shape} (axes: {axes})")

    return raw, axes


def process_chunk(rank, chunk, args, outpath, infile, axes):
    torch.cuda.set_device(rank % torch.cuda.device_count())
    device = torch.device(f"cuda:{torch.cuda.current_device()}")
    model = models.CellposeModel(gpu=args.use_gpu, model_type=args.model, diam_mean=30., device=device)

    print(f"starting process {rank} on {torch.cuda.current_device()} with chunk shape {chunk.shape}")

    out = []

    for c in tqdm(chunk, desc=f"process {rank}"):
        results = model.eval(c,
                             channels=args.channels,
                             channel_axis=-3,
                             batch_size=args.batch_size,
                             diameter=args.diam,
                             cellprob_threshold=args.cellprob_thresh,
                             flow_threshold=args.flow_thresh,
                             do_3D=args.do_3d,
                             stitch_threshold=args.stitch_threshold,
                             normalize={"percentile": [1, args.top_percentile]})

        out.append(results[0])

    masks = np.array(out)

    # save the results
    outfile = outpath / f"{infile.stem}_masks_{rank}.tif"
    tifffile.imwrite(outfile, masks, imagej=True, metadata={"axes": axes})

    print(f"saved {outfile} for process {rank}")


if __name__ == "__main__":
    main()
