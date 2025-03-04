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
import natsort

"""
--Tasks--
- accept toml config file
- accept multiple input files
"""


def main():
    try:
        multiprocessing.set_start_method('spawn')
    except RuntimeError:
        pass

    print("starting main")
    args = process_cli()

    logging.basicConfig(level=args.level)

    inpath = args.input_dir
    inpath = Path(inpath)
    if not inpath.exists():  # try relative path
        inpath = Path().cwd() / args.input

    assert inpath.exists(), f"directory not found: {inpath}"
    print(f"processing files in {inpath}")

    outpath = args.output
    outpath = Path(outpath) if outpath is not None else inpath.parent / f"cellpose_results_{time.time()}"
    outpath.mkdir(exist_ok=True)

    files = natsort.natsorted([f for f in inpath.iterdir() if f.suffix == '.tif'])
    print(f"found {len(files)} tif files")
    np.random.shuffle(files)

    nprocs = args.nprocs

    start = time.time()

    with multiprocessing.Pool(processes=nprocs) as pool:
        print(f"pool initialized in {time.time() - start} seconds")

        jobs = []
        for i, file in tqdm(enumerate(files)):
            job = pool.apply_async(process_file, (i, str(file.absolute()), args, str(outpath.absolute())))
            jobs.append(job)

        # Wait for all jobs to finish
        for job in jobs:
            job.get()


def process_cli() -> argparse.Namespace:
    argparser = argparse.ArgumentParser(description="script to process raw data from tif")

    argparser.add_argument("-i", "--input_dir", dest="input_dir", help="path to raw file to process", default=None)
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
    argparser.add_argument("--flow3D_smooth", default=0, type=float)

    argparser.add_argument_group("multiprocessing")
    argparser.add_argument("--nprocs", default=1, type=int)

    argparser.add_argument_group("other")
    argparser.add_argument("-l", "--level", default="INFO")
    argparser.add_argument("--axes", default="tyx")

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


def process_file(iter, infile, args, outpath):
    logging.basicConfig(level=args.level)

    infile = Path(infile)
    outpath = Path(outpath)
    outfile = outpath / f"{infile.stem}_masks.tif"

    logging.info(f"outile: {outfile}")

    if outfile.exists():
        logging.info(f"skipping {outfile}, already exists.")
        return

    logging.info(f"processing file {infile.stem} on iter {iter}")

    raw = tifffile.imread(infile)
    raw, axes = handle_axes(raw, args)

    print(raw.shape)

    torch.cuda.set_device(iter % torch.cuda.device_count())
    device = torch.device(f"cuda:{torch.cuda.current_device()}")
    model = models.CellposeModel(gpu=args.use_gpu, model_type=args.model, diam_mean=30., device=device)

    logging.info(f"processing image {infile.stem} on gpu {torch.cuda.current_device()}")

    print([image.shape for image in raw])

    results = model.eval([image for image in raw],
                         channels=[0, 0],
                         channel_axis=3,
                         z_axis=2,
                         batch_size=args.batch_size,
                         diameter=args.diam,
                         cellprob_threshold=args.cellprob_thresh,
                         flow_threshold=args.flow_thresh,
                         flow3D_smooth=args.flow3D_smooth,
                         do_3D=args.do_3d,
                         stitch_threshold=args.stitch_threshold,
                         normalize={"percentile": [1, args.top_percentile]})

    logging.info(f"completed {infile.stem}")
    logging.info(results)


    masks = np.array(results[0])
    probabilities = np.array(results[1][0][2])

    # save the results
    tifffile.imwrite(outfile, masks)

    print(f"saved {outfile}")
    return masks


if __name__ == "__main__":
    main()
