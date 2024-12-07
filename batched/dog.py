import argparse
import logging
import tifffile
import numpy as np
from tqdm import tqdm
import time
import multiprocessing
from pathlib import Path
import natsort
from skimage.filters import difference_of_gaussians
from skimage.feature import peak_local_max
import pandas as pd


def main():
    args = process_cli()

    logging.basicConfig(level=args.level)

    inpath = args.input_dir
    inpath = Path(inpath)
    if not inpath.exists():  # try relative path
        inpath = Path().cwd() / args.input

    assert inpath.exists(), f"directory not found: {inpath}"
    print(f"processing files in {inpath}")

    tmpdir = Path(args.input_dir) / "tmp"
    if not tmpdir.exists():
        tmpdir.mkdir()

    lo = args.sigma_low
    hi = args.sigma_high

    outfile = args.output
    outfile = Path(outfile) if outfile is not None else inpath.parent / f"dog_lo{lo}_hi{hi}.csv"

    files = natsort.natsorted([f for f in inpath.iterdir() if f.suffix == '.tif'])
    print(f"found {len(files)} tif files")

    nprocs = args.nprocs

    start = time.time()

    with multiprocessing.Pool(processes=nprocs) as pool:

        jobs = []
        for i, file in tqdm(enumerate(files)):
            job = pool.apply_async(process_file, (i, str(file.absolute()), args, tmpdir))
            jobs.append(job)

        # Wait for all jobs to finish
        for job in jobs:
            job.get()

    print(f"done in {time.time() - start} seconds")
    print(f"writing to {outfile}")

    outfiles = natsort.natsorted([f for f in tmpdir.iterdir()])
    dfs = [pd.read_csv(f) for f in outfiles]
    df = pd.concat(dfs)
    df.to_csv(outfile)


def process_cli() -> argparse.Namespace:
    argparser = argparse.ArgumentParser(description="script to process raw data from tif")

    argparser.add_argument("-i", "--input_dir", dest="input_dir", help="path to raw file to process", default=None)
    argparser.add_argument("-o", "--output", dest="output", help="results directory", default=None)

    argparser.add_argument_group("dog keywords")
    argparser.add_argument("--sigma_low", default=3, type=float)
    argparser.add_argument("--sigma_high", default=5, type=float)
    argparser.add_argument("--min_distance", default=4, type=int)
    argparser.add_argument("--threshold_abs", default=25.0, type=float)

    argparser.add_argument_group("multiprocessing")
    argparser.add_argument("--nprocs", default=1, type=int)

    argparser.add_argument_group("other")
    argparser.add_argument("-l", "--level", default="INFO")

    return argparser.parse_args()


def process_file(i, infile, args, tmpdir):
    volume = tifffile.imread(infile)

    v = difference_of_gaussians(volume, args.sigma_low, args.sigma_high)
    pts = peak_local_max(v, min_distance=args.min_distance, threshold_abs=args.threshold_abs)
    vals = v[pts[:, 0], pts[:, 1], pts[:, 2]]

    frame = [i] * pts.shape[0]

    pd.DataFrame({"x": pts[:, 0], "y": pts[:, 1], "z": pts[:, 2], "val": vals, "frame": frame}).to_csv(tmpdir / f"file_{i}.csv")


if __name__ == "__main__":
    main()
