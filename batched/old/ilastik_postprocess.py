import numpy as np
import tifffile
import argparse
import logging
import multiprocessing
from pathlib import Path

import h5py
from scipy.ndimage import convolve
from skimage import morphology


def process_file(infile, output_dir):
    logging.info(f"processing {infile.name}")
    outfile = output_dir / infile.with_suffix(".tif").name

    if outfile.exists():
        logging.info(f"skipping {outfile}, already exists.")
        return

    with h5py.File(infile, "r") as f:
        data = f["exported_data"][:]

    logging.info(f"read {infile.name}")

    # convolve ball over image
    ball = morphology.ball(4)
    data = data / np.max(data)
    data = convolve(data[..., 0], ball)

    # convert to 16-bit
    data = ((data - np.quantile(data, 0.01)) / (np.quantile(data, .9999) - data.min())) * 65535
    data = data.clip(0, 65535)
    data = data.astype(np.uint16)

    tifffile.imwrite(outfile, data)

    logging.info(f"saved {outfile}")


def main():
    args = parse_args()
    logging.basicConfig(level=args.level)

    input_dir = Path(args.input_dir)
    assert input_dir.exists(), f"directory not found: {input_dir}"

    output_dir = Path(args.output)
    assert output_dir.exists(), f"directory not found: {output_dir}"

    files = sorted([f for f in input_dir.iterdir() if f.suffix == '.h5'])
    nprocs = args.nprocs

    with multiprocessing.Pool(processes=nprocs) as pool:
        jobs = [pool.apply_async(process_file, (infile, output_dir)) for infile in files]

        for job in jobs:
            job.get()

def parse_args():
    parser = argparse.ArgumentParser(description="script to process ilastik h5 files with filter")
    parser.add_argument("-i", "--input_dir", help="process all tifs in directory", default=None)
    parser.add_argument("-o", "--output", help="results directory", default=None)
    parser.add_argument("-l", "--level", default="INFO")
    parser.add_argument("--nprocs", help="number of processes", default=None, type=int)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    main()
