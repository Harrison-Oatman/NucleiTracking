import skimage
import numpy as np
import tifffile
import argparse
import logging
import time

from pathlib import Path


"""
Copies files from input to output directory, as a multiprocess task.
converts data to 16-bit integer, downscales by a factor of 0.5 in x, y, z, and saves as tif.
"""

def main():
    args = parse_args()
    logging.basicConfig(level=args.level)

    input_dir = Path(args.input_dir)
    assert input_dir.exists(), f"directory not found: {input_dir}"

    output_dir = Path(args.output)
    assert output_dir.exists(), f"directory not found: {output_dir}"

    files = [f for f in input_dir.iterdir() if f.suffix == '.tif']

    process_id = args.rank
    nprocs = args.nprocs

    this_files = np.array_split(files, nprocs)[process_id]

    for infile in this_files:
        logging.info(f"processing {infile}")
        raw = tifffile.imread(infile)

        # downscale by 0.5
        raw = skimage.transform.downscale_local_mean(raw, (2, 2, 2))

        # convert to 16-bit integer
        raw = raw.astype(np.uint16)

        # save
        outfile = output_dir / infile.name
        tifffile.imwrite(outfile, raw, imagej=True, metadata={"axes": "zyx"})

        logging.info(f"saved {outfile}")


def parse_args():
    parser = argparse.ArgumentParser(description="script to copy and process Daniel's data")
    parser.add_argument("-i", "--input_dir",  help="process all tifs in directory", default=None)
    parser.add_argument("-o", "--output", help="results directory", default=None)
    parser.add_argument("-l", "--level", default="INFO")

    parser.add_argument("--rank", help="process id", default=None)
    parser.add_argument("--nprocs", help="number of processes", default=None)

    args = parser.parse_args()
    return args