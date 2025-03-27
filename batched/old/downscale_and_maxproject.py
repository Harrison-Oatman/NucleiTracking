import skimage
import numpy as np
import tifffile
import argparse
import logging
import multiprocessing
from pathlib import Path


def process_file(infile, output_dir):
    print("test")
    logging.info(f"processing {infile.name}")

    downscaled_outfile = output_dir / "downscaled" / infile.name
    mips_outfile = output_dir / "mips" / infile.name

    if downscaled_outfile.exists():
        logging.info(f"skipping {downscaled_outfile}, already exists.")
        return

    raw = tifffile.imread(infile)
    logging.info(f"read {infile.name}")

    # downscale by 0.5
    raw = skimage.transform.downscale_local_mean(raw, (1, 2, 2))[2:]

    # globally normalize and convert to 16 bit
    data = raw.astype(np.int16)

    # save
    tifffile.imwrite(downscaled_outfile, data)

    mip = np.max(data, axis=0)
    tifffile.imwrite(mips_outfile, mip)

    logging.info(f"saved {downscaled_outfile}")


def main():
    args = parse_args()
    logging.basicConfig(level=args.level)

    input_dir = Path(args.input_dir)
    assert input_dir.exists(), f"directory not found: {input_dir}"

    output_dir = Path(args.output)
    assert output_dir.exists(), f"directory not found: {output_dir}"
    (output_dir / "downscaled").mkdir(exist_ok=True, parents=True)
    (output_dir / "mips").mkdir(exist_ok=True, parents=True)

    files = sorted([f for f in input_dir.iterdir() if f.suffix == '.tif'])
    nprocs = args.nprocs

    with multiprocessing.Pool(processes=nprocs) as pool:
        jobs = [pool.apply_async(process_file, (infile, output_dir)) for infile in files]

        for job in jobs:
            job.get()


def parse_args():
    parser = argparse.ArgumentParser(description="script to downscale and maxproject tifs from a trajectory")
    parser.add_argument("-i", "--input_dir", help="process all tifs in directory", default=None)
    parser.add_argument("-o", "--output", help="results directory", default=None)
    parser.add_argument("-l", "--level", default="INFO")
    parser.add_argument("--nprocs", help="number of processes", default=None, type=int)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    main()
