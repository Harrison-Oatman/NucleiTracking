import skimage
import numpy as np
import tifffile
import argparse
import logging
import multiprocessing
from pathlib import Path
from tqdm import tqdm

import scipy.ndimage as ndi
from scipy.ndimage._ni_support import _normalize_sequence


def rolling_ball_filter(data, ball_radius, spacing=None, top=False, **kwargs):
    """Rolling ball filter implemented with morphology operations

    This implenetation is very similar to that in ImageJ and uses a top hat transform
    with a ball shaped structuring element
    https://en.wikipedia.org/wiki/Top-hat_transform

    Parameters
    ----------
    data : ndarray
        image data (assumed to be on a regular grid)
    ball_radius : float
        the radius of the ball to roll
    spacing : int or sequence
        the spacing of the image data
    top : bool
        whether to roll the ball on the top or bottom of the data
    kwargs : key word arguments
        these are passed to the ndimage morphological operations

    Returns
    -------
    data_nb : ndarray
        data with background subtracted
    bg : ndarray
        background that was subtracted from the data
    """
    ndim = data.ndim
    if spacing is None:
        spacing = 1

    spacing = _normalize_sequence(spacing, ndim)

    radius = np.asarray(_normalize_sequence(ball_radius, ndim))
    mesh = np.array(np.meshgrid(*[np.arange(-r, r + s, s) for r, s in zip(radius, spacing)], indexing="ij"))
    structure = 2 * np.sqrt(1 - ((mesh / radius.reshape(-1, *((1,) * ndim)))**2).sum(0))
    structure[~np.isfinite(structure)] = 0
    if not top:
        # ndi.white_tophat(data, structure=structure, output=background)
        background = ndi.grey_erosion(data, structure=structure, **kwargs)
        background = ndi.grey_dilation(background, structure=structure, **kwargs)
    else:
        # ndi.black_tophat(data, structure=structure, output=background)
        background = ndi.grey_dilation(data, structure=structure, **kwargs)
        background = ndi.grey_erosion(background, structure=structure, **kwargs)

    return data - background, background


def process_file(infile, output_dir):
    print("test")
    logging.info(f"processing {infile.name}")

    outfile = output_dir / infile.name
    if outfile.exists():
        logging.info(f"skipping {outfile}, already exists.")
        return

    raw = tifffile.imread(infile)
    logging.info(f"read {infile.name}")


    # downscale by 0.5
    raw = skimage.transform.downscale_local_mean(raw, (2, 2, 2))[2:]

    # convert to 16-bit
    raw = ((raw - np.quantile(raw, 0.01)) / (np.quantile(raw, .9999) - raw.min())) * 65535
    raw = raw.clip(0, 65535)
    raw = raw.astype(np.uint16)

    # subtract background
    # raw = np.array([rolling_ball_filter(sl, 25)[0] for sl in tqdm(raw, desc=f"subtracting background {infile.name}")])
    # raw = raw[100:105]

    # save
    tifffile.imwrite(outfile, raw, imagej=True, metadata={"axes": "zyx"})

    logging.info(f"saved {outfile}")


def main():
    args = parse_args()
    logging.basicConfig(level=args.level)

    input_dir = Path(args.input_dir)
    assert input_dir.exists(), f"directory not found: {input_dir}"

    output_dir = Path(args.output)
    assert output_dir.exists(), f"directory not found: {output_dir}"

    files = sorted([f for f in input_dir.iterdir() if f.suffix == '.tif'])
    nprocs = args.nprocs

    # for infile in files:
    #     process_file(infile, output_dir)

    with multiprocessing.Pool(processes=nprocs) as pool:
        jobs = [pool.apply_async(process_file, (infile, output_dir)) for infile in files]

        for job in jobs:
            job.get()


def parse_args():
    parser = argparse.ArgumentParser(description="script to copy and process Daniel's data")
    parser.add_argument("-i", "--input_dir", help="process all tifs in directory", default=None)
    parser.add_argument("-o", "--output", help="results directory", default=None)
    parser.add_argument("-l", "--level", default="INFO")
    parser.add_argument("--nprocs", help="number of processes", default=None, type=int)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    main()
