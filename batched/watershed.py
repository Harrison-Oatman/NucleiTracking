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
from skimage.segmentation import watershed
from scipy.ndimage import distance_transform_edt
import pandas as pd



class Peak:

    def __init__(self, label, loc, val):
        self.label = label
        self.loc = loc
        self.val = val
        self.z = loc[0]
        self.y = loc[1]
        self.x = loc[2]


def read_peak_csv(file):
    df = pd.read_csv(file)
    return {row["Unnamed: 0"]: Peak(row["Unnamed: 0"], np.array([row["z"], row["y"], row["x"]]), row["val"]) for i, row in df.iterrows()}


def main():
    args = process_cli()

    logging.basicConfig(level=args.level)

    inpath = args.input_dir
    inpath = Path(inpath)
    if not inpath.exists():  # try relative path
        inpath = Path().cwd() / args.input

    assert inpath.exists(), f"directory not found: {inpath}"
    print(f"processing files in {inpath}")

    outpath = args.output
    outpath = Path(outpath) if outpath is not None else inpath.parent / "watershed"
    w_path = outpath / "watershed"
    nprocs = args.nprocs

    if not outpath.exists():
        outpath.mkdir()

    if not w_path.exists():
        w_path.mkdir()

        files = natsort.natsorted([f for f in inpath.iterdir() if f.suffix == '.tif'])
        print(f"found {len(files)} tif files")

        with multiprocessing.Pool(processes=nprocs) as pool:

            jobs = []
            for i, file in tqdm(enumerate(files)):
                job = pool.apply_async(apply_watershed, (i, str(file.absolute()), args, w_path))
                jobs.append(job)

            peak_maps = [job.get() for job in jobs]

    else:
        peak_maps = []
        for file in natsort.natsorted(w_path.glob("*_peaks.csv")):

            peak_maps.append(read_peak_csv(file))

    watershed_files = natsort.natsorted([f for f in w_path.glob("*.tif")])

    with multiprocessing.Pool(processes=nprocs) as pool:
        jobs = []
        for i, file in tqdm(enumerate(watershed_files[:-1])):
            this_pm = peak_maps[i].copy()
            next_pm = peak_maps[i+1].copy()

            job = pool.apply_async(process_points, (i, file, next_pm, this_pm, args))
            jobs.append(job)

        dfs = [job.get() for job in jobs]

    df = pd.concat(dfs)
    df.to_csv(outpath / "points.csv")


def process_cli() -> argparse.Namespace:
    argparser = argparse.ArgumentParser(description="script to process raw data from tif")

    argparser.add_argument("-i", "--input_dir", dest="input_dir", help="path to raw file to process", default=None)
    argparser.add_argument("-o", "--output", dest="output", help="results directory", default=None)

    argparser.add_argument_group("dog keywords")
    argparser.add_argument("--sigma_low", default=2, type=float)
    argparser.add_argument("--sigma_high", default=6, type=float)
    argparser.add_argument("--min_distance", default=2, type=int)
    argparser.add_argument("--seed_threshold", default=20.0, type=float)

    argparser.add_argument_group("multiprocessing")
    argparser.add_argument("--nprocs", default=1, type=int)

    argparser.add_argument_group("other")
    argparser.add_argument("-l", "--level", default="INFO")

    return argparser.parse_args()


def process_points(i, w_file, next_pts, this_pts, args):
    w = tifffile.imread(w_file)
    X = np.array([p.loc for p in next_pts.values()])

    next_labels = [int(w[tuple(p.astype(int))]) for p in X]
    next_pos = [this_pts[l] for l in next_labels]

    df = {
        "ID": [f"{i+1:03d}:{k:05d}" for k in next_pts.keys()],
        "z": [p.z for p in next_pts.values()],
        "y": [p.y for p in next_pts.values()],
        "x": [p.x for p in next_pts.values()],
        "val": [p.val for p in next_pts.values()],
        "parent": [f"{i:03d}:{l:05d}" for l in next_labels],
        "parent_z": [p.z for p in next_pos],
        "parent_y": [p.y for p in next_pos],
        "parent_x": [p.x for p in next_pos],
    }

    df = pd.DataFrame(df)

    if i == 0:
        df0 = {
            "ID": [f"{i:03d}:{k:05d}" for k in this_pts.keys()],
            "z": [p.z for p in this_pts.values()],
            "y": [p.y for p in this_pts.values()],
            "x": [p.x for p in this_pts.values()],
            "val": [p.val for p in this_pts.values()],
            "parent": [f"{i:03d}:{0:05d}" for l in this_pts.keys()],
            "parent_z": [0 for p in this_pts],
            "parent_y": [0 for p in this_pts],
            "parent_x": [0 for p in this_pts],
        }

        df0 = pd.DataFrame(df0)
        df = pd.concat([df0, df])

    return df


def apply_watershed(i, infile, args, outpath) -> dict:
    volume = tifffile.imread(infile)

    dog = difference_of_gaussians(volume, args.sigma_low, args.sigma_high)  # apply band pass filter

    # find local peaks
    w_peaks = peak_local_max(dog, min_distance=args.min_distance, threshold_abs=args.seed_threshold)
    intensities = dog[w_peaks[:, 0], w_peaks[:, 1], w_peaks[:, 2]]
    peaks_map = {i + 1: Peak(i, p, v) for i, (p, v) in enumerate(zip(w_peaks, intensities))}
    peaks_map[0] = Peak(0, np.array([0, 0, 0]), 0)

    logging.info(f"found {len(peaks_map)} peaks")

    # generate watershed seeds
    img = np.zeros(volume.shape)
    p_locs = np.array([p.loc for p in peaks_map.values()])
    p_labs = np.array([p.label for p in peaks_map.values()])
    p_vals = np.array([p.val for p in peaks_map.values()])
    img[p_locs[:, 0], p_locs[:, 1], p_locs[:, 2]] = p_labs

    # used to mask out the background
    dist = distance_transform_edt(img == 0)

    # apply watershed
    w = watershed(-dog, img.astype(int), mask=dist < 18)

    tifffile.imwrite(outpath / f"{i:03d}.tif", w)

    df = pd.DataFrame({"z": p_locs[:, 0], "y": p_locs[:, 1], "x": p_locs[:, 2], "val": p_vals}, index=p_labs)
    df.to_csv(outpath / f"{i:03d}_peaks.csv")

    return peaks_map


if __name__ == "__main__":
    main()
