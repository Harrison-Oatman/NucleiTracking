import argparse
import logging
import tifffile
import numpy as np
from tqdm import tqdm
import time
import multiprocessing
from pathlib import Path
import natsort
from skimage.filters import difference_of_gaussians, gaussian
from skimage.feature import peak_local_max
from scipy.ndimage import maximum_filter
import pandas as pd
from tomllib import load
import skimage


def main():
    args = process_cli()

    logging.basicConfig(level=args.level)

    inpath = args.input_dir
    inpath = Path(inpath)
    if not inpath.exists():  # try relative path
        inpath = Path().cwd() / args.input

    assert inpath.exists(), f"directory not found: {inpath}"
    print(f"processing files in {inpath}")

    toml_found = False

    if args.toml is not None:
        experiment_toml = Path(args.toml)
        if experiment_toml.exists():
            toml_found = True

    if not toml_found:
        experiment_toml = Path(__file__).parent / "dog_sweep.toml"
        print(f"using default toml: {experiment_toml}")

    dogs, min_distances = process_toml(str(experiment_toml))

    tmpdir = Path(args.input_dir) / experiment_toml.stem
    if not tmpdir.exists():
        tmpdir.mkdir()

    files = natsort.natsorted([f for f in inpath.iterdir() if f.suffix == '.tif'])
    print(f"found {len(files)} tif files")

    nprocs = args.nprocs
    start = time.time()

    with multiprocessing.Pool(processes=nprocs) as pool:

        jobs = []
        for i, file in tqdm(enumerate(files)):
            job = pool.apply_async(process_file, (i, str(file.absolute()), dogs, min_distances, args, tmpdir))
            jobs.append(job)

        # Wait for all jobs to finish
        for job in jobs:
            job.get()

    print(f"done in {time.time() - start} seconds")


def process_toml(fp):
    with open(fp, "rb") as f:
        tt = load(f)

    dog = tt["dog"]
    min_distance = tt["min-distance"]

    return dog, min_distance


def process_cli() -> argparse.Namespace:
    argparser = argparse.ArgumentParser(description="script to process raw data from tif")

    argparser.add_argument("-i", "--input_dir", dest="input_dir", help="path to raw file to process", default=None)
    argparser.add_argument("-o", "--output", dest="output", help="results directory", default=None)
    argparser.add_argument("-t", "--toml", help="path to sweep toml", default=None)

    argparser.add_argument("--threshold_abs", default=25.0, type=float)
    argparser.add_argument("--threshold_rel", default=0.0)
    argparser.add_argument("--rescale_factor", default=2, type=int)

    argparser.add_argument_group("multiprocessing")
    argparser.add_argument("--nprocs", default=1, type=int)

    argparser.add_argument_group("other")
    argparser.add_argument("-l", "--level", default="INFO")

    return argparser.parse_args()


def process_file(i, infile, dogs: dict, min_distances: dict, args, tmpdir):
    logging.info(f"processing file {i}: {infile}")
    volume = tifffile.imread(infile)

    r = args.rescale_factor

    volume = skimage.transform.downscale_local_mean(volume, (r, r, r))

    dfs = []

    for dname, (siglo, sighi) in dogs.items():

        v = difference_of_gaussians(volume, siglo, sighi)
        v_local = 100 * v / maximum_filter(v, size=19)
        logging.info(f"v: {v.max()}")

        for mname, mind in min_distances.items():
            # logging.info(f"mind: {mind}, {type(mind)}")
            # logging.info(f"v: {v.shape}")

            pts = peak_local_max(v, min_distance=mind, threshold_abs=args.threshold_abs)
            vals = v[pts[:, 0], pts[:, 1], pts[:, 2]]
            local_vals = v_local[pts[:, 0], pts[:, 1], pts[:, 2]]

            sublocs = []
            for axis in [0, 1, 2]:
                xoff = np.arange(-1, 2)
                maskarr = np.array([0, 0, 0])
                maskarr[axis] = 1
                xrow = pts[:, np.newaxis, :] + xoff[np.newaxis, :, np.newaxis] * np.broadcast_arrays(maskarr, pts)[0][:,
                                                                                 np.newaxis, :]
                xrow = np.moveaxis(xrow, -1, 0)
                rowvals = v[*xrow]
                subpx = (rowvals[:, 2] - rowvals[:, 0]) / (2 * (rowvals[:, 0] + rowvals[:, 2] - 2 * rowvals[:, 1]))
                sublocs.append(subpx)

            sublocs = np.stack(sublocs, -1)

            subloc_pts = pts + sublocs

            pts = pts*r
            subloc_pts = subloc_pts*r

            df = pd.DataFrame({"z": subloc_pts[:, 0], "y": subloc_pts[:, 1], "x": subloc_pts[:, 2],
                               "zd": pts[:, 0], "yd": pts[:, 1], "xd": pts[:, 2], "val": vals,
                               "local": local_vals})
            df["frame"] = i
            df["dog"] = dname
            df["min-distance"] = mname
            df["sigma-hi"] = sighi
            df["sigma-lo"] = siglo

            dfs.append(df)

    pd.concat(dfs).to_csv(tmpdir / f"file_{i}.csv")


if __name__ == "__main__":
    main()
