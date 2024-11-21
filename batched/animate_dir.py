import argparse
import logging
import tifffile
import numpy as np
from tqdm import tqdm
import time
import multiprocessing
from pathlib import Path
import natsort
import napari


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

    outfile = args.output
    outfile = Path(outfile) if outfile is not None else inpath.parent / f"animation_{time.time()}.tif"

    files = natsort.natsorted([f for f in inpath.iterdir() if f.suffix == '.tif'])
    print(f"found {len(files)} tif files")

    nprocs = args.nprocs
    n_files = len(files)
    n_frames = args.frames_per_file * n_files
    if args.angle_final != args.angle_init:
        angles = np.linspace(args.angle_init, args.angle_final, n_frames)
    else:
        angles = np.array([args.angle_init] * n_frames)

    start = time.time()

    with multiprocessing.Pool(processes=nprocs) as pool:
        print(f"pool initialized in {time.time() - start} seconds")

        jobs = []
        for i, file in tqdm(enumerate(files)):
            this_angles = angles[i * args.frames_per_file: (i + 1) * args.frames_per_file]
            job = pool.apply_async(process_file, (i, str(file.absolute()), this_angles, args, tmpdir))
            jobs.append(job)

        # Wait for all jobs to finish
        for job in jobs:
            job.get()

    print(f"done in {time.time() - start} seconds")
    print(f"writing to {outfile}")

    tifffile.imwrite(outfile, np.array([tifffile.imread(f) for f in tmpdir.iterdir()]))


def process_cli() -> argparse.Namespace:
    argparser = argparse.ArgumentParser(description="script to process raw data from tif")

    argparser.add_argument("-i", "--input_dir", dest="input_dir", help="path to raw file to process", default=None)
    argparser.add_argument("-o", "--output", dest="output", help="results directory", default=None)

    argparser.add_argument_group("napari keywords")
    argparser.add_argument("-r", "--renderer", default="mip")

    argparser.add_argument_group("animation keywords")
    argparser.add_argument("--angle_init", default=0, type=float)
    argparser.add_argument("--angle_final", default=0, type=float)
    argparser.add_argument("-n", "--frames_per_file", default=1, type=int)

    argparser.add_argument_group("multiprocessing")
    argparser.add_argument("--nprocs", default=1, type=int)

    argparser.add_argument_group("other")
    argparser.add_argument("-l", "--level", default="INFO")

    return argparser.parse_args()


def process_file(iter, infile, angles, args, tmpdir):
    volume = tifffile.imread(infile)

    for i, angle in enumerate(angles):
        fpf = len(angles)
        i = iter * fpf + i

        viewer = napari.view_image(volume, name="volume", rendering=args.renderer, scale=(1, 1, 1), translate=(0, 0, 0),
                         rotate=(0, 0, angle), ndisplay=3)
        out = viewer.screenshot()
        tifffile.imwrite(tmpdir / f"{i:04d}.tif", out)
        viewer.close()


if __name__ == "__main__":
    main()
