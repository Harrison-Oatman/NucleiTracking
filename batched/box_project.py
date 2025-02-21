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
    outpath = Path(outpath) if outpath is not None else inpath.parent / f"box_project"
    outpath.mkdir(exist_ok=True)

    files = natsort.natsorted([f for f in inpath.iterdir() if f.suffix == '.tif'])
    print(f"found {len(files)} tif files")

    nprocs = args.nprocs

    start = time.time()

    with multiprocessing.Pool(processes=nprocs) as pool:
        print(f"pool initialized in {time.time() - start} seconds")

        jobs = []
        for i, file in tqdm(enumerate(files)):
            job = pool.apply_async(process_file, (i, str(file.absolute()), args, str(outpath.absolute())))
            jobs.append(job)

        vals_and_locs = [job.get() for job in jobs]

    vals = [v for v, _ in vals_and_locs]
    locs = [l for _, l in vals_and_locs]

    for i in range(len(vals[0])):
        v_i = np.stack([v[i] for v in vals], 0)
        l_i = np.stack([l[i] for l in locs], 0)

        tifffile.imwrite(outpath / f"all_vals_{i}.tif", v_i)
        tifffile.imwrite(outpath / f"all_locs_{i}.tif", np.array(l_i, dtype=int))


def process_cli() -> argparse.Namespace:
    argparser = argparse.ArgumentParser(description="script to process raw data from tif")

    argparser.add_argument("-i", "--input_dir", dest="input_dir", help="path to raw file to process", default=None)
    argparser.add_argument("-o", "--output", dest="output", help="results directory", default=None)

    argparser.add_argument_group("position keywords")

    argparser.add_argument("z_lo", help="low value axis 0", type=int)
    argparser.add_argument("z_hi", help="high value axis 0", type=int)
    argparser.add_argument("y_lo", help="low value axis 1", type=int)
    argparser.add_argument("y_hi", help="high value axis 1", type=int)
    argparser.add_argument("x_lo", help="low value axis 2", type=int)
    argparser.add_argument("x_hi", help="high value axis 2", type=int)

    argparser.add_argument_group("multiprocessing")
    argparser.add_argument("--nprocs", default=1, type=int)

    argparser.add_argument_group("other")
    argparser.add_argument("-l", "--level", default="INFO")

    return argparser.parse_args()


def process_file(j, infile, args, outpath):
    logging.basicConfig(level=args.level)

    infile = Path(infile)
    outpath = Path(outpath)

    logging.info(f"processing file {infile.stem} on iter {j}")

    raw = tifffile.imread(str(infile))

    ls = []
    vs = []

    ls.append(np.argmax(raw[:args.z_lo], axis=0))
    vs.append(np.max(raw[:args.z_lo], axis=0))

    ls.append(np.argmax(raw[args.z_hi:], axis=0) + args.z_hi)
    vs.append(np.max(raw[args.z_hi:], axis=0))

    ls.append(np.argmax(raw[:, :args.y_lo], axis=1))
    vs.append(np.max(raw[:, :args.y_lo], axis=1))

    ls.append(np.argmax(raw[:, args.y_hi:], axis=1) + args.y_hi)
    vs.append(np.max(raw[:, args.y_hi:], axis=1))

    ls.append(np.argmax(raw[:, :, :args.x_lo], axis=2))
    vs.append(np.max(raw[:, :, :args.x_lo], axis=2))

    ls.append(np.argmax(raw[:, :, args.x_hi:], axis=2) + args.x_hi)
    vs.append(np.max(raw[:, :, args.x_hi:], axis=2))

    for i, (v, l) in enumerate(zip(vs, ls)):
        val_outfile = outpath / f"{infile.stem}_box_project_{i}.tif"
        loc_outfile = outpath / f"{infile.stem}_box_project_{i}_locs.tif"

        # save the results
        tifffile.imwrite(val_outfile, v)
        tifffile.imwrite(loc_outfile, np.array(l))

    return vs, ls


if __name__ == "__main__":
    main()
