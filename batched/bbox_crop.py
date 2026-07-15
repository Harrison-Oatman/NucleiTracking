import argparse
import contextlib
import json
import logging
import multiprocessing
import time
from pathlib import Path

import natsort
import tifffile
from tqdm import tqdm


def main():
    with contextlib.suppress(RuntimeError):
        multiprocessing.set_start_method("spawn")

    args = process_cli()
    logging.basicConfig(level=args.level)

    inpath = Path(args.input_dir)
    assert inpath.exists(), f"directory not found: {inpath}"
    logging.info(f"processing files in {inpath}")

    outpath = Path(args.output) if args.output else inpath.parent / args.name
    crops_dir = outpath / "crops"
    crops_dir.mkdir(exist_ok=True, parents=True)

    bbox = {
        "z_lo": args.z_lo,
        "z_hi": args.z_hi,
        "y_lo": args.y_lo,
        "y_hi": args.y_hi,
        "x_lo": args.x_lo,
        "x_hi": args.x_hi,
        "name": args.name,
    }
    with open(outpath / "bbox.json", "w") as f:
        json.dump(bbox, f, indent=2)

    files = natsort.natsorted([f for f in inpath.iterdir() if f.suffix == ".tif"])
    logging.info(f"found {len(files)} tif files")

    start = time.time()

    with multiprocessing.Pool(processes=args.nprocs) as pool:
        logging.info(f"pool initialized in {time.time() - start:.1f}s")
        jobs = [
            pool.apply_async(crop_file, (i, str(f), args, str(crops_dir)))
            for i, f in enumerate(files)
        ]
        for job in tqdm(jobs):
            job.get()


def crop_file(i, infile, args, crops_dir):
    logging.basicConfig(level=args.level)
    infile = Path(infile)
    vol = tifffile.imread(str(infile))
    crop = vol[args.z_lo : args.z_hi, args.y_lo : args.y_hi, args.x_lo : args.x_hi]
    outfile = Path(crops_dir) / f"{args.name}_tp{i:03d}.tif"
    tifffile.imwrite(str(outfile), crop)
    logging.info(f"saved crop {outfile.name}, shape {crop.shape}")


def process_cli():
    parser = argparse.ArgumentParser(
        description="Crop recon tif files to a bounding box."
    )
    parser.add_argument(
        "-i", "--input_dir", required=True, help="directory of recon tif files"
    )
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="output directory (default: sibling of input named after --name)",
    )
    parser.add_argument(
        "--name", required=True, help="name for this bounding box region"
    )
    parser.add_argument("--z_lo", type=int, required=True)
    parser.add_argument("--z_hi", type=int, required=True)
    parser.add_argument("--y_lo", type=int, required=True)
    parser.add_argument("--y_hi", type=int, required=True)
    parser.add_argument("--x_lo", type=int, required=True)
    parser.add_argument("--x_hi", type=int, required=True)
    parser.add_argument("--nprocs", type=int, default=1)
    parser.add_argument("-l", "--level", default="INFO")
    return parser.parse_args()


if __name__ == "__main__":
    main()
