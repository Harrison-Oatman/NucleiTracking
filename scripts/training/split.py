import numpy as np
import tifffile
import argparse
import logging
from pathlib import Path
from skimage.io import imread

"""
The purpose of this module is to split a tif movie into individual frames for the purpose of cellpose inference.
This script looks through a directory for specified tif files and saves to desired output
"""

arr_writers = {
    "tif": tifffile.imwrite,
    "npy": np.save,
}


def main():
    argparser = argparse.ArgumentParser(description="splits tif for cellpose inference")

    argparser.add_argument("-i", "--input", dest="input", default=None)
    argparser.add_argument("-o", "--output", dest="output", default=None)

    argparser.add_argument("--in_ext", dest="in_ext", default="tif")
    argparser.add_argument("--out_ext", dest="out_ext", default="tif")

    argparser.add_argument("-l", "--level", dest="level", default="INFO")
    argparser.add_argument("-p", "--pattern", dest="pattern", default=r"{}_{:03d}.tif", help="output pattern")

    args = argparser.parse_args()

    assert args.input != args.output, "input and output cannot match"

    logging.basicConfig(level=args.level)

    imwriter = arr_writers[args.out_ext]

    movies = Path(args.input).rglob(f"*.{args.in_ext}")
    for movie in list(movies):
        m = imread(str(movie))
        outdir = Path(args.output) / movie.stem
        outdir.mkdir(exist_ok=True)
        logging.info(f"writing cellpose series at {outdir}")
        for i, frame in enumerate(m):
            outfile = outdir / args.pattern.format(movie.stem, i)
            imwriter(outfile, frame)


if __name__ == "__main__":
    main()
