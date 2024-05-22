import tifffile
import numpy as np


FILEPATH = r"D:\Tracking\NucleiTracking\data\interim\confocal\embryo027\Embryo027-cleaned.tif"
# T_RANGE = (0, 5)  # get the first 5 frames of the tif file


def main():
    out = []

    raw = tifffile.imread(FILEPATH)

    # take average over 3 frames
    for i in range(0, raw.shape[0], 3):
        out.append(raw[i:i+3].mean(axis=0))

    tifffile.imwrite(FILEPATH.replace(".tif", "_averaged.tif"), np.array(out, dtype=np.int16), imagej=True, metadata={"axes": "tzyx"})


if __name__ == "__main__":
    main()