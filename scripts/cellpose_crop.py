import numpy as np
import tifffile
from pathlib import Path


folder = Path(r"D:\Tracking\NucleiTracking\data\interim\lightsheet\2025_02_06\recon\box_project")
out = folder / "train"

random_crop_size = 224


def main():

    files = folder.glob("*tif")

    for f in files:
        raw = tifffile.imread(f)
        shape = raw.shape

        for t in range(shape[0]):
            frame = raw[t]

            y_start = np.random.randint(0, shape[1] - random_crop_size)
            y_end = y_start + random_crop_size
            x_start = np.random.randint(0, shape[2] - random_crop_size)
            x_end = x_start + random_crop_size

            tifffile.imwrite(out / f"{f.stem}_t{t}_random_crop.tif", frame[y_start:y_end, x_start:x_end])


if __name__ == "__main__":
    main()
