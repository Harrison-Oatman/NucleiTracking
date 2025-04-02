import numpy as np
import tifffile
from pathlib import Path
from tqdm import tqdm


folder = Path(r"D:\Tracking\NucleiTracking\data\interim\lightsheet\2025_03_18_trk\uv_unwrap")
out = folder / "train"

random_crop_size = 224


def main():

    files = [folder / "all_vals.tif"]

    for f in files:
        raw = tifffile.imread(f)
        shape = raw.shape

        for t in tqdm(range(shape[0])):
            frame = raw[t]

            minv = np.min(frame[frame > 0])
            maxv = np.max(frame[frame > 0])
            frame = (frame - minv) / (maxv - minv) * 255
            frame = frame.astype(np.uint8)

            for k in range(4):

                h = shape[1] // 2
                w = shape[2] // 2

                row = k // 2
                col = k % 2

                y_start = np.random.randint(h*row, h*(row + 1) - random_crop_size)
                y_end = y_start + random_crop_size
                x_start = np.random.randint(w*col, w*(col + 1) - random_crop_size)
                x_end = x_start + random_crop_size

                tifffile.imwrite(out / f"{f.stem}_t{t}_random_crop_{k}.tif", frame[y_start:y_end, x_start:x_end])


if __name__ == "__main__":
    main()
