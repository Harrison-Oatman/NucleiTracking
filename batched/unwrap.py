import numpy as np
import tifffile
import argparse
import logging
from pathlib import Path
from collections import Counter
from circle_fit import taubinSVD
from sklearn.cluster import DBSCAN
import pandas as pd
import json
from tqdm import tqdm


def find_circle(pts):
    if len(pts) < 3:
        return (0, 0, 0)
    cluster = DBSCAN(eps=5, min_samples=5).fit(pts)
    maxlabel = [a[0] for a in Counter(cluster.labels_).most_common(4)]
    pts = pts[np.in1d(cluster.labels_, maxlabel)]

    if len(pts) < 3:
        return (0, 0, 0)
    x, z, r, sig = taubinSVD(pts)
    return x, z, r


def moving_median(vals, k):
    return np.array([np.median(vals[max(0, i - k):min(len(vals), i + k)]) for i in range(len(vals))])


def make_circles(files):

    all_x, all_z, all_r = [], [], []

    for f in files:
        data = tifffile.imread()
        data = data/np.max(data)

        circs = [find_circle(np.argwhere(data[:, i, :, 0] > 0.5)) for i in range(data.shape[1])]
        circs = np.array(circs)
        xs, zs, rs = moving_median(circs[:, 0], 10), moving_median(circs[:, 1], 10), moving_median(circs[:, 2], 10)

        all_x.append(xs)
        all_z.append(zs)
        all_r.append(rs)

        del data

    return np.median(all_x, axis=0), np.median(all_z, axis=0), np.median(all_r, axis=0)


def cirlce_meanip(vals, x, z, r, tol=20, bins=500):
    Z, X = np.meshgrid(np.arange(vals.shape[1]), np.arange(vals.shape[0]))
    mindis2, maxdis2 = (r - tol) ** 2, (r + tol) ** 2
    dis2 = (X - x) ** 2 + (Z - z) ** 2
    thresh = (dis2 > mindis2) & (dis2 < maxdis2)
    X, Z, vals = X[thresh], Z[thresh], vals[thresh]
    theta = np.arctan2(Z - z, X - x) + np.pi
    thetabins = np.floor(theta / (2 * np.pi) * bins).astype(int)
    df = pd.DataFrame({"theta": thetabins, "vals": vals})
    series = pd.Series({i: 0.0 for i in range(bins)})
    series.update(df.groupby("theta")["vals"].mean())
    return np.array(series.astype(float).values)


def order_files(files):
    # detect first position in which strings vary
    for i in range(len(files[0])):
        if len(set([str(f)[i] for f in files])) > 1:
            break

    for j in range(len(files[0])):
        if len(set([str(f)[::-1][j] for f in files])) > 1:
            break

    return sorted(files, key=lambda x: int(str(x)[i:-j]))

def main():
    args = parse_args()
    logging.basicConfig(level=args.level)

    input_dir = Path(args.input_dir)
    assert input_dir.exists(), f"directory not found: {input_dir}"

    output_dir = Path(args.output)
    assert output_dir.exists(), f"directory not found: {output_dir}"

    files = sorted([f for f in input_dir.iterdir() if f.suffix == '.h5'])

    sample_files = np.random.choice(files, 10)
    xs, zs, rs = make_circles(sample_files)

    with open(output_dir / "circle_fit.json", "w") as f:
        json.dump({"x": xs.tolist(), "z": zs.tolist(), "r": rs.tolist()}, f)

    files = order_files(files)

    out = []

    for file in tqdm(files):
        data = tifffile.imread(file)
        data = data/np.max(data)
        unwrapped = np.array([cirlce_meanip(data[:, i, :, 0], xs[i], zs[i], rs[i]) for i in range(data.shape[1])])

        out.append(unwrapped)

    tifffile.imwrite(output_dir / "unwrapped.tif", np.array(out))


def parse_args():
    parser = argparse.ArgumentParser(description="script to unwrap embryos as cylinders")
    parser.add_argument("-i", "--input_dir", help="process all tifs in directory", default=None)
    parser.add_argument("-o", "--output", help="results directory", default=None)
    parser.add_argument("-l", "--level", default="INFO")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    main()
