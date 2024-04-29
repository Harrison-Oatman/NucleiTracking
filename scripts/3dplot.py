import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from json import load
import tifffile
from tqdm import tqdm
from skimage.draw import polygom
from src.utils.tracklets import import_tracklets


def main():

    roots = ["embryo008", "embryo014a", "embryo016", "embryo018"]
    datapath = Path(r"D:\Tracking\NucleiTracking") / "data" / "interim" / "confocal"
    plotpath = datapath / "plots"
    plotpath.mkdir(exist_ok=True)

    spots, tracklets, metadata, tracklets_joined = import_tracklets(datapath, roots)

    root = "embryo018"
    spot = spots[root]
    ft_spot = spot[spot["track_n_tracklets"] == 31]

    cycle = 13
    sample = ft_spot[ft_spot["cycle"] == cycle].sample(5000)

    rawfile = datapath / root / f"{root}_MaxIP_bgs.tif"
    raw = tifffile.imread(rawfile)
    shape = raw.shape

    sample["intensity_mean"] = 0.0
    sample["intensity_std"] = 0.0
    sample["intensity_varrat"] = 0.0

    for i, spot in tqdm(sample.iterrows()):
        x, y = spot["POSITION_X"], spot["POSITION_Y"]
        t = round(spot["FRAME"])
        new_track_id = spot["track_id"]

        roi = [float(pt.lstrip("[ ").rstrip("] ")) for pt in spot["roi"].split(",")]

        xs = [round(pt + x) for pt in roi[::2]]
        ys = [round(pt + y) for pt in roi[1::2]]

        rr, cc = polygon(ys, xs, shape[1:])
        intensity_vals = raw[tuple([t] + [rr, cc])]

        # take variance ratio of intensity vals
        sample.loc[i, "intensity_mean"] = intensity_vals.mean()
        sample.loc[i, "intensity_std"] = intensity_vals.std()
        sample.loc[i, "intensity_varrat"] = intensity_vals.var() / intensity_vals.mean()

    sns.scatterplot(sample, x="AREA", y="intensity_mean", hue="time", palette="gist_rainbow")
    plt.title(f"Cycle {cycle}")
    plt.xlabel("Area")
    plt.ylabel("Mean Intensity")
    plt.savefig(plotpath / f"cycle_{cycle}_intensity_scatter.png")

    # area = sample.groupby("time")["AREA"].mean()
    # intensity = sample.groupby("time")["intensity_varrat"].mean()
    # aspect = sample.groupby("time")["ELLIPSE_ASPECTRATIO"].mean()

    # plt.plot(list(area), list(intensity), c="k")
    plt.show()

    fig = plt.figure()
    ax1 = fig.add_subplot(111, projection='3d')

    ax1.scatter(sample["AREA"], sample["intensity_mean"], sample["ELLIPSE_ASPECTRATIO"], c=sample["time"],
                cmap="gist_rainbow")
    # ax1.plot(list(area), list(intensity), list(aspect), c="k")
    ax1.set_xlabel("Area")
    ax1.set_ylabel("Mean Intensity")
    ax1.set_zlabel("Aspect Ratio")
    plt.title(f"Cycle {cycle}")
    plt.savefig(plotpath / f"cycle_{cycle}_3dscatter.png")
    plt.show()


if __name__ == "__main__":
    main()