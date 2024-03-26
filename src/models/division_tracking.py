import pandas as pd
from ..utils import identify_peaks
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx


def separate_mitoses(tracklets, n_divisions):
    peaks = sorted(identify_peaks(tracklets[tracklets["edge_distance"] > 50]["start_time"])[:n_divisions + 1])
    interphase_dividers = [(p1 + p2) / 2 for p1, p2 in zip(peaks, peaks[1:])]
    interphase_dividers.append((tracklets["end_time"].max() + peaks[-1]) / 2)

    print(f"detected division times: {peaks}")

    return interphase_dividers


def quick_tracklets(spots_df):
    start_times = spots_df.groupby("track_id")["FRAME"].min()
    end_times = spots_df.groupby("track_id")["FRAME"].max()
    start_id = spots_df.groupby("track_id")["ID"].first()
    length = end_times - start_times
    edge_distance = spots_df.groupby("track_id")["distance_from_edge"].mean()
    end_x = spots_df.groupby("track_id")["POSITION_X"].last()
    end_y = spots_df.groupby("track_id")["POSITION_Y"].last()

    tracklets = pd.DataFrame(
        {"start_time": start_times, "end_time": end_times, "edge_distance": edge_distance, "end_x": end_x,
         "end_y": end_y,
         "length": length, "start_id": start_id})

    return tracklets


def get_sister_distances(spots_df, tracklets, div_start, div_end) -> (np.array, list[tuple[str, str, str]]):
    """
    Compute sister distances from every sl tracklet to every fl tracklet
    returns (n_sl, n_fl) array of distances,
    (n_sl, 3) array of indices (index, closest_parent_index, closest_sister_index)
    """
    division_spots = spots_df[(spots_df["FRAME"] < div_end) & (spots_df["FRAME"] >= div_start)]
    division_tracklets = tracklets[(tracklets["start_time"] < div_end) & (tracklets["end_time"] > div_end)]

    # subset the full length tracklets and the shorter length tracklets
    division_fl = division_tracklets[division_tracklets["start_time"] <= div_start + 0.9]
    division_fl = division_fl[division_fl.index != 0]
    division_sl = division_tracklets[division_tracklets["start_time"] > div_start + 0.9]
    division_sl = division_sl[division_sl.index != 0]

    # get the spots present in those tracklets
    spots_fl = division_spots[division_spots["track_id"].isin(division_fl.index)]
    spots_sl = division_spots[division_spots["track_id"].isin(division_sl.index)]
    first_spots_sl = spots_sl[spots_sl["ID"].isin(division_sl["start_id"])]

    distances = []
    indices = []

    for i, spot in first_spots_sl.iterrows():
        earlier_fl_spots = spots_fl[spots_fl["FRAME"] < spot["FRAME"]]
        now_or_later_fl_spots = spots_fl[spots_fl["FRAME"] >= spot["FRAME"]]

        # for each track id get last spot from earlier fl spots
        prev_frame_fl_spots = earlier_fl_spots.groupby("track_id").last()
        this_or_next_fl_spots = now_or_later_fl_spots.groupby("track_id").first()

        # parent_offset
        po_x, po_y = (prev_frame_fl_spots["POSITION_X"] - spot["POSITION_X"]), (
                prev_frame_fl_spots["POSITION_Y"] - spot["POSITION_Y"])
        # mean_sister_offset
        ms_x, ms_y = (this_or_next_fl_spots["POSITION_X"] - spot["POSITION_X"]) / 2, (
                this_or_next_fl_spots["POSITION_Y"] - spot["POSITION_Y"]) / 2

        distance = (np.sqrt((ms_x - po_x) ** 2 + (ms_y - po_y) ** 2))
        distances.append(distance)

        # get the indices of the closest parent and sister
        closest_parent = prev_frame_fl_spots["ID"].iloc[np.argmin(distance)]
        closest_sister = this_or_next_fl_spots["ID"].iloc[np.argmin(distance)]
        indices.append((spot["ID"], closest_parent, closest_sister))

    return np.array(distances), indices


def plot_stairplot(sorted_distances, ax, d_max):
    colors = ["blue", "red"]
    label_val = ["closest neighbor", "second closest neighbor"]
    for k, (color, lab) in enumerate(zip(colors, label_val)):
        edges = sorted(list(sorted_distances[:, k]))
        edges.insert(0, 0)
        n = sorted_distances.shape[0]
        values = np.arange(1, n + 1) / n
        ax.stairs(values, edges, color=color, label=lab)

    ax.vlines(d_max, 0, 1, color="black", linestyle="--")
    ax.legend()
    ax.set_xlabel("distance")
    ax.set_ylabel("cumulative distribution")


def plot_sisters(spots_df, sorted_distances, d_max, indices, ax):
    spots = [ind[0] for ind in indices]
    sisters = [ind[2] for ind in indices]

    include = sorted_distances[:, 0] < d_max
    included_sisters = [s for s, i in zip(sisters, include) if i]

    sister_spots = spots_df[spots_df["ID"].isin(included_sisters)]
    sns.scatterplot(x=sister_spots["POSITION_X"], y=sister_spots["POSITION_Y"], color="red", ax=ax, label="sister", s=6)

    spots_spots = spots_df[spots_df["ID"].isin(spots)]
    n_neighbors = np.sum(sorted_distances < d_max, axis=1)

    sns.scatterplot(x=spots_spots["POSITION_X"], y=spots_spots["POSITION_Y"], hue=n_neighbors, ax=ax, s=6)


def map_divisions(spots_df, graph, n_divisions, savepath=None) -> nx.DiGraph:

    if savepath:
        fig1, axes1 = plt.subplots(2, 2, figsize=(10, 10))
        fig2, axes2 = plt.subplots(2, 2, figsize=(10, 10))

    tracklets = quick_tracklets(spots_df)
    interphase_dividers = separate_mitoses(tracklets, n_divisions)

    for division in range(n_divisions):
        div_start, div_end = interphase_dividers[division], interphase_dividers[division + 1]

        distances, indices = get_sister_distances(spots_df, tracklets, div_start, div_end)
        sorted_distances = np.sort(distances, axis=1)

        # get the minimum second-closest neighbor distance
        d_max = np.min(sorted_distances[:, 1]) * 0.8

        print(d_max)

        for distance, (spot, parent, sister) in zip(distances, indices):
            if np.sum(distance < d_max) > 1:
                print(f"division {division} spot {spot} has {np.sum(distance < d_max)} neighbors")

            if min(distance) < d_max:
                graph.add_edge(str(int(parent)), str(int(spot)))

        if not savepath:
            continue

        plot_stairplot(sorted_distances, axes1.flatten()[division], d_max)
        plot_sisters(spots_df, sorted_distances, d_max, indices, axes2.flatten()[division])

    if savepath:
        fig1.savefig(savepath / "stairplot.png")
        fig2.savefig(savepath / "sisters.png")

    return graph