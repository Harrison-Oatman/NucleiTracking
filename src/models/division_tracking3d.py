import pandas as pd
from ..utils import identify_peaks
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from scipy.interpolate import interp1d
from scipy.optimize import linear_sum_assignment


def quick_tracklets(spots_df):
    start_times = spots_df.groupby("track_id")["FRAME"].min()
    end_times = spots_df.groupby("track_id")["FRAME"].max()
    start_id = spots_df.groupby("track_id")["ID"].first()

    tracklets = pd.DataFrame(
        {"start_time": start_times, "end_time": end_times, "start_id": start_id})

    return tracklets


def get_sister_distances2(spots_df, tracklets, div_start, div_end):
    """
    Compute sister distances from every sl tracklet to every fl tracklet
    returns (n_sl, n_fl) array of distances,
    (n_sl, 3) array of indices (index, closest_parent_index, closest_sister_index)
    """
    division_spots = spots_df[(spots_df["FRAME"] >= div_start)]
    division_tracklets = tracklets[(tracklets["start_time"] < div_end) & (tracklets["end_time"] > div_end)]

    # subset the full length tracklets and the shorter length tracklets
    division_fl = division_tracklets[division_tracklets["start_time"] < div_start]
    division_fl = division_fl[division_fl.index != 0]
    division_sl = division_tracklets[division_tracklets["start_time"] > div_start + 0.9]
    division_sl = division_sl[division_sl.index != 0]

    # get the spots present in those tracklets
    spots_fl = division_spots[division_spots["track_id"].isin(division_fl.index)]
    spots_sl = division_spots[division_spots["track_id"].isin(division_sl.index)]

    # get positions for each fl tracklet at each potential time point (using interpolation for missing vals)
    t_needed = np.arange(div_start, div_end+1)
    all_positions = []
    spot_ids = []
    for fl_track in division_fl.index:
        fl_track_spots = spots_fl[spots_fl["track_id"] == fl_track]
        this_positions = fl_track_spots[["POSITION_X", "POSITION_Y", "POSITION_Z"]].values
        t_vals = fl_track_spots["FRAME"].values
        interp = interp1d(t_vals, this_positions, axis=0, bounds_error=False)
        all_positions.append(interp(t_needed))

        # get the spot ids corresponding to each time point (fill in missing values with previous spot)
        this_spot_ids = {t: spot_id for t, spot_id in zip(t_vals.astype(int), fl_track_spots["ID"])}
        id_start = this_spot_ids[min(this_spot_ids.keys())]
        this_spot_ids_list = []
        for t in t_needed:
            if t in this_spot_ids:
                id_start = this_spot_ids[t]
            this_spot_ids_list.append(id_start)
        spot_ids.append(this_spot_ids_list)

    all_positions = np.array(all_positions)
    spot_ids = np.array(spot_ids).astype(int)

    # print(spot_ids)

    distance_matrix = np.zeros((len(division_sl), len(division_fl)))
    parent_indices = np.zeros((len(division_sl), len(division_fl)), dtype=int)
    child_indices = []

    for i, sl_track in enumerate(division_sl.index):
        sl_track_spots = spots_sl[spots_sl["track_id"] == sl_track]
        first_spot = sl_track_spots[sl_track_spots["ID"] == division_sl.loc[sl_track, "start_id"]]
        first_spot_pos = first_spot[["POSITION_X", "POSITION_Y", "POSITION_Z"]].values
        t_ind = np.where(t_needed == first_spot["FRAME"].values[0])[0][0]
        for j, fl_track in enumerate(division_fl.index):
            this_position = all_positions[j, t_ind]
            prev_position = all_positions[j, t_ind - 1]
            distance = np.linalg.norm(((this_position + first_spot_pos) / 2) - prev_position, axis=1)

            distance_matrix[i, j] = distance
            parent_indices[i, j] = spot_ids[j, t_ind - 1]

        child_indices.append(first_spot["ID"].values[0])


    print(distance_matrix.shape)
    print(np.sort(distance_matrix, axis=1))

    return distance_matrix, parent_indices, child_indices


def plot_stairplot(sorted_distances, ax, d_max=None):

    colors = ["blue", "red"]
    label_val = ["closest neighbor", "second closest neighbor"]
    for k, (color, lab) in enumerate(zip(colors, label_val)):
        edges = sorted(list(sorted_distances[:, k]))
        edges.insert(0, 0)
        n = sorted_distances.shape[0]
        values = np.arange(1, n + 1) / n
        ax.stairs(values, edges, color=color, label=lab)

    if d_max is not None:
        ax.vlines(d_max, 0, 1, color="black", linestyle="--")
    ax.legend()
    ax.set_xlabel("distance")
    ax.set_ylabel("cumulative distribution")


def map_divisions(spots_df, graph, interphase_dividers, savepath=None) -> nx.DiGraph:

    if savepath:
        fig1, axes1 = plt.subplots(2, 2, figsize=(10, 10))
        fig2, axes2 = plt.subplots(2, 2, figsize=(10, 10))

    tracklets = quick_tracklets(spots_df)

    for division in range(len(interphase_dividers) - 1):
        div_start, div_end = interphase_dividers[division], interphase_dividers[division + 1]

        distances, parent_indices, child_indices = get_sister_distances2(spots_df, tracklets, div_start, div_end)
        distances = np.nan_to_num(distances, nan=1000)
        sorted_distances = np.sort(distances, axis=1)

        assignment = linear_sum_assignment(distances)

        childs = [child_indices[i] for i in assignment[0]]
        parents = [parent_indices[i, j] for i, j in zip(*assignment)]

        graph.add_edges_from([(str(p), str(c)) for p, c in zip(parents, childs)])

        if not savepath:
            continue

        plot_stairplot(sorted_distances, axes1.flatten()[division])

    if savepath:
        fig1.savefig(savepath / "stairplot.png")
        fig2.savefig(savepath / "sisters.png")

    return graph
