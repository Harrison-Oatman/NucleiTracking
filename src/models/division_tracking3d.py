import pandas as pd
from ..utils import identify_peaks
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from scipy.interpolate import interp1d
from scipy.optimize import linear_sum_assignment
from scipy.spatial import KDTree


def quick_tracklets(spots_df):
    start_times = spots_df.groupby("track_id")["FRAME"].min()
    end_times = spots_df.groupby("track_id")["FRAME"].max()
    start_id = spots_df.groupby("track_id")["ID"].first()
    end_id = spots_df.groupby("track_id")["ID"].last()

    tracklets = pd.DataFrame(
        {"start_time": start_times, "end_time": end_times, "start_id": start_id, "end_id": end_id})

    print(tracklets)

    return tracklets


def get_sister_distances2(spots_df, tracklets, div_start, div_end):
    """
    Compute sister distances from every sl tracklet to every fl tracklet
    returns (n_sl, n_fl) array of distances,
    (n_sl, 3) array of indices (index, closest_parent_index, closest_sister_index)
    """
    division_spots = spots_df[(spots_df["FRAME"] >= div_start)]
    division_tracklets = tracklets[(tracklets["start_time"] < div_end) & (tracklets["end_time"] > div_end)]
    print(division_tracklets)

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


def merge_close_tracklets(tracklets: pd.DataFrame, spots_df: pd.DataFrame, graph: nx.DiGraph, max_dis=6, latest_time=120):
    graph = graph.copy()
    final_tp = spots_df["FRAME"].max()
    spots_df = spots_df.copy()

    data = np.array(spots_df[["FRAME", "POSITION_X", "POSITION_Y", "POSITION_Z"]].values)
    data[:, 0] = data[:, 0]*max_dis*2
    tree = KDTree(data)

    loc_map = {idx: i for i, idx in zip(spots_df.index, spots_df["ID"])}

    n_changed = 0

    for tracklet in tracklets.itertuples():

        if tracklet.Index == 0:
            continue

        t = tracklet.end_time
        start = tracklet.start_time
        sp = tracklet.end_id
        if t > latest_time:
            continue


        pt_loc = loc_map[sp]
        pt = np.array(spots_df.loc[pt_loc, ["FRAME", "POSITION_X", "POSITION_Y", "POSITION_Z"]].values)
        pt_id = spots_df.loc[pt_loc, "ID"]
        pt[0] = pt[0]*max_dis*2

        dd, ii = tree.query(pt, 2)

        if dd[1] > max_dis:
            # no neighbor found near end of track
            continue

        this_spot_loc = spots_df.index[ii[1]]
        tid = spots_df.loc[this_spot_loc, "track_id"]
        tloc = spots_df.loc[this_spot_loc, ["FRAME", "POSITION_X", "POSITION_Y", "POSITION_Z"]].values
        this_spot_id = spots_df.loc[this_spot_loc, "ID"]

        if tracklets.loc[tid, "start_time"] < start or tracklets.loc[tid, "end_time"] <= t:
            continue

        print(tid)
        if tid == 0:
            continue

        prev_spot = spots_df[(spots_df["FRAME"] == tloc[0]) & (spots_df["track_id"] == tracklet.Index)]
        prev_spot_loc = prev_spot[["FRAME", "POSITION_X", "POSITION_Y", "POSITION_Z"]].values

        print(np.linalg.norm(tloc - prev_spot_loc[0]))
        print(t, tracklets.loc[tid, "start_time"], start)

        dis = np.linalg.norm(tloc - prev_spot_loc[0])
        if dis > max_dis:
            continue

        # swap tracklet ids after time point
        outs = list(graph.neighbors(str(this_spot_id)))
        print(outs)
        if len(outs) == 0:
            continue

        assert len(outs) == 1, "tid has multiple children"
        child = outs[0]

        print(str(pt_id))
        graph.add_edge(str(pt_id), str(child))
        graph.remove_edge(str(this_spot_id), child)

        before = (spots_df["track_id"] == tid) & (spots_df["FRAME"] <= t)
        after = spots_df[(spots_df["track_id"] == tid) & (spots_df["FRAME"] > t)].index

        spots_df.loc[after, "track_id"] = tracklet.Index

        n_changed += 1

    print(f"made {n_changed} swaps")

    return spots_df, graph


def interpolate_tracklets(spots_df, graph):
    # finds tracklets with frame gaps and interpolates position
    graph = graph.copy()

    parents = np.array([parent for parent, _ in graph.edges])
    children = np.array([child for _, child in graph.edges])

    print(parents)
    print(spots_df["ID"])

    loc_map = {idx: i for i, idx in zip(spots_df.index, spots_df["ID"])}

    parent_locs = np.array([loc_map[p] for p in parents])
    child_locs = np.array([loc_map[c] for c in children])

    print(parent_locs)

    parent_frame = spots_df.loc[parent_locs, "FRAME"]
    children_frame = spots_df.loc[child_locs, "FRAME"]

    diff = children_frame.values - parent_frame.values

    assert np.min(diff) > 0, "GRAPH EDGES MUST BE PARENT -> CHILD"
    print(f"interpolating {np.sum(diff > 1)} edges")
    new_pts = {
        "ID": [],
        "FRAME": [],
        "track_id": [],
        "POSITION_X": [],
        "POSITION_Y": [],
        "POSITION_Z": []
    }

    this_id = spots_df.index.max() + 10000

    for p, c, pid, cid in zip(parent_locs[diff > 1], child_locs[diff > 1], parents[diff > 1], children[diff > 1]):
        pt, *ploc = spots_df.loc[p, ["FRAME", "POSITION_Z", "POSITION_Y", "POSITION_X"]].values
        ct, *cloc = spots_df.loc[c, ["FRAME", "POSITION_Z", "POSITION_Y", "POSITION_X"]].values

        last_id = int(pid)

        for new_pt_v in range(1, round(ct - pt)):
            new_pt_loc = interp1d([pt, ct], [ploc, cloc], axis=0)(pt + new_pt_v)

            new_pts["ID"].append(this_id)
            graph.add_edge(str(last_id), str(this_id))
            last_id = this_id
            this_id = this_id + 1

            new_pts["FRAME"].append(pt + new_pt_v)
            new_pts["track_id"].append(spots_df.loc[p, "track_id"])
            new_pts["POSITION_Z"].append(new_pt_loc[0])
            new_pts["POSITION_Y"].append(new_pt_loc[1])
            new_pts["POSITION_X"].append(new_pt_loc[2])

        graph.remove_edge(str(pid), str(cid))

    new_pts = pd.DataFrame(new_pts, index=new_pts["ID"])

    spots_df = pd.concat([spots_df, new_pts], axis="index")

    print(len(list(graph.nodes)))
    print(len(spots_df))

    return spots_df, graph


def map_divisions(spots_df, graph, interphase_dividers, savepath=None, no_assign_dis_cost=100) -> nx.DiGraph:

    if savepath:
        fig1, axes1 = plt.subplots(2, 2, figsize=(10, 10))
        fig2, axes2 = plt.subplots(2, 2, figsize=(10, 10))

    tracklets = quick_tracklets(spots_df)
    print(tracklets["start_time"].describe())



    for division in range(len(interphase_dividers) - 1):
        div_start, div_end = interphase_dividers[division], interphase_dividers[division + 1]

        distances, parent_indices, child_indices = get_sister_distances2(spots_df, tracklets, div_start, div_end)
        # print(distances)
        distances = np.nan_to_num(distances, nan=1000)
        n_in = distances.shape[0]
        n_out = distances.shape[1]
        no_assign = np.ones((n_in, n_in)) * no_assign_dis_cost

        print(distances.shape, no_assign.shape)

        distances = np.hstack([distances, no_assign])

        # print(distances.shape)
        sorted_distances = np.sort(distances, axis=1)

        assignment = linear_sum_assignment(distances)
        # print(assignment)
        print(f"number unassigned: {np.sum(assignment[1] > n_out) + max(0, distances.shape[0] - distances.shape[1])}")

        childs = [child_indices[i] for i, j in zip(*assignment) if j < n_out]
        parents = [parent_indices[i, j] for i, j in zip(*assignment) if j < n_out]

        graph.add_edges_from([(str(p), str(c)) for p, c in zip(parents, childs)])

        if not savepath:
            continue

        plot_stairplot(sorted_distances, axes1.flatten()[division])

    if savepath:
        fig1.savefig(savepath / "stairplot.png")
        fig2.savefig(savepath / "sisters.png")

    return graph
