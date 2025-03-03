import pandas as pd

from tqdm import tqdm
from xml.etree import ElementTree as ET
from networkx import DiGraph, connected_components
import numpy as np
from sklearn.cluster import DBSCAN
from scipy.interpolate import interp1d
from scipy.spatial import KDTree
from scipy.optimize import linear_sum_assignment


### rules:
# spot id is always an integer


def quick_tracklets(spots_df, column="track_id") -> pd.DataFrame:
    spots_df = spots_df.sort_values(by=["FRAME"])

    start_times = spots_df.groupby(column)["FRAME"].min()
    end_times = spots_df.groupby(column)["FRAME"].max()
    start_id = spots_df.groupby(column)["graph_key"].first()
    end_id = spots_df.groupby(column)["graph_key"].last()

    tracklets = pd.DataFrame(
        {"start_time": start_times, "end_time": end_times, "start_id": start_id, "end_id": end_id})

    return tracklets


def merge_close_tracklets(spots_df: pd.DataFrame, graph: DiGraph, max_dis=6, latest_time=180):
    graph = graph.copy()
    spots_df = spots_df.copy()
    tracklets = quick_tracklets(spots_df, column="linear_track_id")

    print(np.sum(tracklets["start_time"].isna()))

    # get locations of points in space; time axis is spread out
    data = np.array(spots_df[["FRAME", "POSITION_X", "POSITION_Y", "POSITION_Z"]].values)
    data[:, 0] = data[:, 0]*max_dis*2
    tree = KDTree(data)

    n_changed = 0

    tracklets_for_removal = []
    spots_df["is_swapped"] = False

    for tracklet in tqdm(tracklets.sort_values(by=["end_time"], ascending=False).itertuples(), desc="merging tracklets"):

        tracklet_a = tracklet.Index
        # exclude tracklet id 0
        if tracklet_a == 0:
            continue

        print_all = False
        if tracklet_a == 372:
            print_all = True

        # get tracklet data
        t = tracklet.end_time
        tracklet_a_start = tracklet.start_time
        point_end_a = tracklet.end_id

        if print_all:
            print(f"")
            print(f"tracklet {tracklet_a} at time {t}")
            print(f"start time {tracklet_a_start}")
            print(f"end time {t}")

        # exclude time points that are too late
        if t > latest_time:
            continue

        # locate point a
        point_end_a_x = np.array(spots_df.loc[point_end_a, ["FRAME", "POSITION_X", "POSITION_Y", "POSITION_Z"]].values)
        point_end_a_x[0] = point_end_a_x[0]*max_dis*2

        # find nearest neighbor
        dd, ii = tree.query(point_end_a_x, 2)

        if print_all:
            print(dd, ii)
            print(f"{point_end_a_x}")
            print(f"nearest neighbor {ii[1]} at distance {dd[1]}")

        if dd[1] > max_dis:
            # no neighbor found near end of track
            continue

        # get tracklet corresponding to nearest neighbor
        point_end_b = spots_df.index[ii[1]]
        tracklet_b = spots_df.loc[point_end_b, "linear_track_id"]

        if pd.isna(tracklet_b):
            continue

        # tracklet b must have started after tracklet a
        if tracklets.loc[tracklet_b, "start_time"] < tracklet_a_start or tracklets.loc[tracklet_b, "end_time"] <= t:
            continue

        # get tracklet a and b points corresponding to the start of tracklet b
        point_start_b = tracklets.loc[tracklet_b, "start_id"]
        point_start_b_x = spots_df.loc[point_start_b, ["FRAME", "POSITION_X", "POSITION_Y", "POSITION_Z"]].values
        point_coincident_a = spots_df[(spots_df["FRAME"] == point_start_b_x[0]) & spots_df["linear_track_id"] == tracklet_a]
        point_coincident_a_x = point_coincident_a[["FRAME", "POSITION_X", "POSITION_Y", "POSITION_Z"]].values

        # tracklet b must have started close to tracklet a
        dis = np.linalg.norm(point_coincident_a_x - point_start_b_x)
        if dis > max_dis:
            continue

        # swap tracklet ids after time point
        outs = list(graph.neighbors(point_end_b))
        if len(outs) == 0:
            print("no children")
            continue

        assert len(outs) == 1, "tid has multiple children"
        child = outs[0]

        graph.add_edge(point_end_a, child, track_id=tracklet_a, time=1)
        graph.remove_edge(point_end_b, child)

        before = (spots_df["linear_track_id"] == tracklet_b) & (spots_df["FRAME"] <= t)
        after = spots_df[(spots_df["linear_track_id"] == tracklet_b) & (spots_df["FRAME"] > t)].index

        spots_df.loc[after, "linear_track_id"] = tracklet_a
        spots_df.loc[after, "is_swapped"] = True

        tracklets_for_removal.append(tracklet_b)
        n_changed += 1

    print(f"made {n_changed} swaps")
    for_removal = spots_df.index[spots_df["linear_track_id"].isin(tracklets_for_removal)]
    graph.remove_nodes_from(for_removal)
    spots_df = spots_df[~spots_df["linear_track_id"].isin(tracklets_for_removal)]

    return spots_df, graph


def interpolate_points(spots_df: pd.DataFrame, graph: DiGraph):
    """
    :param spots_df:
    :param graph:
    :return:
    """

    spots_df = spots_df.copy()
    spots_df["interpolated"] = False

    graph = graph.copy()

    new_spot_idx = spots_df.index.max() + 1

    for source, target, edge in tqdm(graph.copy().edges(data=True)):

        if edge["time"] == 1:
            continue

        source_spot = spots_df.loc[source]
        target_spot = spots_df.loc[target]

        source_spot_frame = source_spot["FRAME"]
        target_spot_frame = target_spot["FRAME"]

        source_spot_x = source_spot[["POSITION_Z", "POSITION_X", "POSITION_Y"]].values
        target_spot_x = target_spot[["POSITION_Z", "POSITION_X", "POSITION_Y"]].values

        interp = interp1d([source_spot_frame, target_spot_frame], [source_spot_x, target_spot_x], axis=0)

        new_edge_source_idx = source_spot["graph_key"]

        # interpolate intermediate spots, and connect them to the graph
        for t_offset in range(1, edge["time"]):
            t = source_spot_frame + t_offset
            new_spot = source_spot.copy()

            new_spot["FRAME"] = t
            new_spot[["POSITION_Z", "POSITION_X", "POSITION_Y"]] = interp(t)
            new_spot["graph_key"] = new_spot_idx
            new_spot["interpolated"] = True

            # adds a new edge to the graph (starts with source_spot
            graph.add_edge(new_edge_source_idx, new_spot_idx, track_id=edge["track_id"], time=1)
            spots_df.loc[new_spot_idx] = new_spot

            new_edge_source_idx = new_spot_idx
            new_spot_idx += 1

        # finish by connecting the last interpolated spot to the target
        graph.add_edge(new_edge_source_idx, target_spot["graph_key"], track_id=edge["track_id"], time=1)

        # then remove the original edge
        graph.remove_edge(source, target)

    return spots_df, graph


def detect_positional_outliers(spots):
    """
    Detects outliers according to x and y positions
    Uses DBSCAN and keeps only the largest cluster
    """
    if "POSITION_Z" in spots.columns:
        x = spots[["POSITION_X", "POSITION_Y", "POSITION_Z"]].values
    else:
        x = spots[["POSITION_X", "POSITION_Y"]].values
    dbscan = DBSCAN(eps=3, min_samples=1)
    return dbscan.fit_predict(x)


# def process_trackmate_tree(tree: ET) -> (pd.DataFrame, DiGraph):
#     """
#     Process trackmate tree
#     :param tree: ElementTree object from trackmate xml file
#     :return:
#     """
#
#     graph = DiGraph()
#     root = tree.getroot()
#
#     # iterate through spot elements and collect attributes
#     spots = root.find("Model").find("AllSpots")
#     spots_collect = []
#
#     for spot_frame in tqdm(spots.iterchildren(), desc="parsing spots; frame"):
#         for spot in spot_frame.iterchildren():
#             # spot id is always an int
#             spot_id = int(spot.get("ID"))
#             graph.add_node(spot_id)
#
#             # get all attributes and convert to floats
#             spot_attributes = spot.attrib
#             spot_attributes = {key: float(value) for key, value in spot_attributes.items() if key != "name"}
#
#             spot_attributes["graph_key"] = spot_id
#             spot_attributes["FRAME"] = int(spot_attributes["FRAME"])
#
#             # mostly used in 2d
#             if spot.text:
#                 spot_attributes["roi"] = [float(pt) for pt in spot.text.split(" ")]
#
#             spots_collect.append(spot_attributes)
#
#     # use graph key universally as an index
#     spots_df = pd.DataFrame(spots_collect, index=[c["graph_key"] for c in spots_collect])
#     assert np.all(spots_df.index == spots_df["ID"])
#
#     # iterate through track elements to construct graph and assign trackid
#     tracks = root.find("Model").find("AllTracks")
#     spot_tracks = {idx: 0 for idx in spots_df.index}
#
#     for i, track in enumerate(tqdm(tracks.iterchildren(), desc="parsing edges; track")):
#         track_id = i
#         for edge in track.iterchildren():
#             edge_attributes = edge.attrib
#
#             source_spot_id = int(edge_attributes["SPOT_SOURCE_ID"])
#             target_spot_id = int(edge_attributes["SPOT_TARGET_ID"])
#
#             source_spot_frame = int(spots_df.loc[source_spot_id]["FRAME"])
#             target_spot_frame = int(spots_df.loc[target_spot_id]["FRAME"])
#
#             # add edge to graph
#             graph.add_edge(int(edge_attributes["SPOT_SOURCE_ID"]),
#                            int(edge_attributes["SPOT_TARGET_ID"]),
#                            track_id=track_id,
#                            displacement=float(edge_attributes["DISPLACEMENT"]),
#                            time=target_spot_frame - source_spot_frame,)
#
#             # collect track id for each spot
#             spot_tracks[int(edge_attributes["SPOT_TARGET_ID"])] = track_id
#             spot_tracks[int(edge_attributes["SPOT_SOURCE_ID"])] = track_id
#
#     # assign track_id to each spot
#     spots_df["linear_track_id"] = [spot_tracks[idx] for idx in spots_df.index]
#     print(f"track id 0 corresponds to {np.sum(spots_df['linear_track_id'] == 0)} edgeless spots")
#
#     # remove positional outliers
#     print("starting positional outlier detection")
#     spots_df["position_cluster"] = detect_positional_outliers(spots_df)
#     print("completed positional outlier detection")
#
#     largest_cluster_index = spots_df.groupby("position_cluster").size().idxmax()
#     largest_cluster = spots_df.groupby("position_cluster").size().index[largest_cluster_index]
#     is_outlier = spots_df["position_cluster"] != largest_cluster
#
#     graph.remove_nodes_from(spots_df[is_outlier]["graph_key"])
#     spots_df = spots_df[~is_outlier]
#
#     return spots_df, graph


def process_trackmate_tree(tree: ET) -> (pd.DataFrame, DiGraph):
    """
    Process trackmate tree
    :param tree: ElementTree object from trackmate xml file
    :return:
    """

    graph = DiGraph()
    root = tree.getroot()

    # iterate through spot elements and collect attributes
    spots = root.find("Model").find("AllSpots")
    spots_collect = []

    for spot_frame in tqdm(spots.iterchildren(), desc="parsing spots; frame"):
        for spot in spot_frame.iterchildren():
            # spot id is always an int
            spot_id = int(spot.get("ID"))
            graph.add_node(spot_id)

            # get all attributes and convert to floats
            spot_attributes = spot.attrib
            spot_attributes = {key: float(value) for key, value in spot_attributes.items() if key != "name"}

            spot_attributes["graph_key"] = spot_id
            spot_attributes["FRAME"] = int(spot_attributes["FRAME"])

            # mostly used in 2d
            if spot.text:
                spot_attributes["roi"] = [float(pt) for pt in spot.text.split(" ")]

            spots_collect.append(spot_attributes)

    # use graph key universally as an index
    spots_df = pd.DataFrame(spots_collect, index=[c["graph_key"] for c in spots_collect])
    assert np.all(spots_df.index == spots_df["ID"])

    # iterate through track elements to construct graph and assign trackid
    tracks = root.find("Model").find("AllTracks")

    for i, track in enumerate(tqdm(tracks.iterchildren(), desc="parsing edges; track"), start=1):
        track_id = i

        this_track_spots = set()

        for edge in track.iterchildren():
            edge_attributes = edge.attrib

            source_spot_id = int(edge_attributes["SPOT_SOURCE_ID"])
            target_spot_id = int(edge_attributes["SPOT_TARGET_ID"])

            this_track_spots.add(source_spot_id)
            this_track_spots.add(target_spot_id)

        this_track_spots = list(this_track_spots)

        track_spots = spots_df.loc[this_track_spots].sort_values(by=["FRAME"]).index
        for source, target in zip(track_spots[:-1], track_spots[1:]):
            source_spot_frame = int(spots_df.loc[source]["FRAME"])
            target_spot_frame = int(spots_df.loc[target]["FRAME"])

            # add edge to graph
            graph.add_edge(source, target, track_id=track_id, time=target_spot_frame - source_spot_frame)

        spots_df.loc[this_track_spots, "linear_track_id"] = track_id

    print(f"track id 0 corresponds to {np.sum(spots_df['linear_track_id'].isna())} edgeless spots")

    # # remove positional outliers
    # print("starting positional outlier detection")
    # spots_df["position_cluster"] = detect_positional_outliers(spots_df)
    # print("completed positional outlier detection")
    #
    # largest_cluster_index = spots_df.groupby("position_cluster").size().idxmax()
    # largest_cluster = spots_df.groupby("position_cluster").size().index[largest_cluster_index]
    # is_outlier = spots_df["position_cluster"] != largest_cluster
    #
    # graph.remove_nodes_from(spots_df[is_outlier]["graph_key"])
    # spots_df = spots_df[~is_outlier]

    return spots_df, graph


def get_sister_distances(spots_df: pd.DataFrame, graph: DiGraph, tracklets: pd.DataFrame,
                         div_start, div_end, max_distance, extent_factor=1):
    """
    Compute sister distances from every sl tracklet to every fl tracklet
    returns (n_sl, n_fl) array of distances,
    (n_sl, 3) array of indices (index, closest_parent_index, closest_sister_index)
    """
    # get the tracklets that are present at the end of mitosis
    division_tracklets = tracklets[(tracklets["start_time"] < div_end) & (tracklets["end_time"] > div_end)]
    division_spots = spots_df[spots_df["linear_track_id"].isin(division_tracklets.index)]
    division_spots = division_spots[division_spots["FRAME"] < div_end]
    division_spots = division_spots[division_spots["FRAME"] > div_start].copy()
    division_spots["frame_rescaled"] = division_spots["FRAME"]*max_distance*2

    spots_df["frame_rescaled"] = spots_df["FRAME"]*max_distance*2

    # get the locations of the division spots
    division_spots_x = division_spots[["frame_rescaled", "POSITION_X", "POSITION_Y", "POSITION_Z"]].values
    tree = KDTree(division_spots_x)

    # subset the full length tracklets and the shorter length tracklets
    division_fl = division_tracklets[division_tracklets["start_time"] < div_start]
    division_fl = division_fl[division_fl.index != 0]
    division_sl = division_tracklets[division_tracklets["start_time"] > div_start]
    division_sl = division_sl[division_sl.index != 0]

    assert not division_fl.index.intersection(division_sl.index).any(), "fl and sl tracklets overlap"

    sl_start_x = division_spots.loc[division_sl["start_id"], ["frame_rescaled", "POSITION_X", "POSITION_Y", "POSITION_Z"]].values

    sl_children = []
    sl_parents = []
    sl_parent_tracklets = []
    sl_cost = []

    for spot_a, spot_a_x in tqdm(zip(division_sl["start_id"], sl_start_x), desc="computing sister distances"):

        dd, ii = tree.query(spot_a_x, 15, distance_upper_bound=max_distance)

        # print(f"outdegree of spot a{spot_a} is {graph.out_degree(spot_a)}")

        spot_a_next = list(graph.successors(spot_a))[0]
        spot_a_next_x = spots_df.loc[spot_a_next, ["frame_rescaled", "POSITION_X", "POSITION_Y", "POSITION_Z"]].values

        spot_a_parents = []
        spot_a_parent_tracklets = []
        spot_a_cost = []

        for i in range(len(ii)):

            if dd[i] == np.inf:
                continue

            spot_b = division_spots.index[ii[i]]
            spot_b_x = spots_df.loc[spot_b, ["frame_rescaled", "POSITION_X", "POSITION_Y", "POSITION_Z"]].values
            tracklet_b = spots_df.loc[spot_b, "linear_track_id"]

            if tracklet_b not in division_fl.index:
                continue

            # print(tracklet_b)
            #
            # print(f"in degree of spot b{spot_b} is {graph.in_degree(spot_b)}")
            # print(f"{tracklets.loc[tracklet_b, 'start_time']} < {div_start} < {div_end} < {tracklets.loc[tracklet_b, 'end_time']}")
            # print(spot_b in spots_df[spots_df["linear_track_id"] == tracklet_b]["graph_key"])

            spot_b_next = list(graph.successors(spot_b))[0]
            spot_b_prev = list(graph.predecessors(spot_b))[0]

            spot_b_next_x = spots_df.loc[spot_b_next, ["frame_rescaled", "POSITION_X", "POSITION_Y", "POSITION_Z"]].values
            spot_b_prev_x = spots_df.loc[spot_b_prev, ["frame_rescaled", "POSITION_X", "POSITION_Y", "POSITION_Z"]].values

            pred_spot_a_prev_x = spot_a_x - extent_factor*(spot_a_next_x - spot_a_x)
            pred_spot_b_prev_x = spot_b_x - extent_factor*(spot_b_next_x - spot_b_x)
            pred_joint_prev_x = (pred_spot_a_prev_x + pred_spot_b_prev_x)/2

            cost = np.linalg.norm((pred_joint_prev_x - spot_b_prev_x)[1:])

            spot_a_parents.append(spot_b_prev)
            spot_a_parent_tracklets.append(tracklet_b)
            spot_a_cost.append(cost)

        sl_parents.append(spot_a_parents)
        sl_parent_tracklets.append(spot_a_parent_tracklets)
        sl_cost.append(spot_a_cost)
        sl_children.append(spot_a)

    cost_matrix = np.ones((len(sl_children), len(division_fl.index))) * 10000
    parent_map = {}

    for i, child in tqdm(enumerate(sl_children), desc="constructing cost matrix"):
        parent_map[i] = {}
        for parent, parent_tracklet, cost in zip(sl_parents[i], sl_parent_tracklets[i], sl_cost[i]):

            j = division_fl.index.get_loc(parent_tracklet)
            cost_matrix[i, j] = cost
            parent_map[i][j] = parent

    return cost_matrix, parent_map, sl_children


def map_divisions(spots_df: pd.DataFrame, graph: DiGraph, interphase_dividers, new_track_cost=20) -> DiGraph:
    spots_df = spots_df.copy()
    graph = graph.copy()

    for start, end in zip(interphase_dividers[:-1], interphase_dividers[1:]):
        print(f"mapping divisions between {start} and {end}")
        tracklets = quick_tracklets(spots_df, column="linear_track_id")
        cost_matrix, parent_map, sl_children = get_sister_distances(spots_df, graph, tracklets, start, end, 25)

        # add no assignment cost
        n_in, n_out = cost_matrix.shape
        new_track_costs = np.ones((n_in, n_out)) * new_track_cost
        cost_matrix = np.hstack([cost_matrix, new_track_costs])

        # find the best matching
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        print(f"{np.sum(col_ind < n_out)} / {n_in} (max {min(n_in, n_out)}) new tracks assigned")

        for i, j in zip(row_ind, col_ind):
            if j >= n_out:
                continue

            child = sl_children[i]
            parent = parent_map[i][j]

            graph.add_edge(parent, child, track_id=0, time=1)

    return graph


def process_graph(spots_df: pd.DataFrame, graph: DiGraph) -> pd.DataFrame:
    """
    Assigns track_id, tracklet_id, parent_id, and daughter_id to spots_df based on graph structure

    taken as a postprocessing step after division detection
    """
    spots_df = spots_df.copy()
    undirected_graph = graph.copy().to_undirected()

    # assign track index as entire connected lineage of a tracked nucleus
    new_track_idx = {idx: 0 for idx in spots_df.index}

    cc = connected_components(graph.to_undirected())
    cc = [c for c in sorted(cc, key=len, reverse=True)]

    for track, c in enumerate(cc, start=1):
        for spot in c:
            new_track_idx[spot] = track

    spots_df["track_id"] = spots_df.index.map(new_track_idx)

    graph = graph.copy()

    # breaks the graph into tracklets, by removing edges after divisions
    parents = [node for node in graph.nodes if graph.out_degree(node) == 2]

    for parent in parents:
        children = list(graph.successors(parent))
        for child in children:
            graph.remove_edge(parent, child)

    # assigns tracklet index based on new graph
    new_tracklet_idx = {idx: 0 for idx in spots_df.index}

    for tracklet, c in enumerate(connected_components(graph.to_undirected()), start=1):
        for spot in c:
            new_tracklet_idx[spot] = tracklet

    print(f"number of tracklets detected: {tracklet}")
    spots_df["tracklet_id"] = spots_df.index.map(new_tracklet_idx)

    return spots_df
