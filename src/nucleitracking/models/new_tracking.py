import itertools
from collections import defaultdict
from xml.etree import ElementTree as ET

import numpy as np
import pandas as pd
from networkx import DiGraph, connected_components
from scipy.interpolate import interp1d
from scipy.optimize import linear_sum_assignment
from scipy.spatial import KDTree
from sklearn.cluster import DBSCAN
from tqdm import tqdm

### rules:
# spot id is always an integer


def quick_tracklets(spots_df, column="track_id") -> pd.DataFrame:
    spots_df = spots_df.sort_values(by=["frame"])

    start_times = spots_df.groupby(column)["frame"].min()
    end_times = spots_df.groupby(column)["frame"].max()
    start_id = spots_df.groupby(column)["graph_key"].first()
    end_id = spots_df.groupby(column)["graph_key"].last()

    tracklets = pd.DataFrame(
        {
            "start_time": start_times,
            "end_time": end_times,
            "start_id": start_id,
            "end_id": end_id,
        }
    )

    return tracklets


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

        source_spot_frame = source_spot["frame"]
        target_spot_frame = target_spot["frame"]

        source_spot_x = source_spot[["px_z", "px_x", "px_y"]].values
        target_spot_x = target_spot[["px_z", "px_x", "px_y"]].values

        interp = interp1d(
            [source_spot_frame, target_spot_frame],
            [source_spot_x, target_spot_x],
            axis=0,
        )

        new_edge_source_idx = source_spot["graph_key"]

        # interpolate intermediate spots, and connect them to the graph
        for t_offset in range(1, edge["time"]):
            t = source_spot_frame + t_offset
            new_spot = source_spot.copy()

            new_spot["frame"] = t
            new_spot[["px_z", "px_x", "px_y"]] = interp(t)
            new_spot["graph_key"] = new_spot_idx
            new_spot["ID"] = new_spot_idx
            new_spot["interpolated"] = True

            # print(new_edge_source_idx, new_spot_idx)

            # adds a new edge to the graph (starts with source_spot
            graph.add_edge(
                new_edge_source_idx, new_spot_idx, track_id=edge["track_id"], time=1
            )
            spots_df.loc[new_spot_idx] = new_spot

            new_edge_source_idx = new_spot_idx
            new_spot_idx += 1

        # finish by connecting the last interpolated spot to the target
        graph.add_edge(
            new_edge_source_idx,
            target_spot["graph_key"],
            track_id=edge["track_id"],
            time=1,
        )

        # then remove the original edge
        graph.remove_edge(source, target)

    return spots_df, graph


def detect_positional_outliers(spots):
    """
    Detects outliers according to x and y positions
    Uses DBSCAN and keeps only the largest cluster
    """
    if "px_z" in spots.columns:
        x = spots[["px_x", "px_y", "px_z"]].values
    else:
        x = spots[["px_x", "px_y"]].values
    dbscan = DBSCAN(eps=3, min_samples=1)
    return dbscan.fit_predict(x)


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
            spot_attributes = {
                key: float(value)
                for key, value in spot_attributes.items()
                if key != "name"
            }

            spot_attributes["graph_key"] = spot_id
            spot_attributes["frame"] = int(spot_attributes["frame"])

            # mostly used in 2d
            if spot.text:
                spot_attributes["roi"] = [float(pt) for pt in spot.text.split(" ")]

            spots_collect.append(spot_attributes)

    # use graph key universally as an index
    spots_df = pd.DataFrame(
        spots_collect, index=[c["graph_key"] for c in spots_collect]
    )
    spots_df["ID"] = spots_df["ID"].astype(int)

    assert np.all(spots_df.index == spots_df["ID"])

    # iterate through track elements to construct graph and assign trackid
    tracks = root.find("Model").find("AllTracks")

    for i, track in enumerate(
        tqdm(tracks.iterchildren(), desc="parsing edges; track"), start=1
    ):
        track_id = i

        this_track_spots = set()

        for edge in track.iterchildren():
            edge_attributes = edge.attrib

            source_spot_id = int(edge_attributes["SPOT_SOURCE_ID"])
            target_spot_id = int(edge_attributes["SPOT_TARGET_ID"])

            this_track_spots.add(source_spot_id)
            this_track_spots.add(target_spot_id)

        this_track_spots = list(this_track_spots)

        track_spots = spots_df.loc[this_track_spots].sort_values(by=["frame"]).index
        for source, target in itertools.pairwise(track_spots):
            source_spot_frame = int(spots_df.loc[source]["frame"])
            target_spot_frame = int(spots_df.loc[target]["frame"])

            # add edge to graph
            graph.add_edge(
                source,
                target,
                track_id=track_id,
                time=target_spot_frame - source_spot_frame,
            )

        spots_df.loc[this_track_spots, "linear_track_id"] = track_id

    spots_df["linear_track_id"] = spots_df["linear_track_id"].fillna(-1)
    spots_df["linear_track_id"] = spots_df["linear_track_id"].astype(int)

    print(
        f"track id -1 corresponds to {np.sum(spots_df['linear_track_id'].isna())} edgeless spots"
    )

    return spots_df, graph


def merge_close_tracklets(spots_df: pd.DataFrame, graph: DiGraph, max_distance=10):
    raise NotImplementedError


def get_sister_distances(
    spots_df: pd.DataFrame,
    graph: DiGraph,
    tracklets: pd.DataFrame,
    div_start,
    div_end,
    max_distance,
    extent_factor=1,
):
    """
    Compute sister distances from every sl tracklet to every fl tracklet
    returns (n_sl, n_fl) array of distances,
    (n_sl, 3) array of indices (index, closest_parent_index, closest_sister_index)
    """
    # get the tracklets that are present at the end of mitosis
    division_tracklets = tracklets[
        (tracklets["start_time"] < div_end) & (tracklets["end_time"] > div_end)
    ]
    division_spots = spots_df[
        spots_df["linear_track_id"].isin(division_tracklets.index)
    ]
    division_spots = division_spots[division_spots["frame"] < div_end]
    division_spots = division_spots[division_spots["frame"] > div_start].copy()
    division_spots["frame_rescaled"] = division_spots["frame"] * max_distance * 2

    spots_df["frame_rescaled"] = spots_df["frame"] * max_distance * 2

    # get the locations of the division spots
    division_spots_x = division_spots[["frame_rescaled", "px_x", "px_y", "px_z"]].values
    tree = KDTree(division_spots_x)

    # subset the full length tracklets and the shorter length tracklets
    division_fl = division_tracklets[division_tracklets["start_time"] < div_start]
    division_fl = division_fl[division_fl.index != 0]
    division_sl = division_tracklets[division_tracklets["start_time"] > div_start]
    division_sl = division_sl[division_sl.index != 0]

    assert not division_fl.index.intersection(division_sl.index).any(), (
        "fl and sl tracklets overlap"
    )

    sl_start_x = division_spots.loc[
        division_sl["start_id"],
        ["frame_rescaled", "px_x", "px_y", "px_z"],
    ].values

    sl_children = []
    sl_parents = []
    sl_parent_tracklets = []
    sl_cost = []

    for spot_a, spot_a_x in tqdm(
        zip(division_sl["start_id"], sl_start_x, strict=False),
        desc="computing sister distances",
    ):
        dd, ii = tree.query(spot_a_x, 15, distance_upper_bound=max_distance)

        # print(f"outdegree of spot a{spot_a} is {graph.out_degree(spot_a)}")

        spot_a_next = next(iter(graph.successors(spot_a)))
        spot_a_next_x = spots_df.loc[
            spot_a_next, ["frame_rescaled", "px_x", "px_y", "px_z"]
        ].values

        spot_a_parents = []
        spot_a_parent_tracklets = []
        spot_a_cost = []

        for i in range(len(ii)):
            if dd[i] == np.inf:
                continue

            spot_b = division_spots.index[ii[i]]
            spot_b_x = spots_df.loc[
                spot_b, ["frame_rescaled", "px_x", "px_y", "px_z"]
            ].values
            tracklet_b = spots_df.loc[spot_b, "linear_track_id"]

            if tracklet_b not in division_fl.index:
                continue

            # print(tracklet_b)
            #
            # print(f"in degree of spot b{spot_b} is {graph.in_degree(spot_b)}")
            # print(f"{tracklets.loc[tracklet_b, 'start_time']} < {div_start} < {div_end} < {tracklets.loc[tracklet_b, 'end_time']}")
            # print(spot_b in spots_df[spots_df["linear_track_id"] == tracklet_b]["graph_key"])

            if graph.out_degree(spot_b) != 1:
                continue

            spot_b_next = next(iter(graph.successors(spot_b)))
            spot_b_prev = next(iter(graph.predecessors(spot_b)))

            spot_b_next_x = spots_df.loc[
                spot_b_next,
                ["frame_rescaled", "px_x", "px_y", "px_z"],
            ].values
            spot_b_prev_x = spots_df.loc[
                spot_b_prev,
                ["frame_rescaled", "px_x", "px_y", "px_z"],
            ].values

            pred_spot_a_prev_x = spot_a_x - extent_factor * (spot_a_next_x - spot_a_x)
            pred_spot_b_prev_x = spot_b_x - extent_factor * (spot_b_next_x - spot_b_x)
            pred_joint_prev_x = (pred_spot_a_prev_x + pred_spot_b_prev_x) / 2

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

    flp = division_spots[
        division_spots["linear_track_id"].isin(division_fl.index)
    ].index
    slp = division_spots[
        division_spots["linear_track_id"].isin(division_sl.index)
    ].index

    spots_df.loc[flp, "status"] = 1
    spots_df.loc[slp, "status"] = 2

    for i, child in tqdm(enumerate(sl_children), desc="constructing cost matrix"):
        parent_map[i] = {}
        for parent, parent_tracklet, cost in zip(
            sl_parents[i], sl_parent_tracklets[i], sl_cost[i], strict=False
        ):
            j = division_fl.index.get_loc(parent_tracklet)
            cost_matrix[i, j] = cost
            parent_map[i][j] = parent

    return cost_matrix, parent_map, sl_children, spots_df


def map_divisions(
    spots_df: pd.DataFrame, graph: DiGraph, interphase_dividers, new_track_cost=20
):
    spots_df = spots_df.copy()
    spots_df["status"] = 0
    graph = graph.copy()

    for start, end in itertools.pairwise(interphase_dividers):
        print(f"mapping divisions between {start} and {end}")
        tracklets = quick_tracklets(spots_df, column="linear_track_id")
        cost_matrix, parent_map, sl_children, spots_df = get_sister_distances(
            spots_df, graph, tracklets, start, end, 25
        )

        # add no assignment cost
        n_in, n_out = cost_matrix.shape
        new_track_costs = np.ones((n_in, n_in)) * new_track_cost
        cost_matrix = np.hstack([cost_matrix, new_track_costs])

        # find the best matching
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        print(
            f"{np.sum(col_ind < n_out)} / {n_in} (max {min(n_in, n_out)}) new tracks assigned"
        )

        for i, j in zip(row_ind, col_ind, strict=False):
            if j >= n_out:
                continue

            child = sl_children[i]
            parent = parent_map[i][j]

            graph.add_edge(parent, child, track_id=0, time=1)

    return graph, spots_df


def process_graph(spots_df: pd.DataFrame, graph: DiGraph) -> pd.DataFrame:
    """
    Assigns track_id, tracklet_id, parent_id, and daughter_id to spots_df based on graph structure

    taken as a postprocessing step after division detection
    """
    spots_df = spots_df.copy()
    graph.copy().to_undirected()

    # assign track index as entire connected lineage of a tracked nucleus
    new_track_idx = dict.fromkeys(spots_df.index, 0)

    cc = connected_components(graph.to_undirected())
    cc = sorted(cc, key=len, reverse=True)

    for track, c in enumerate(cc, start=1):
        for spot in c:
            new_track_idx[spot] = track

    spots_df["track_id"] = spots_df.index.map(new_track_idx)

    broken_graph = graph.copy()

    # breaks the graph into tracklets, by removing edges after divisions
    parents = [
        node for node in broken_graph.nodes if broken_graph.out_degree(node) == 2
    ]

    for parent in parents:
        children = list(broken_graph.successors(parent))
        for child in children:
            broken_graph.remove_edge(parent, child)

    # assigns tracklet index based on new graph
    new_tracklet_idx = dict.fromkeys(spots_df.index, 0)
    undirected_ccs = connected_components(broken_graph.to_undirected())

    for tracklet, c in enumerate(undirected_ccs, start=1):
        for spot in c:
            new_tracklet_idx[spot] = tracklet

    parent_map = defaultdict(
        lambda: -1, {child: parent for parent, child in graph.edges()}
    )
    n_children = defaultdict(
        lambda: -1, {spot: graph.out_degree(spot) for spot in graph.nodes()}
    )
    n_parents = defaultdict(
        lambda: -1, {spot: graph.in_degree(spot) for spot in graph.nodes()}
    )

    spots_df = spots_df.sort_values(by=["frame"])

    print(f"number of tracklets detected: {tracklet}")
    spots_df["tracklet_id"] = spots_df.index.map(new_tracklet_idx)
    spots_df["parent_id"] = spots_df["graph_key"].map(parent_map)
    spots_df["n_children"] = spots_df["graph_key"].map(n_children)
    spots_df["n_parents"] = spots_df["graph_key"].map(n_parents)
    print(spots_df.groupby("tracklet_id")["parent_id"].first(skipna=False))
    spots_df["tracklet_first_parent_id"] = spots_df["tracklet_id"].map(
        spots_df.groupby("tracklet_id")["parent_id"].first(skipna=False)
    )
    spots_df["tracklet_parent_tracklet"] = spots_df["tracklet_first_parent_id"].map(
        new_tracklet_idx
    )

    return spots_df
