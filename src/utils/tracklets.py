import pandas as pd
import numpy as np
from tqdm import tqdm
from pathlib import Path
from sklearn.cluster import KMeans, DBSCAN
from sklearn.linear_model import SGDOneClassSVM
from scipy.spatial import ConvexHull
from typing import Union
from json import load

from .peak_identification import get_persistent_homology


def compute_tracklets(spots: pd.DataFrame, edges: pd.DataFrame, t_delta=0.25):
    # count number of times each spots["ID"] appears in edges["SPOT_SOURCE_ID"]
    source_count = edges["SPOT_SOURCE_ID"].value_counts()
    spots["source_count"] = spots["ID"].map(source_count).fillna(0)

    # count number of times each spots["ID"] appears in edges["SPOT_TARGET_ID"]
    target_count = edges["SPOT_TARGET_ID"].value_counts()
    spots["target_count"] = spots["ID"].map(target_count).fillna(0)

    # determine sinks as spots that have two or zero children
    sinks = spots[spots["source_count"].isin([0, 2])]
    sinks_ids = sinks["ID"].values

    # initialize tracklets
    spot_tracklets = {spot: idx for idx, spot in enumerate(sinks["ID"], 1)}
    tracklet_parents = {idx: 0 for idx in spot_tracklets.values()}
    tracklet_sinks = spot_tracklets.copy()
    tracklet_sources = {}

    # make time numeric
    sorted_edges = edges.sort_values(by="EDGE_TIME", ascending=False)

    for source, target in tqdm(zip(sorted_edges["SPOT_SOURCE_ID"], sorted_edges["SPOT_TARGET_ID"])):
        tracklet = spot_tracklets[target]
        if source in sinks_ids:
            tracklet_parents[tracklet] = spot_tracklets[source]
            continue
        spot_tracklets[source] = tracklet
        tracklet_sources[tracklet] = source

    spots["TRACKLET_ID"] = spots["ID"].map(spot_tracklets)

    tracklets = pd.DataFrame.from_dict(tracklet_parents, orient="index", columns=["parent"])
    tracklets["start"] = tracklets.index.map(spots.groupby("TRACKLET_ID")["FRAME"].min())
    tracklets["end"] = tracklets.index.map(spots.groupby("TRACKLET_ID")["FRAME"].max())
    tracklets["length"] = (tracklets["end"] - tracklets["start"]) * t_delta
    tracklets["start_time"] = tracklets["start"] * t_delta
    tracklets["end_time"] = tracklets["end"] * t_delta
    tracklets["source"] = tracklets.index.map(tracklet_sources)
    tracklets["sink"] = tracklets.index.map(tracklet_sinks)

    tracklets["source_x"] = tracklets["source"].map(spots.groupby("ID")["POSITION_X"].first())
    tracklets["source_y"] = tracklets["source"].map(spots.groupby("ID")["POSITION_Y"].first())
    tracklets["sink_x"] = tracklets["sink"].map(spots.groupby("ID")["POSITION_X"].last())
    tracklets["sink_y"] = tracklets["sink"].map(spots.groupby("ID")["POSITION_Y"].last())

    tracklets["track_id"] = tracklets.index.map(spots.groupby("TRACKLET_ID")["TRACK_ID"].first())

    spots["time"] = spots["FRAME"] * t_delta

    return spots, tracklets


def old_ap_axis_position(spots, tracklets, anterior, posterior):
    x_a, y_a = anterior
    x_b, y_b = posterior

    x, y = spots["POSITION_X"] - x_a, spots["POSITION_Y"] - y_a
    x_ref, y_ref = x_b - x_a, y_b - y_a

    spots["ap_position"] = (x * x_ref + y * y_ref) / (x_ref ** 2 + y_ref ** 2)
    spots["edge_position"] = (x * y_ref - y * x_ref) / np.sqrt(x_ref ** 2 + y_ref ** 2)

    tracklets["mean_ap_position"] = tracklets.index.map(spots.groupby("TRACKLET_ID")["ap_position"].mean())
    tracklets["source_ap_position"] = tracklets.index.map(spots.groupby("TRACKLET_ID")["ap_position"].first())
    tracklets["sink_ap_position"] = tracklets.index.map(spots.groupby("TRACKLET_ID")["ap_position"].last())

    tracklets["mean_edge_distance"] = tracklets.index.map(spots.groupby("TRACKLET_ID")["distance_to_edge"].mean())
    tracklets["source_edge_distance"] = tracklets.index.map(spots.groupby("TRACKLET_ID")["distance_to_edge"].first())
    tracklets["sink_edge_distance"] = tracklets.index.map(spots.groupby("TRACKLET_ID")["distance_to_edge"].last())

    return spots, tracklets


def ap_axis_position(spots, anterior, posterior):
    x_a, y_a = anterior
    x_b, y_b = posterior

    x, y = spots["POSITION_X"] - x_a, spots["POSITION_Y"] - y_a
    x_ref, y_ref = x_b - x_a, y_b - y_a

    spots["ap_position"] = (x * x_ref + y * y_ref) / (x_ref ** 2 + y_ref ** 2)
    spots["edge_position"] = (x * y_ref - y * x_ref) / (x_ref ** 2 + y_ref ** 2)

    return spots


def identify_peaks(start_times):
    t_s = np.arange(start_times.min(), start_times.max())
    time_series = np.histogram(start_times, bins=t_s)[0]

    return [peak.born for peak in get_persistent_homology(time_series)]


def detect_nuclear_cycle(tracklets, n_clusters=3):
    # determine time series peaks
    peaks = identify_peaks(tracklets["start"])[:n_clusters]

    kmeans = KMeans(n_clusters=n_clusters, init=np.array(peaks).reshape(-1, 1), verbose=0)
    tracklets["cycle"] = kmeans.fit_predict(tracklets[["start"]])

    # order cycle by mean start time
    order_mapping = tracklets.groupby("cycle")["start"].mean().sort_values().index
    tracklets["cycle"] = tracklets["cycle"].map({cycle: idx for idx, cycle in enumerate(order_mapping, 1)})

    tracklets["cycle_median_start_time"] = tracklets["cycle"].map(tracklets.groupby("cycle")["start_time"].median())
    tracklets["cycle_median_end_time"] = tracklets["cycle"].map(tracklets.groupby("cycle")["end_time"].median())

    tracklets["cycle_start_deviation"] = tracklets["start_time"] - tracklets["cycle_median_start_time"]
    tracklets["cycle_end_deviation"] = tracklets["end_time"] - tracklets["cycle_median_end_time"]

    tracklets = detect_nc_outliers(tracklets)

    return peaks, tracklets


def sister_tracklets(tracklets):
    """
    Computes features related to sister / daughter tracklets, such as division orientation
    """

    tracklets["source_div_angle"] = [None for _ in range(len(tracklets))]
    tracklets["sink_div_angle"] = [None for _ in range(len(tracklets))]

    for tracklet in tracklets.index:
        daughters = tracklets[tracklets["parent"] == tracklet].index
        if len(daughters) == 2:
            division_orientation = np.arctan2(
                tracklets.loc[daughters[0], "source_y"] - tracklets.loc[daughters[1], "source_y"],
                tracklets.loc[daughters[0], "source_x"] - tracklets.loc[daughters[1], "source_x"]
            ) % np.pi
            tracklets.loc[tracklet, "sink_div_angle"] = division_orientation
            tracklets.loc[daughters[0], "source_div_angle"] = division_orientation
            tracklets.loc[daughters[1], "source_div_angle"] = division_orientation

    so, si = tracklets.source_div_angle, tracklets.sink_div_angle
    tracklets["division_angle_difference"] = np.min([np.abs(so - si), np.abs(np.abs(so - si)-np.pi)], axis=0)

    return tracklets


def detect_positional_outliers(spots):
    """
    Detects outliers according to x and y positions
    Uses DBSCAN and keeps only the largest cluster
    """
    x = spots[["POSITION_X", "POSITION_Y"]].values
    dbscan = DBSCAN(eps=40, min_samples=1)
    return dbscan.fit_predict(x)


def compute_edge_distance(points: Union[pd.DataFrame, np.array], x_label="POSITION_X", y_label="POSITION_Y"):
    """
    Computes the distance of each point to the convex hull of the points
    :param points: [n, 2] array of points or pd.DataFrame of points
    :param x_label: column names if points is a pd.DataFrame
    :param y_label: column names if points is a pd.DataFrame
    :return: [n,] array of distances from convex hull
    """

    if isinstance(points, pd.DataFrame):
        points = points[[x_label, y_label]].values

    hull = ConvexHull(points)
    return distance_to_hull(hull, points)


def distances_to_line(points, p1, p2):
    """
    points: (n, 2) array of points
    p1, p2: (2,) array of points defining the line
    """
    x1, y1 = p1
    x2, y2 = p2
    x0, y0 = points[:, 0], points[:, 1]

    return np.abs((x2 - x1) * (y1 - y0) - (x1 - x0) * (y2 - y1)) / np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def distance_to_hull(hull: ConvexHull, points):
    n = len(hull.vertices)
    ds = []
    for i in range(n):
        p1, p2 = hull.vertices[(i - 1) % n], hull.vertices[i]
        ds.append(distances_to_line(points, hull.points[p1], hull.points[p2]))

    return np.min(ds, axis=0)


def tracklet_from_path(tracking_path, root, t_delta=0.25, anterior=None, posterior=None):
    # processes cellpose input, with file names as below
    # import dfs and remove first three redundant rows
    spots = pd.read_csv(tracking_path / f"{root}_spots.csv", skiprows=[1, 2, 3])
    edges = pd.read_csv(tracking_path / f"{root}_edges.csv", skiprows=[1, 2, 3])
    tracks = pd.read_csv(tracking_path / f"{root}_tracks.csv", skiprows=[1, 2, 3])

    spots["distance_to_edge"] = compute_edge_distance(spots)

    spots, tracklets = compute_tracklets(spots, edges, t_delta=t_delta)

    if anterior == "auto":
        anterior = anterior or (spots["POSITION_X"].min(), spots["POSITION_Y"].min())
        posterior = posterior or (spots["POSITION_X"].max(), spots["POSITION_Y"].max())

    if anterior and posterior:
        spots, tracklets = old_ap_axis_position(spots, tracklets, anterior, posterior)

    return spots, edges, tracks, tracklets


def detect_nc_outliers(tracklets):
    """
    Detects outliers in nuclear cycle tracklets, by nuclear cycle
    """
    outliers = []

    for cycle in tracklets["cycle"].unique():
        cycle_tracklets = tracklets[tracklets["cycle"] == cycle]

        # outliers have start or end deviation greater than 1.5 times the median absolute deviation
        start_deviation = cycle_tracklets["cycle_start_deviation"]
        end_deviation = cycle_tracklets["cycle_end_deviation"]

        start_outliers = np.abs(start_deviation) > 5 * np.median(np.abs(start_deviation))
        end_outliers = np.abs(end_deviation) > 5 * np.median(np.abs(end_deviation))

        outliers.extend(cycle_tracklets[start_outliers | end_outliers].index)

    tracklets["nc_outlier"] = tracklets.index.isin(outliers)

    return tracklets


def import_tracklets(datapath, roots):
    spots = {}
    tracklets = {}
    metadatas = {}

    for root in roots:
        spots_df = pd.read_csv(datapath / root / f"{root}_spots.csv")

        with open(datapath / root / f"{root}_metadata.json") as f:
            metadata = load(f)
        division_times = np.array(metadata["division_times"])
        group = spots_df.sort_values(by="FRAME").groupby("tracklet_id")
        cols = {
            "start_time": group["time"].min(),
            "end_time": group["time"].max(),
            "start_frame": group["FRAME"].min(),
            "end_frame": group["FRAME"].max(),
            "length": group["time"].max() - group["time"].min(),
            "source_spot": group["ID"].first(),
            "sink_spot": group["ID"].last(),
            "mean_ap_position": group["ap_position"].mean(),
            "source_ap_position": group["ap_position"].first(),
            "sink_ap_position": group["ap_position"].last(),
            "initial_x": group["POSITION_X"].first(),
            "initial_y": group["POSITION_Y"].first(),
            "final_x": group["POSITION_X"].last(),
            "final_y": group["POSITION_Y"].last(),
            "initial_x_um": group["um_x"].first(),
            "initial_y_um": group["um_y"].first(),
            "final_x_um": group["um_x"].last(),
            "final_y_um": group["um_y"].last(),
            "track_id": group["track_id"].first(),
            "mean_edge_distance": group["um_from_edge"].mean(),
        }
        tracklets_df = pd.DataFrame(cols)
        tracklets_df["track_n_tracklets"] = tracklets_df["track_id"].map(
            tracklets_df["track_id"].value_counts()
        )
        spots_df["track_n_tracklets"] = spots_df["tracklet_id"].map(tracklets_df["track_n_tracklets"])
        # map each tracklet start time to the nearest division time
        tracklets_df["cycle"] = tracklets_df["start_frame"].apply(
            lambda x: np.argmin(np.abs(division_times - x)) + 10
        )
        spots_df["cycle"] = spots_df["tracklet_id"].map(tracklets_df["cycle"])
        tracklets_df["embryo"] = root
        tracklets_df["tracklet_id"] = tracklets_df.index

        spots_df = ap_axis_position(spots_df,[metadata["a_x"], metadata["a_y"]], [metadata["p_x"], metadata["p_y"]])

        sink_to_tracklet = {idx: tracklet for tracklet, idx in tracklets_df["sink_spot"].items()}
        sink_to_tracklet[0] = -1
        id_to_parent = {idx: parent for idx, parent in zip(spots_df["ID"], spots_df["parent_id"])}

        tracklets_df["parent_tracklet"] = tracklets_df["source_spot"].map(id_to_parent).map(sink_to_tracklet)
        tracklets_df["parent_tracklet"] = tracklets_df["parent_tracklet"].fillna(0).astype(int)
        tracklets_df["n_children"] = tracklets_df["tracklet_id"].map(
            tracklets_df["parent_tracklet"].value_counts()
        ).fillna(0).astype(int)
        tracklets_df["e_id"] = [f"{root}_{idx}" for idx in tracklets_df.index]
        tracklets_df["e_parent_id"] = [f"{root}_{idx}" for idx in tracklets_df["parent_tracklet"]]

        tracklets[root] = tracklets_df
        spots[root] = spots_df
        metadatas[root] = metadata

    tracklets_joined = pd.concat(tracklets.values(), ignore_index=True)
    tracklets_joined.set_index("e_id", inplace=True)
    return spots, tracklets, metadatas, tracklets_joined
