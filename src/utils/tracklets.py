import pandas as pd
import numpy as np
from tqdm import tqdm
from pathlib import Path
from sklearn.cluster import KMeans

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


def ap_axis_position(spots, tracklets, anterior, posterior):
    x_a, y_a = anterior
    x_b, y_b = posterior

    x, y = spots["POSITION_X"] - x_a, spots["POSITION_Y"] - y_a
    x_ref, y_ref = x_b - x_a, y_b - y_a

    spots["ap_position"] = (x * x_ref + y * y_ref) / (x_ref ** 2 + y_ref ** 2)
    spots["edge_position"] = (x * y_ref - y * x_ref) / np.sqrt(x_ref ** 2 + y_ref ** 2)

    tracklets["mean_ap_position"] = tracklets.index.map(spots.groupby("TRACKLET_ID")["ap_position"].mean())
    tracklets["source_ap_position"] = tracklets.index.map(spots.groupby("TRACKLET_ID")["ap_position"].first())
    tracklets["sink_ap_position"] = tracklets.index.map(spots.groupby("TRACKLET_ID")["ap_position"].last())

    return spots, tracklets


def detect_nuclear_cycle(tracklets, n_clusters=3):
    # convert start times to time series
    t_s = np.arange(tracklets["start"].min(), tracklets["end"].max())
    time_series = np.histogram(tracklets["start"], bins=t_s)[0]

    peaks = [peak.born for peak in get_persistent_homology(time_series)][:n_clusters]

    kmeans = KMeans(n_clusters=n_clusters, init=np.array(peaks).reshape(-1, 1), verbose=0)
    tracklets["cycle"] = kmeans.fit_predict(tracklets[["start"]])

    # order cycle by mean start time
    order_mapping = tracklets.groupby("cycle")["start"].mean().sort_values().index
    tracklets["cycle"] = tracklets["cycle"].map({cycle: idx for idx, cycle in enumerate(order_mapping, 1)})

    tracklets["cycle_median_start_time"] = tracklets["cycle"].map(tracklets.groupby("cycle")["start_time"].median())
    tracklets["cycle_median_end_time"] = tracklets["cycle"].map(tracklets.groupby("cycle")["end_time"].median())

    tracklets["cycle_start_deviation"] = tracklets["start_time"] - tracklets["cycle_median_start_time"]
    tracklets["cycle_end_deviation"] = tracklets["end_time"] - tracklets["cycle_median_end_time"]

    return peaks, tracklets




def tracklet_from_path(tracking_path, root, t_delta=0.25, anterior=None, posterior=None):
    # processes cellpose input, with file names as below
    # import dfs and remove first three redundant rows
    spots = pd.read_csv(tracking_path / f"{root}_spots.csv", skiprows=[1, 2, 3])
    edges = pd.read_csv(tracking_path / f"{root}_edges.csv", skiprows=[1, 2, 3])
    tracks = pd.read_csv(tracking_path / f"{root}_tracks.csv", skiprows=[1, 2, 3])

    spots, tracklets = compute_tracklets(spots, edges, t_delta=t_delta)

    if anterior == "auto":
        anterior = anterior or (spots["POSITION_X"].min(), spots["POSITION_Y"].min())
        posterior = posterior or (spots["POSITION_X"].max(), spots["POSITION_Y"].max())

    if anterior and posterior:
        spots, tracklets = ap_axis_position(spots, tracklets, anterior, posterior)

    return spots, edges, tracks, tracklets
