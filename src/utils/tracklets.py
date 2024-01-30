import pandas as pd
from tqdm import tqdm
from pathlib import Path


def compute_tracklets(spots: pd.DataFrame, edges: pd.DataFrame):
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

    # make time numeric
    sorted_edges = edges.sort_values(by="EDGE_TIME", ascending=False)

    for source, target in tqdm(zip(sorted_edges["SPOT_SOURCE_ID"], sorted_edges["SPOT_TARGET_ID"])):
        if source in sinks_ids:
            tracklet_parents[spot_tracklets[target]] = spot_tracklets[source]
            continue
        spot_tracklets[source] = spot_tracklets[target]
    spots["TRACKLET_ID"] = spots["ID"].map(spot_tracklets)

    tracklets = pd.DataFrame.from_dict(tracklet_parents, orient="index", columns=["parent"])
    tracklets["start"] = tracklets.index.map(spots.groupby("TRACKLET_ID")["FRAME"].min())
    tracklets["end"] = tracklets.index.map(spots.groupby("TRACKLET_ID")["FRAME"].max())

    return spots, tracklets


def tracklet_from_path(tracking_path, root):
    # processes cellpose input, with file names as below
    # import dfs and remove first three redundant rows
    spots = pd.read_csv(tracking_path / f"{root}_spots.csv", skiprows=[1, 2, 3])
    edges = pd.read_csv(tracking_path / f"{root}_edges.csv", skiprows=[1, 2, 3])
    tracks = pd.read_csv(tracking_path / f"{root}_tracks.csv", skiprows=[1, 2, 3])

    spots, tracklets = compute_tracklets(spots, edges)

    return spots, edges, tracks, tracklets
