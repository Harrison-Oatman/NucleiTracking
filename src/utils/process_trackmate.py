import pandas as pd

from tqdm import tqdm
from xml.etree import ElementTree as ET
from networkx import Graph, DiGraph
from . tracklets import detect_positional_outliers, compute_edge_distance


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

    for spot_frame in tqdm(spots.iterchildren()):
        for spot in spot_frame.iterchildren():
            spot_id = spot.get("ID")
            spot_attributes = spot.attrib
            spot_attributes = {key: float(value) for key, value in spot_attributes.items() if key != "name"}
            spot_attributes["ID"] = spot_id
            spot_attributes["roi"] = [float(pt) for pt in spot.text.split(" ")]
            spots_collect.append(spot_attributes)

    spots_df = pd.DataFrame(spots_collect)

    # iterate through track elements to construct graph and assign trackid
    tracks = root.find("Model").find("AllTracks")
    spot_tracks = {idx: 0 for idx in spots_df["ID"]}

    for track in tqdm(tracks.iterchildren()):
        track_id = track.get("TRACK_ID")
        for edge in track.iterchildren():
            edge_attributes = edge.attrib
            graph.add_edge(edge_attributes["SPOT_SOURCE_ID"], edge_attributes["SPOT_TARGET_ID"], track_id=track_id)
            spot_tracks[str(int(edge_attributes["SPOT_TARGET_ID"]))] = track_id
            spot_tracks[str(int(edge_attributes["SPOT_SOURCE_ID"]))] = track_id

    spots_df["track_id"] = [spot_tracks[idx] for idx in spots_df["ID"]]

    # remove positional outliers
    spots_df["position_cluster"] = detect_positional_outliers(spots_df)
    largest_cluster = spots_df.groupby("position_cluster").size().idxmax()
    spots_df = spots_df[spots_df["position_cluster"] == largest_cluster]

    spots_df["distance_from_edge"] = compute_edge_distance(spots_df)

    return spots_df, graph
