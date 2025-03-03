import pandas as pd
import networkx as nx
import numpy as np

from argparse import ArgumentParser
from pathlib import Path
from lxml import etree
from skimage.draw import polygon
from tqdm import tqdm
from tifffile import imwrite
from json import load, dump

from src.utils.process_trackmate import process_trackmate_tree
from src.utils.tracklets import compute_edge_distance, identify_peaks
from src.models.division_tracking import map_divisions


def make_lineage_tif(spots_df: pd.DataFrame, h=1360, w=1360) -> np.ndarray:
    shape = (round(spots_df["FRAME"].max()) + 1, h, w)
    output_tif = np.zeros(shape, dtype=np.uint16)

    for i, spot in tqdm(spots_df.iterrows()):
        x, y = spot["POSITION_X"], spot["POSITION_Y"]
        t = round(spot["FRAME"])
        new_track_id = spot["track_id"]

        xs = [round(pt + x) for pt in spot["roi"][::2]]
        ys = [round(pt + y) for pt in spot["roi"][1::2]]

        rr, cc = polygon(ys, xs, shape[1:])
        output_tif[t, rr, cc] = new_track_id + 1

    return output_tif


def process_graph(spots_df: pd.DataFrame, graph: nx.DiGraph) -> pd.DataFrame:
    """
    Assigns track_id, tracklet_id, parent_id, and daughter_id to spots_df based on graph structure

    taken as a postprocessing step after division detection
    """
    spots_df = spots_df.copy()

    # assign track index as entire connected lineage of a tracked nucleus
    new_track_idx = {idx: 0 for idx in spots_df["ID"]}

    cc = nx.connected_components(graph.to_undirected())
    cc = [c for c in sorted(cc, key=len, reverse=True)]

    for track, c in enumerate(cc, start=1):
        for spot in c:
            new_track_idx[spot] = track

    spots_df["track_id"] = [new_track_idx[idx] for idx in spots_df["ID"]]

    graph = graph.to_directed()

    # breaks the graph into tracklets, by removing edges after divisions
    parents = [node for node in graph.nodes if graph.out_degree(node) == 2]
    graph_copy = graph.copy()
    for parent in parents:
        children = list(graph.successors(parent))
        for child in children:
            graph_copy.remove_edge(parent, child)

    # assigns tracklet index based on new graph
    new_tracklet_idx = {idx: 0 for idx in spots_df["ID"]}
    tracklet = 0

    for tracklet, c in enumerate(nx.connected_components(graph_copy.to_undirected())):
        for spot in c:
            new_tracklet_idx[spot] = tracklet

    print(f"number of tracklets detected: {tracklet}")
    spots_df["tracklet_id"] = [new_tracklet_idx[idx] for idx in spots_df["ID"]]

    # collect graph edges in csv
    spot_parents = {idx: 0 for idx in spots_df["ID"]}
    spot_daughters = {idx: 0 for idx in spots_df["ID"]}

    for edge in graph.edges:
        source, target = edge
        spot_parents[target] = source
        spot_daughters[source] = target

    spots_df["parent_id"] = [spot_parents[idx] for idx in spots_df["ID"]]
    spots_df["daughter_id"] = [spot_daughters[idx] for idx in spots_df["ID"]]

    return spots_df


def prep_spots_df(spots_df: pd.DataFrame, metadata: dict) -> pd.DataFrame:

    kept_columns = ["ID", "track_id", "tracklet_id", "distance_from_edge", "parent_id", "daughter_id", "roi",
                    "FRAME", "POSITION_X", "POSITION_Y", "POSITION_Z",
                    "ELLIPSE_MAJOR", "ELLIPSE_MINOR", "ELLIPSE_THETA", "ELLIPSE_Y0", "ELLIPSE_X0",
                    "ELLIPSE_ASPECTRATIO",
                    "CIRCULARITY", "AREA", "SHAPE_INDEX", "MEDIAN_INTENSITY_CH1"]

    spots_df = spots_df[kept_columns].copy()

    spots_df["time"] = spots_df["FRAME"] / metadata["frames_per_minute"]
    spots_df["um_from_edge"] = spots_df["distance_from_edge"] / metadata["pixels_per_um"]
    spots_df["um_x"] = spots_df["POSITION_X"] / metadata["pixels_per_um"]
    spots_df["um_y"] = spots_df["POSITION_Y"] / metadata["pixels_per_um"]

    x_a, y_a = metadata["a_x"], metadata["a_y"]
    x_b, y_b = metadata["p_x"], metadata["p_y"]

    x, y = spots_df["POSITION_X"] - x_a, spots_df["POSITION_Y"] - y_a
    x_ref, y_ref = x_b - x_a, y_b - y_a

    spots_df["ap_position"] = (x * x_ref + y * y_ref) / (x_ref ** 2 + y_ref ** 2)
    spots_df["edge_position"] = (x * y_ref - y * x_ref) / np.sqrt(x_ref ** 2 + y_ref ** 2)

    return spots_df


def main():
    parser = ArgumentParser(description="process TrackMate output and apply division detection")
    parser.add_argument("-i", "--input", help="input file path")
    parser.add_argument("-o", "--output", help="output directory")
    parser.add_argument("--stem", help="output file stem", default=None)

    parser.add_argument("--suppress_plots", help="plot results", action="store_true")

    args = parser.parse_args()

    # use input path to handle paths
    print(f"input: {args.input}")
    stem = args.stem if args.stem else Path(args.input).parent.stem
    outpath = args.output if args.output else Path(args.input).parent
    plotpath = outpath / "plots"
    plotpath.mkdir(exist_ok=True, parents=True)

    # load metadata
    with open(Path(args.input).parent / f"{stem}_metadata.json") as f:
        metadata = load(f)

    # process trackmate input xml
    xml_path = args.input
    tree = etree.parse(str(xml_path))
    spots_df, graph = process_trackmate_tree(tree)

    # apply division model
    graph, division_times = map_divisions(spots_df, graph, metadata["n_divisions"], savepath=plotpath)
    metadata["division_times"] = division_times

    # reindex tracks, split graph at divisions to assign tracklets
    spots_df = process_graph(spots_df, graph)
    print(spots_df[spots_df["FRAME"] == spots_df["FRAME"].max()].groupby("track_id").size().value_counts())

    # use new track ids to create a lineage visualization
    output_tif = make_lineage_tif(spots_df, metadata["h"], metadata["w"])
    imwrite(outpath / f"{stem}_lineages.tif", output_tif)

    # trim columns, process metadata, and save
    spots_df = prep_spots_df(spots_df, metadata)
    spots_df.to_csv(outpath / f"{stem}_spots.csv")

    with open(outpath / f"{stem}_metadata.json", "w") as f:
        dump(metadata, f)


if __name__ == '__main__':
    main()
