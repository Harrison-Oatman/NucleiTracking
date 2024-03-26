import pandas as pd
import networkx as nx
import numpy as np

from argparse import ArgumentParser
from pathlib import Path
from lxml import etree
from skimage.draw import polygon
from tqdm import tqdm
from tifffile import imwrite

from src.utils.process_trackmate import process_trackmate_tree
from src.utils.tracklets import compute_edge_distance, identify_peaks
from src.models.division_tracking import map_divisions


def main():
    parser = ArgumentParser(description="process TrackMate output and apply division detection")
    parser.add_argument("-i", "--input", help="input file path")
    parser.add_argument("-o", "--output", help="output directory")
    parser.add_argument("--stem", help="output file stem", default=None)

    parser.add_argument("--n_divisions", help="number of divisions to detect", default=4, type=int)
    # parser.add_argument("r_max", help="maximum distance for each division (default auto detect)",
    #                     nargs="*", type=float, default=None)

    parser.add_argument("--suppress_plots", help="plot results", action="store_true")

    args = parser.parse_args()

    print(f"input: {args.input}")

    stem = args.stem if args.stem else Path(args.input).parent.stem
    outpath = args.output if args.output else Path(args.input).parent
    plotpath = outpath / "plots"
    plotpath.mkdir(exist_ok=True, parents=True)

    xml_path = args.input
    tree = etree.parse(str(xml_path))

    spots_df, graph = process_trackmate_tree(tree)

    graph = map_divisions(spots_df, graph, args.n_divisions, savepath=plotpath)

    new_track_idx = {idx: 0 for idx in spots_df["ID"]}

    for track, c in enumerate(nx.connected_components(graph.to_undirected())):
        for spot in c:
            new_track_idx[spot] = track

    spots_df["track_id"] = [new_track_idx[idx] for idx in spots_df["ID"]]

    shape = (round(spots_df["FRAME"].max()) + 1, 1360, 1360)
    output_tif = np.zeros(shape, dtype=np.uint16)

    for i, spot in tqdm(spots_df.iterrows()):
        x, y = spot["POSITION_X"], spot["POSITION_Y"]
        t = round(spot["FRAME"])
        new_track_id = spot["track_id"]

        xs = [round(pt + x) for pt in spot["roi"][::2]]
        ys = [round(pt + y) for pt in spot["roi"][1::2]]

        rr, cc = polygon(ys, xs, shape[1:])
        output_tif[t, rr, cc] = new_track_id + 1

    imwrite(outpath / f"{stem}_lineages.tif", output_tif)

    print(spots_df[spots_df["FRAME"] == spots_df["FRAME"].max()].groupby("track_id").size().value_counts())

    graph = graph.to_directed()

    parents = [node for node in graph.nodes if graph.out_degree(node) == 2]
    graph_copy = graph.copy()
    for parent in parents:
        children = list(graph.successors(parent))
        for child in children:
            graph_copy.remove_edge(parent, child)

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

    kept_columns = ["ID", "track_id", "tracklet_id", "distance_from_edge", "parent_id", "daughter_id",
                    "FRAME", "POSITION_X", "POSITION_Y", "POSITION_Z",
                    "ELLIPSE_MAJOR", "ELLIPSE_MINOR", "ELLIPSE_THETA", "ELLIPSE_Y0", "ELLIPSE_X0", "ELLIPSE_ASPECTRATIO",
                    "CIRCULARITY", "AREA", "SHAPE_INDEX", "MEDIAN_INTENSITY_CH1"]

    spots_df = spots_df[kept_columns]
    spots_df.to_csv(outpath / f"{stem}_spots.csv")


if __name__ == '__main__':
    main()