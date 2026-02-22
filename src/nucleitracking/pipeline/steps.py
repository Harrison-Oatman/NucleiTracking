from pathlib import Path

import networkx as nx
import pandas as pd

from nucleitracking.models.lap_tracking import run_lap_tracking
from nucleitracking.models.new_tracking import (
    interpolate_points,
    map_divisions,
    merge_close_tracklets,
    process_graph,
)
from nucleitracking.pipeline.config import PipelineConfig


def run_tracking(dataset: Path, config: PipelineConfig):
    out_dir = dataset / f"tracking_{config.param_set_name}"
    out_file = out_dir / "lap_tracked_spots.csv"
    if out_file.exists():
        print(f"[{dataset.name}] LAP Tracking (Skipped: {out_file.name} exists)")
        return

    print(f"[{dataset.name}] Running LAP Tracking...")

    # Load 3D centroids generated from the local post-processing Reconstruction step
    # For now, we assume a standard name
    centroids_path = dataset / f"tracking_{config.param_set_name}" / "new_centroids.csv"
    if not centroids_path.exists():
        print(
            f"  [Warning] Centroids file not found at {centroids_path}. Skipping tracking."
        )
        return

    centroids = pd.read_csv(centroids_path)

    # Filter frames based on user configuration
    if config.local_post.tracking.start_frame > 0:
        print(
            f"  Filtering out frames before {config.local_post.tracking.start_frame}..."
        )
        centroids = centroids[
            centroids["FRAME"] >= config.local_post.tracking.start_frame
        ]

    if config.local_post.tracking.skip_frames:
        print(f"  Skipping frames: {config.local_post.tracking.skip_frames}...")
        centroids = centroids[
            ~centroids["FRAME"].isin(config.local_post.tracking.skip_frames)
        ]

    # Run the new Python LAP tracker instead of TrackMate
    spots_df, _graph = run_lap_tracking(
        centroids,
        max_distance=config.local_post.tracking.search_radius,
        max_gap_frames=config.local_post.tracking.max_gap_frames,
    )

    # Save preliminary tracking results
    out_dir = dataset / f"tracking_{config.param_set_name}"
    spots_df.to_csv(out_dir / "lap_tracked_spots.csv", index=False)
    print(f"  Saved LAP tracked spots to {out_dir / 'lap_tracked_spots.csv'}")


def run_division_mapping(dataset: Path, config: PipelineConfig):
    out_dir = dataset / f"tracking_{config.param_set_name}"
    out_file = out_dir / "final_lineages.csv"
    if out_file.exists():
        print(f"[{dataset.name}] Division Mapping (Skipped: {out_file.name} exists)")
        return

    print(f"[{dataset.name}] Running Division Mapping...")

    spots_path = out_dir / "lap_tracked_spots.csv"

    if not spots_path.exists():
        print("  [Warning] LAP tracked spots not found. Skipping division mapping.")
        return

    spots_df = pd.read_csv(spots_path)

    graph = nx.DiGraph()

    track_id_most_recent = {}

    for frame, spots in spots_df.groupby("FRAME"):
        parents = spots["linear_track_id"].map(track_id_most_recent)

        for parent, child in zip(parents, spots[["graph_key"]]):
            if pd.isna(parent):
                continue
            graph.add_edge(parent, child, time=1)

        track_id_most_recent.update(
            dict(spots[["linear_track_id", "graph_key"]].values)
        )

    print("  Interpolating points...")
    interpolated_spots_df, interpolated_graph = interpolate_points(spots_df, graph)

    print("  Merging close tracklets...")
    merged_spots_df, merged_graph = merge_close_tracklets(
        interpolated_spots_df, interpolated_graph, max_dis=12
    )

    print("  Mapping divisions...")
    mapped_graph, _test_spots_df = map_divisions(
        merged_spots_df,
        merged_graph,
        config.local_post.division_mapping.interphase_dividers,
        new_track_cost=config.local_post.division_mapping.new_track_cost,
    )

    print("  Processing final lineage graph...")
    final_spots_df = process_graph(merged_spots_df, mapped_graph)

    # Save the final results ready for HDF5 extraction!
    final_spots_df.to_csv(out_dir / "final_lineages.csv", index=False)
    print(f"  Saved final lineages to {out_dir / 'final_lineages.csv'}")
