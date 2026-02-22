from pathlib import Path

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
    print(f"[{dataset.name}] Running Division Mapping...")

    out_dir = dataset / f"tracking_{config.param_set_name}"
    spots_path = out_dir / "lap_tracked_spots.csv"

    if not spots_path.exists():
        print("  [Warning] LAP tracked spots not found. Skipping division mapping.")
        return

    spots_df = pd.read_csv(spots_path)

    # Needs the graph to proceed. In a full implementation, we'd serialize/deserialize the networkx graph.
    # For this script we will reconstruct it from lap_tracking again or save it as a pickle.
    # We will assume we can recreate the graph or we simply call lap_tracking again for simplicity in this wrapper
    spots_df, graph = run_lap_tracking(
        spots_df,
        max_distance=config.local_post.tracking.search_radius,
        max_gap_frames=config.local_post.tracking.max_gap_frames,
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
