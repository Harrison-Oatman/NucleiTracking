import networkx as nx
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.spatial import KDTree
from tqdm import tqdm


def run_lap_tracking(
    spots_df: pd.DataFrame, max_distance: float = 5.0, max_gap_frames: int = 3
) -> tuple[pd.DataFrame, nx.DiGraph]:
    """
    Implements a Linear Assignment Problem (LAP) tracker for 3D nuclei.
    Includes frame-to-frame linking and gap closing.

    Args:
        spots_df (pd.DataFrame): Dataframe containing 'frame', 'px_x', 'px_y', 'px_z'.
                                 Must have a unique index corresponding to the spot ID.
        max_distance (float): Maximum spatial distance to link spots.
        max_gap_frames (int): Maximum frames allowed to bridge a gap.

    Returns:
        pd.DataFrame: Modified spots_df with 'linear_track_id' and 'graph_key'.
        nx.DiGraph: Tracking graph compatible with the division mapping model.
    """
    spots_df = spots_df.copy()

    # Ensure graph_key exists and is integer
    if "graph_key" not in spots_df.columns:
        spots_df["graph_key"] = spots_df.index

    spots_df["graph_key"] = spots_df["graph_key"].astype(int)

    graph = nx.DiGraph()
    for spot_id in spots_df["graph_key"]:
        graph.add_node(spot_id)

    frames = sorted(spots_df["frame"].unique())

    # --- Step 1: Frame-to-Frame Linking ---
    # To mimic TrackMate, we link points between t and t+1.
    for t_idx in tqdm(range(len(frames) - 1), desc="Frame-to-Frame Linking"):
        t_curr = frames[t_idx]
        t_next = frames[t_idx + 1]

        # We only link if frames are consecutive or close (diff = 1)
        if t_next - t_curr > 1:
            continue

        curr_spots = spots_df[spots_df["frame"] == t_curr]
        next_spots = spots_df[spots_df["frame"] == t_next]

        if len(curr_spots) == 0 or len(next_spots) == 0:
            continue

        curr_pos = curr_spots[["px_x", "px_y", "px_z"]].values
        next_pos = next_spots[["px_x", "px_y", "px_z"]].values

        # Build KDTree for quick distance queries and cost matrix construction
        tree = KDTree(next_pos)

        n_curr = len(curr_spots)
        n_next = len(next_spots)

        # Cost matrix: padding with max distance cost for non-assignments
        cost_matrix = np.full((n_curr, n_curr + n_next), max_distance * 10)

        # Fill strictly spatial distances
        for i, pos in enumerate(curr_pos):
            dists, idxs = tree.query(
                pos, k=min(10, n_next), distance_upper_bound=max_distance
            )
            # KDTree query can return scalar if k=1
            if np.isscalar(dists):
                dists, idxs = [dists], [idxs]

            for d, j in zip(dists, idxs):
                if d == np.inf or j == n_next:
                    continue
                cost_matrix[i, j] = d

        # Alternative assignment (cost of not linking)
        for i in range(n_curr):
            cost_matrix[i, n_next + i] = max_distance

        # Solve LAP
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        for i, j in zip(row_ind, col_ind):
            if j < n_next:
                source_id = curr_spots.iloc[i]["graph_key"]
                target_id = next_spots.iloc[j]["graph_key"]

                # Add edge to temporary graph for building segments before gap closing
                # Trackmate assigns 'time' attribute on edges
                graph.add_edge(source_id, target_id, time=1)

    # --- Step 2: Extract preliminary tracklets ---
    ccs = list(nx.weakly_connected_components(graph))
    tracklet_starts = []
    tracklet_ends = []

    tracklet_id_map = {}

    for t_id, cc in enumerate(ccs, start=1):
        # Sort nodes in tracklet by time
        sorted_nodes = sorted(cc, key=lambda n: spots_df.loc[n, "frame"])
        start_node = sorted_nodes[0]
        end_node = sorted_nodes[-1]

        tracklet_starts.append(start_node)
        tracklet_ends.append(end_node)

        for n in cc:
            tracklet_id_map[n] = t_id

    # --- Step 3: Gap Closing ---
    if max_gap_frames > 1 and len(tracklet_ends) > 0 and len(tracklet_starts) > 0:
        print("Performing gap closing...")
        end_spots = spots_df.loc[tracklet_ends]
        start_spots = spots_df.loc[tracklet_starts]

        n_ends = len(end_spots)
        n_starts = len(start_spots)

        gap_cost_matrix = np.full((n_ends, n_ends + n_starts), max_distance * 10)

        for i, (_, end_spot) in enumerate(end_spots.iterrows()):
            end_frame = end_spot["frame"]
            end_pos = end_spot[["px_x", "px_y", "px_z"]].values

            # Find possible starts in the future within gap window
            valid_starts = start_spots[
                (start_spots["frame"] > end_frame)
                & (start_spots["frame"] <= end_frame + max_gap_frames)
            ]

            for start_idx_in_valid, (start_id, start_spot) in enumerate(
                valid_starts.iterrows()
            ):
                # Get index of this start in the full start array
                j = start_spots.index.get_loc(start_id)
                start_pos = start_spot[["px_x", "px_y", "px_z"]].values
                dist = np.linalg.norm(end_pos - start_pos)

                if dist < max_distance:
                    gap_cost_matrix[i, j] = dist

            gap_cost_matrix[i, n_starts + i] = max_distance

        row_ind, col_ind = linear_sum_assignment(gap_cost_matrix)

        for i, j in zip(row_ind, col_ind):
            if j < n_starts:
                source_id = end_spots.iloc[i]["graph_key"]
                target_id = start_spots.iloc[j]["graph_key"]

                dt = int(
                    spots_df.loc[target_id, "frame"] - spots_df.loc[source_id, "frame"]
                )
                graph.add_edge(source_id, target_id, time=dt)

    # --- Step 4: Finalize Graph and Dataframe ---
    # Assign global linear_track_id identical to what TrackMate parses
    final_ccs = list(nx.weakly_connected_components(graph))
    spots_df["linear_track_id"] = -1

    for t_id, cc in enumerate(final_ccs, start=1):
        cc_nodes = list(cc)
        spots_df.loc[cc_nodes, "linear_track_id"] = t_id

        # Ensure all edges have track_id assigned
        subgraph = graph.subgraph(cc_nodes)
        for u, v in subgraph.edges():
            graph[u][v]["track_id"] = t_id

    spots_df["linear_track_id"] = spots_df["linear_track_id"].astype(int)
    print(f"Tracking complete. Found {len(final_ccs)} tracks.")

    return spots_df, graph
