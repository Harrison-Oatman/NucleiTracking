import itertools

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment


def get_division_costs(
    spots_df: pd.DataFrame,
    div_start: int,
    div_end: int,
):
    """
    Generates a cost matrix for the assignment of new tracks to existing tracks
    """

    tracklet_starts = spots_df.groupby("tracklet_id")["frame"].min()
    tracklet_ends = spots_df.groupby("tracklet_id")["frame"].max()
    tracklets = tracklet_starts.index

    full_length_tracklets = tracklets[
        (tracklet_starts <= div_start) & (tracklet_ends >= div_end)
    ]
    new_tracklets = tracklets[
        (tracklet_starts > div_start)
        & (tracklet_starts <= div_end)
        & (tracklet_ends > div_end)
    ]

    spots_df["child_id"] = spots_df.index.map(
        dict(zip(spots_df["parent_id"], spots_df.index))
    )

    for axis in ["px_x", "px_y", "px_z"]:
        spots_df[f"child_{axis}_offset"] = (
            spots_df["child_id"].map(spots_df[axis]) - spots_df[axis]
        )
        spots_df[f"predicted_{axis}"] = spots_df["child_id"].map(
            spots_df[axis]
        ) - spots_df["child_id"].map(spots_df[f"child_{axis}_offset"])

    print(np.nanmean(np.abs(spots_df["predicted_px_x"] - spots_df["px_x"])))

    full_length_tracklet_id_to_idx = {
        tid: idx for idx, tid in enumerate(full_length_tracklets)
    }

    full_length_spots_df = spots_df[spots_df["tracklet_id"].isin(full_length_tracklets)]
    full_length_spots_df_by_frame = {
        frame: group
        for frame, group in full_length_spots_df.groupby("frame")
        if (frame >= div_start) and (frame <= div_end)
    }

    cost_matrix = np.full((len(new_tracklets), len(full_length_tracklets)), np.inf)
    parent_ids = np.full(
        (len(new_tracklets), len(full_length_tracklets)), -1, dtype=int
    )
    child_ids = []
    first_frames = []

    for new_track_idx, (tid, spots) in enumerate(
        spots_df[spots_df["tracklet_id"].isin(new_tracklets)].groupby("tracklet_id")
    ):
        first_spot = spots.iloc[0]
        full_length_group = full_length_spots_df_by_frame.get(first_spot["frame"] - 1)
        first_frames.append(first_spot["frame"])

        distances = full_length_group["px_x"] * 0
        for axis in ["px_x", "px_y", "px_z"]:
            predicted_new_spot = first_spot[axis] - first_spot[f"child_{axis}_offset"]
            predicted_full_length_spot = full_length_group[f"predicted_{axis}"]
            actual = full_length_group[axis]
            distances += (
                (predicted_new_spot + predicted_full_length_spot) / 2 - actual
            ) ** 2

        idx = full_length_group["tracklet_id"].map(full_length_tracklet_id_to_idx)

        cost_matrix[new_track_idx, idx] = distances
        parent_ids[new_track_idx, idx] = full_length_group.index
        child_ids.append(first_spot.name)
        # print(first_spot)
        # print(first_spot.name)

    # print(np.mean(np.min(cost_matrix, axis=1)))
    # print(pd.Series(first_frames).value_counts())

    cost_matrix = np.nan_to_num(cost_matrix, nan=np.inf)
    # print(parent_ids, child_ids)

    return cost_matrix, parent_ids, child_ids, spots_df


def map_divisions(spots_df: pd.DataFrame, interphase_dividers, new_track_cost=20):
    spots_df = spots_df.copy()
    spots_df["status"] = 0

    for start, end in itertools.pairwise(interphase_dividers):
        print(f"mapping divisions between {start} and {end}")
        cost_matrix, parent_ids, child_ids, spots_df = get_division_costs(
            spots_df, start, end
        )

        # add no assignment cost
        n_in, n_out = cost_matrix.shape
        new_track_costs = np.ones((n_in, n_in)) * new_track_cost
        cost_matrix = np.hstack([cost_matrix, new_track_costs])

        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        print(
            f"{np.sum(col_ind < n_out)} / {n_in} (max {min(n_in, n_out)}) new tracks assigned"
        )

        for i, j in zip(row_ind, col_ind, strict=False):
            if j >= n_out:
                continue

            child = child_ids[i]
            parent = parent_ids[i][j]

            spots_df.loc[child, "parent_id"] = parent

    return spots_df
