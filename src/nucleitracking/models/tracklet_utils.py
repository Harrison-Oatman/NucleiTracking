import pandas as pd


def calculate_tracks_and_tracklets(df: pd.DataFrame):
    df = df.copy()
    df.sort_values(by="frame", inplace=True)
    df["track_id"] = df.index
    for _frame, group in df.groupby("frame"):
        group_subset = group[group["parent_id"] != -1]
        df.loc[group_subset.index, "track_id"] = (
            group_subset["parent_id"].map(df["track_id"]).fillna(-1).astype(int)
        )

    df["n_children"] = (
        df.index.map(df["parent_id"].value_counts()).fillna(0).astype(int)
    )
    df["tracklet_id"] = df.index
    for _frame, group in df.groupby("frame"):
        parent_has_one_child = (
            group["parent_id"].map(df["n_children"] == 1).fillna(0).astype(bool)
        )
        group_subset = group[parent_has_one_child]
        df.loc[group_subset.index, "tracklet_id"] = (
            group_subset["parent_id"].map(df["tracklet_id"]).fillna(-1).astype(int)
        )

    return df
