from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from h5py import File
from tqdm import tqdm


def load_spots_data(spots_directory: Path, included: list | None = None):
    spots_dfs = []
    metadatas = []
    stems = []

    directories = list(spots_directory.glob("*_spots.h5"))
    print([f"{d.stem}" for d in directories])

    for i, spots_path in tqdm(enumerate(directories), desc="reading spots dfs"):
        if included:
            if i not in included:
                continue

        load_embryo(spots_path)

        stem, metadata, spots_df = load_embryo(spots_path, source_index=i)
        stems.append(stem)
        metadatas.append(metadata)
        spots_dfs.append(spots_df)

    return spots_dfs, metadatas, stems


def load_embryo(spots_path, source_index=None):
    """
    Load spots data for a single embryo
    :param spots_path:
    :param source_index:
    :return:
    stem, metadata, spots_df
    """
    spots_df = pd.read_hdf(spots_path, key="df")
    spots_df["source"] = source_index
    """
        Get metadata
        """
    metadata = {}
    with File(spots_path, "r") as f:
        m = f["metadata"]
        metadata.update(m.attrs)
    """
        Include dynamic time warp data
        """
    dtw_path = spots_path.parent / "dtw" / f"{spots_path.stem}_dtw.h5"
    if dtw_path.exists():
        dtw_update = pd.read_hdf(dtw_path, key="df")
        spots_df = pd.merge(
            spots_df, dtw_update, left_index=True, right_index=True, how="left"
        )
        spots_df["cycle_pseudotime"] = spots_df["pseudotime"] + spots_df["cycle"] - 11
    else:
        spots_df["distance"] = np.nan  # rename this later
        spots_df["pseudotime"] = np.nan
    """
        Calculate displacement
        """
    cols = ["x", "y", "z", "AP", "theta"]
    for col in cols:
        spots_df[f"d{col}"] = (
            spots_df[col] - spots_df["parent_id"].map(spots_df[col])
        ) / metadata["seconds_per_frame"]
    spots_df["dtot"] = np.sqrt(
        spots_df["dx"] ** 2 + spots_df["dy"] ** 2 + spots_df["dz"] ** 2
    )
    spots_df["dAP_abs"] = spots_df["dAP"].abs()

    return spots_path.stem, metadata, spots_df
