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
    print(spots_df.columns)
    """
        Get metadata
        """
    metadata = {}
    with File(spots_path, "r") as f:
        m = f["metadata"]
        metadata.update(m.attrs)

    return spots_path.stem, metadata, spots_df
