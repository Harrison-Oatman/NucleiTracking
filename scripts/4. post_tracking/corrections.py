import pickle
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from time import time

import colorcet as cc
import napari
import numpy as np
import pandas as pd
from scipy.spatial import KDTree
from sklearn.decomposition import PCA

from nucleitracking.utils.load_hdf5_data import load_embryo

spots_path = Path(r"D:\Tracking\NucleiTracking\data\processed\lightsheet\spots")
corrections_path = spots_path.parent / "corrections"
embryo = r"lightsheet_20250131_spots.h5"
spots_df = load_embryo(spots_path / embryo)[2]
FIRST_FRAME = 25
spots_df = spots_df[spots_df["frame"] > FIRST_FRAME].copy()

starting_points = spots_df["frame"] == spots_df["frame"].min()
spots_df.loc[starting_points, "parent_id"] = -1
spots_df.loc[starting_points, "n_parents"] = 0

ending_points = spots_df["frame"] == spots_df["frame"].max()
spots_df.loc[ending_points, "n_children"] = 0

spots_df["frame"] = spots_df["frame"] - spots_df["frame"].min()

# reset index while preserving parent/child relationships
index_map = {old_idx: new_idx for new_idx, old_idx in enumerate(spots_df.index)}
spots_df.index = range(len(spots_df))
spots_df["parent_id"] = spots_df["parent_id"].map(index_map)


def calculate_tracks(df: pd.DataFrame):
    df = df.sort_values(by="frame")
    df["track_id2"] = df.index
    for _frame, group in df.groupby("frame"):
        group_subset = group[group["n_parents"] == 1]
        df.loc[group_subset.index, "track_id2"] = (
            group_subset["parent_id"].map(df["track_id2"]).fillna(-1).astype(int)
        )

    return df


pal = cc.glasbey


class Action(ABC):

    @abstractmethod
    def execute(self, df: pd.DataFrame):
        pass


class LinkNucleiAction(Action):
    def __init__(self, parent_id: int, child_id: int):
        self.parent_id = parent_id
        self.child_id = child_id

    def execute(self, df: pd.DataFrame):
        # Update the DataFrame
        df.loc[self.child_id, "parent_id"] = self.parent_id
        df.loc[self.parent_id, "n_children"] += 1
        df.loc[self.child_id, "n_parents"] = 1


class ActionHistory:
    def __init__(self):
        self._actions: list[Action] = []

    def push(self, action: Action):
        self._actions.append(action)

    def load(self, path: Path | None):
        with Path.open(path, "rb") as f:
            self._actions = pickle.load(f)

    def save(self, path: Path):
        with Path.open(path, "wb") as f:
            pickle.dump(self._actions, f)

    def get_actions(self):
        return self._actions


class Status(Enum):
    OTHER = 0
    END = 1
    PARENT = 2
    CHILD = 3
    START = 4


status_colors = {
    Status.START: "#2A4275",
    Status.END: "#80212E",
    Status.PARENT: "#708A73",
    Status.CHILD: "#968862",
    Status.OTHER: "#677375",
}

status_colors_rgba = {
    k: np.array((*tuple(int(v.lstrip("#")[i:i + 2], 16) for i in (0, 2, 4)), 255))
    / 255.0
    for k, v in status_colors.items()
}


def assign_status(df):
    terminal = df["n_children"] == 0
    initial = df["n_parents"] == 0

    parent_n_children = 2

    is_parent = df["n_children"] == parent_n_children
    is_child = df["parent_id"].map(is_parent)
    with pd.option_context("future.no_silent_downcasting", True):
        is_child = is_child.fillna(False)

    status = pd.Series(Status.OTHER, index=df.index)
    status[is_parent] = Status.PARENT
    status[is_child] = Status.CHILD
    status[initial] = Status.START
    status[terminal] = Status.END

    return status


@dataclass
class DataSpace:
    df: pd.DataFrame
    status: pd.Series = field(init=False)
    action_history: ActionHistory = field(init=False)
    points: np.ndarray = field(init=False)
    points_center: np.ndarray = field(init=False)
    marked: dict[int, Status] = field(default_factory=dict)
    unmarked: set = field(default_factory=set)

    def __post_init__(self):
        self.action_history: ActionHistory = ActionHistory()
        self.status = assign_status(self.df)
        self.df["status"] = self.status
        calculate_tracks(self.df)


        self.points = self.df[["frame", "z", "y", "x"]].to_numpy()
        self.points_center = np.mean(self.points[:, 1:], axis=0)

        self.tree_points = self.points * np.array([250, 1, 1, 1])
        self.tree = KDTree(self.tree_points)

        self.find_unmarked()

    def find_unmarked(self):
        """
        Adds all terminal nuclei not in marked to unmarked set.
        """
        terminal_nuclei = self.df[self.df["n_children"] == 0].index
        for nuc_id in terminal_nuclei:
            if nuc_id not in self.marked:
                self.unmarked.add(nuc_id)

    def mark_nucleus(self, nuc_id: int, status: Status):
        self.marked[nuc_id] = status
        if nuc_id in self.unmarked:
            self.unmarked.remove(nuc_id)

        self.df.loc[nuc_id, "status"] = status
        self.status = self.df["status"]


class Controller:
    def __init__(self, viewer, df):
        self.viewer = viewer
        self.data = DataSpace(df)
        self.points_layer = self._add_points_layer()
        self.current_view = "track"

    def _add_points_layer(self):
        points = self.data.points
        return self.viewer.add_points(
            points,
            size=self.data.df["radius"].to_numpy() * 2.0,
            face_color=[pal[tid % 256] for tid in self.data.df["track_id2"]],
            border_color=["k" for _ in range(len(points))],
            border_width=0.2,
            name="nuclei",
        )

    def get_nuc_neighbors(self, nuc, k=10):
        tree = self.data.tree
        _dis, neighbors = tree.query(self.data.tree_points[nuc], k)
        return neighbors

    def _nuc_normal_axis(self, nuc_id, k=25):
        points = self.data.points
        neighbors = self.get_nuc_neighbors(nuc_id, k=k)

        pca = PCA(3)
        pca.fit(points[neighbors][:, 1:])
        normal_axis = pca.components_[-1]

        nucleus_loc = points[nuc_id]
        centered_loc = nucleus_loc[1:] - self.data.points_center
        if np.linalg.norm(centered_loc + normal_axis) > np.linalg.norm(centered_loc):
            normal_axis = -normal_axis

        return normal_axis

    def view_nucleus(self, nuc_id):
        points = self.data.points
        viewer = self.viewer

        nucleus_loc = points[nuc_id]
        normal_axis = self._nuc_normal_axis(nuc_id, k=25)

        # set loc for camera center
        viewer.dims.set_current_step(0, nucleus_loc[0])
        viewer.camera.center = nucleus_loc

        viewer.camera.set_view_direction(normal_axis)

    def color_by_status(self):
        status = self.data.status
        colors = np.array([status_colors_rgba[s] for s in status])
        self.points_layer.face_color = colors
        self.points_layer.refresh()
        self.current_view = "status"

    def color_by_track(self):
        df = self.data.df
        df = calculate_tracks(df)
        colors = np.array([pal[tid % 256] for tid in df["track_id2"]])
        self.points_layer.face_color = colors
        self.points_layer.refresh()
        self.current_view = "track"

    def do_action(self, action: Action):
        action.execute(self.data.df)
        self.data.action_history.push(action)

        self.data.status = assign_status(self.data.df)
        self.data.df["status"] = self.data.status

        if self.current_view == "status":
            self.color_by_status()

    def clear_all_selections(self):
        self.points_layer.selected_data.clear()
        self.points_layer.refresh()

        self.points_layer.border_color = ["k" for _ in range(len(self.data.points))]

    def next_unmarked(self):
        if not self.data.unmarked:
            return None

        return min(self.data.unmarked, key=lambda nid: self.data.points[nid, 0])

    def save_corrections(self, path: Path):
        self.data.action_history.save(path)

    def load_corrections(self, path: Path):
        self.data.action_history.load(path)
        for action in self.data.action_history.get_actions():
            action.execute(self.data.df)

        self.data.status = assign_status(self.data.df)
        self.data.df["status"] = self.data.status

        if self.current_view == "status":
            self.color_by_status()

    def track_id_value_counts(self):
        self.data.df = calculate_tracks(self.data.df)
        frame = self.data.df["frame"].max()
        return (
            self.data.df[self.data.df["frame"] == frame]
            .groupby("track_id2")["frame"]
            .count()
            .value_counts()
        )

    def export_dataframe(self):
        return self.data.df


def main():
    viewer = napari.Viewer()
    viewer.theme = "light"

    controller = Controller(viewer, spots_df)

    # load most recent corrections if available
    correction_files = sorted(corrections_path.glob("corrections_*.pkl"))
    if correction_files:
        latest_corrections = correction_files[-1]
        controller.load_corrections(latest_corrections)


    data = controller.data
    points = data.points
    points_layer = controller.points_layer

    @points_layer.bind_key("l")
    def link_nuclei(layer):
        """Link currently selected nuclei as parent-child."""
        selected = layer.selected_data

        nuc_ids = sorted(selected, key=lambda nid: points[nid, 0])
        frames = points[nuc_ids, 0]

        for frame in frames[1:]:
            if frame <= frames[0]:
                return

        parent_ix = nuc_ids[0]

        for child_ix in nuc_ids[1:]:
            parent_id = spots_df.index[parent_ix]
            child_id = spots_df.index[child_ix]

            action = LinkNucleiAction(parent_id, child_id)
            controller.do_action(action)

    @points_layer.bind_key("q")
    def switch_view(_layer):
        if controller.current_view == "status":
            controller.color_by_track()
        else:
            controller.color_by_status()

    @points_layer.bind_key("shift+s")
    def save_corrections(_layer):
        timestamp = int(time())
        save_path = corrections_path / f"corrections_{timestamp}.pkl"
        controller.save_corrections(save_path)

    @points_layer.bind_key("Enter")
    def choose_nucleus(layer):
        nuc = controller.next_unmarked()
        if nuc is None:
            return

        controller.clear_all_selections()
        controller.view_nucleus(nuc)

        layer.selected_data.add(nuc)
        layer.border_color[nuc] = [0.0, 1.0, 0.0, 1.0]
        layer.refresh()

    napari.run()

    @points_layer.bind_key("shift+e")
    def export_dataframe(_layer):
        export_df = controller.export_dataframe()
        export_path = spots_path.parent / f"{embryo[:-9]}_corrected_spots.h5"
        export_df.to_hdf(export_path, key="df")


if __name__ == "__main__":
    main()
