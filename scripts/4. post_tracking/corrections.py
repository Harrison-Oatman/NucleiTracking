import pickle
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from time import time, perf_counter

import colorcet as cc
import napari
import numpy as np
import pandas as pd
from scipy.spatial import KDTree
from sklearn.decomposition import PCA
import seaborn as sns

from h5py import File

from nucleitracking.utils.load_hdf5_data import load_embryo

count_palette = sns.color_palette("Spectral", 16)
count_palette.append((0., 0., 0.))

spots_path = Path(r"D:\Tracking\NucleiTracking\data\processed\lightsheet\spots")
corrections_path = spots_path.parent / "corrections"
embryo = r"lightsheet_20250131_spots.h5"
stem, metadata, spots_df = load_embryo(spots_path / embryo)

print(spots_df.columns)

FIRST_FRAME = 25
spots_df = spots_df[spots_df["frame"] > FIRST_FRAME].copy()

starting_points = spots_df["frame"] == spots_df["frame"].min()
spots_df.loc[starting_points, "parent_id"] = -1
spots_df.loc[starting_points, "n_parents"] = 0

ending_points = spots_df["frame"] == spots_df["frame"].max()
spots_df.loc[ending_points, "n_children"] = 0

spots_df["frame"] = spots_df["frame"] - spots_df["frame"].min()

index_id = {
    ix: ix_id for ix, ix_id in enumerate(spots_df.index)
}

id_index = {
    ix_id: ix for ix, ix_id in enumerate(spots_df.index)
}

# spots_df.sort_values(by="frame", inplace=True)

class Status(Enum):
    OTHER = auto()
    END = auto()
    PARENT = auto()
    CHILD = auto()
    START = auto()
    ISSUE = auto()
    POLE_TERMINAL = auto()


status_colors = {
    Status.START: "#2A4275",
    Status.END: "#80212E",
    Status.PARENT: "#708A73",
    Status.CHILD: "#968862",
    Status.OTHER: "#677375",
    Status.ISSUE: "#EBACDA",
    Status.POLE_TERMINAL: "#588157"
}

status_colors_rgba = {
    k: np.array((*tuple(int(v.lstrip("#")[i:i + 2], 16) for i in (0, 2, 4)), 255))
    / 255.0
    for k, v in status_colors.items()
}

def calculate_tracks(df: pd.DataFrame):
    start = perf_counter()
    df["track_id"] = df.index
    for _frame, group in df.groupby("frame"):
        group_subset = group[group["n_parents"] == 1]
        df.loc[group_subset.index, "track_id"] = (
            group_subset["parent_id"].map(df["track_id"]).fillna(-1).astype(int)
        )

    end = perf_counter()

    print(f"took {end - start:0.3f}s to calculate tracks")


    return df


pal = cc.glasbey

@dataclass
class DataSpace:
    df: pd.DataFrame
    status: pd.Series = field(init=False)
    points: np.ndarray = field(init=False)
    points_center: np.ndarray = field(init=False)
    marked: dict[int, Status] = field(default_factory=dict)
    unmarked: set = field(default_factory=set)

    def __post_init__(self):
        self.status = assign_status(self.df)
        self.df["status"] = self.status
        self.df = calculate_tracks(self.df)


        self.points = self.df[["frame", "z", "y", "x"]].to_numpy()
        self.points_center = np.mean(self.points[:, 1:], axis=0)

        self.tree_points = self.points * np.array([250, 1, 1, 1])
        self.tree = KDTree(self.tree_points)

        self.find_unmarked()

    def find_unmarked(self):
        """
        Adds all terminal nuclei not in marked to unmarked set.
        """
        terminal_nuclei = self.df[self.df["status"] == Status.END].index
        for nuc_id in terminal_nuclei:
            if nuc_id not in self.marked:
                self.unmarked.add(nuc_id)

    def mark_issue(self, nuc_id: int):
        """
        Marks entire branch as issue
        """
        parent_id = self.df.loc[nuc_id, "parent_id"]
        self.mark_nucleus(nuc_id, Status.ISSUE)

        i = 0

        terminated_at_branch = False

        while parent_id != -1:

            nuc_id = parent_id

            if self.df.loc[nuc_id, "n_children"] == 2:
                terminated_at_branch = True
                break

            self.mark_nucleus(nuc_id, Status.ISSUE)


            parent_id = self.df.loc[nuc_id, "parent_id"]

            i += 1

        print(f"{i} issues marked (terminated at {'branch' if terminated_at_branch else 'start'})")


    def mark_nucleus(self, nuc_id: int, status: Status):

        self.marked[nuc_id] = status
        if nuc_id in self.unmarked:
            self.unmarked.remove(nuc_id)
            print(f"{len(self.unmarked)} unmarked termini remain")

        print(self.df.loc[nuc_id, "status"])

        # noinspection PyTypeChecker
        self.df.loc[nuc_id, "status"] = status
        self.status = self.df["status"]

        print(self.df.loc[nuc_id, "status"])


class Action(ABC):

    @abstractmethod
    def execute(self, data: DataSpace):
        pass


class LinkNucleiAction(Action):
    def __init__(self, parent_id: int, child_id: int):
        self.parent_id = parent_id
        self.child_id = child_id

    def execute(self, data: DataSpace):
        df = data.df

        df.loc[self.child_id, "parent_id"] = self.parent_id
        df.loc[self.parent_id, "n_children"] += 1
        df.loc[self.child_id, "n_parents"] = 1

        if self.parent_id in data.unmarked:
            data.unmarked.remove(self.parent_id)


class MarkNucleusAction(Action):

    def __init__(self, nuc_id: int, status: Status):
        self.nuc_id = nuc_id
        self.status = status

    def execute(self, data: DataSpace):
        data.mark_nucleus(self.nuc_id, self.status)

class MarkIssueAction(Action):

    def __init__(self, nuc_id: int):
        self.nuc_id = nuc_id

    def execute(self, data: DataSpace):
        data.mark_issue(self.nuc_id)


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
            # noinspection PyTypeChecker
            pickle.dump(self._actions, f)

        print("saved")

    def get_actions(self):
        return self._actions



def assign_status(df):
    start = perf_counter()
    df["n_children"] = df.index.map(df["parent_id"].value_counts()).fillna(0).astype(int)
    df["n_parents"] = (df["parent_id"] != -1).astype(int)

    terminal = (df["n_children"] == 0) & (df["frame"] < df["frame"].max())
    initial = df["n_parents"] == 0

    parent_n_children = 2

    is_parent = df["n_children"] == parent_n_children
    is_child = df["parent_id"].map(is_parent)
    with pd.option_context("future.no_silent_downcasting", True):
        is_child = is_child.fillna(False)

    # noinspection PyTypeChecker
    status = pd.Series(Status.OTHER, index=df.index)
    status[is_parent] = Status.PARENT
    status[is_child] = Status.CHILD
    status[initial] = Status.START
    status[terminal] = Status.END

    end = perf_counter()
    print(f"status calculated in {end - start :0.3f}.")

    return status


class Controller:
    def __init__(self, viewer, df):
        self.viewer = viewer
        self.data = DataSpace(df)
        self.points_layer = self._add_points_layer()
        self.current_view = "track"
        self.action_history: ActionHistory = ActionHistory()

    def _add_points_layer(self):
        points = self.data.points
        return self.viewer.add_points(
            points,
            size=self.data.df["radius"].to_numpy() * 2.0,
            face_color=[pal[tid % 256] for tid in self.data.df["track_id"]],
            border_color=["k" for _ in range(len(points))],
            border_width=0.2,
            name="nuclei",
        )

    def get_nuc_neighbors(self, nuc_id, k=10):
        tree = self.data.tree

        nuc_ix = id_index[nuc_id]

        _dis, neighbors = tree.query(self.data.tree_points[nuc_ix], k)
        return neighbors

    def _nuc_normal_axis(self, nuc_id, k=25):
        points = self.data.points
        neighbors = self.get_nuc_neighbors(nuc_id, k=k)

        pca = PCA(3)
        pca.fit(points[neighbors][:, 1:])
        normal_axis = pca.components_[-1]

        nuc_ix = id_index[nuc_id]

        nucleus_loc = points[nuc_ix]
        centered_loc = nucleus_loc[1:] - self.data.points_center
        if np.linalg.norm(centered_loc + normal_axis) > np.linalg.norm(centered_loc):
            normal_axis = -normal_axis

        return normal_axis

    def view_nucleus(self, nuc_id):
        points = self.data.points
        viewer = self.viewer

        nuc_ix = id_index[nuc_id]

        nucleus_loc = points[nuc_ix]
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
        colors = np.array([pal[tid % 256] for tid in df["track_id"]])
        self.points_layer.face_color = colors
        self.points_layer.refresh()
        self.current_view = "track"

    def color_by_count(self):
        df = self.data.df
        df = calculate_tracks(df)

        df["is_parent"] = df["status"] == Status.PARENT

        track_counts = df.groupby("track_id")["is_parent"].sum() + 1

        colors = [count_palette[min(track_counts[tid] - 1, 16)] for tid in df["track_id"]]
        self.points_layer.face_color = colors
        self.points_layer.refresh()
        self.current_view = "count"

    def do_action(self, action: Action):
        action.execute(self.data)
        self.action_history.push(action)


    def refresh_status(self):
        self.data.df["status"] = assign_status(self.data.df)

        for marked, status in self.data.marked.items():
            self.data.df.loc[marked, "status"] = status

        self.data.status = self.data.df["status"]


        if self.current_view == "status":
            self.color_by_status()
            self.points_layer.refresh()

    def clear_all_selections(self):
        self.points_layer.selected_data.clear()
        self.points_layer.refresh()

        self.points_layer.border_color = ["k" for _ in range(len(self.data.points))]

    def next_unmarked(self):
        if not self.data.unmarked:
            return None

        return min(self.data.unmarked, key=lambda nid: self.data.points[id_index[nid], 0])

    def save_corrections(self, path: Path):
        self.action_history.save(path)

    def load_corrections(self, path: Path):
        self.action_history.load(path)
        for action in self.action_history.get_actions():
            action.execute(self.data)

        self.data.status = assign_status(self.data.df)
        self.data.df["status"] = self.data.status

        if self.current_view == "status":
            self.color_by_status()

    def track_id_value_counts(self):
        self.data.df = calculate_tracks(self.data.df)
        frame = self.data.df["frame"].max()
        # noinspection PyArgumentList
        return (
            self.data.df[self.data.df["frame"] == frame]
            .groupby("track_id")["frame"]
            .count()
            .value_counts()
        )

    def export_dataframe(self):
        self.data.df = calculate_tracks(self.data.df)

        export_df = self.data.df.copy()

        tid_remap = {tid: ix for ix, tid in enumerate(spots_df["track_id"].value_counts().index)}
        export_df["track_id"] = export_df["track_id"].map(tid_remap)

        return export_df

    def get_nuc_branch(self, nuc_id):
        branch_ids = {nuc_id}

        df = self.data.df

        parent_id = df.loc[nuc_id, "parent_id"]
        terminated_at_branch = False

        while parent_id != -1:

            nuc_id = parent_id

            if df.loc[nuc_id, "n_children"] == 2:
                terminated_at_branch = True
                break

            branch_ids.add(nuc_id)
            parent_id = df.loc[nuc_id, "parent_id"]

        return branch_ids


def main():
    viewer = napari.Viewer(ndisplay=3)

    custom_theme = napari.utils.theme.get_theme('dark')
    custom_theme.canvas = "#7A8CA3"
    napari.utils.theme.register_theme('custom', custom_theme, 'custom')
    viewer.theme = "custom"

    controller = Controller(viewer, spots_df)

    print(controller.track_id_value_counts())

    # load most recent corrections if available
    correction_files = sorted(corrections_path.glob("corrections_*.pkl"))
    if correction_files:
        latest_corrections = correction_files[-1]
        controller.load_corrections(latest_corrections)

    controller.refresh_status()

    print(controller.track_id_value_counts())


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

        controller.refresh_status()

    @points_layer.bind_key("q")
    def switch_view(_layer):
        if controller.current_view == "status":
            controller.color_by_track()
        elif controller.current_view == "track":
            controller.color_by_count()
        else:
            controller.color_by_status()

    @points_layer.bind_key("shift+s")
    def save_corrections(_layer):
        timestamp = int(time())
        save_path = corrections_path / f"corrections_{timestamp}.pkl"
        controller.save_corrections(save_path)

    @points_layer.bind_key("shift+e")
    def export_dataframe(_layer):
        export_df = controller.export_dataframe()
        export_path = spots_path.parent / f"{embryo[:-9]}_corrected_spots.h5"
        export_df["status"] = [s.value for s in export_df["status"]]
        export_df["frame"] = export_df["frame"] + FIRST_FRAME
        del export_df["source"]
        export_df.to_hdf(export_path, key="df", format="fixed", mode="w")

        with File(export_path, "a") as f:
            m = f.create_group("metadata")
            for k, v in metadata.items():
                m.attrs[k] = v

    @points_layer.bind_key("t")
    def mark_as_terminal(layer):
        df = data.df

        for nuc_ix in layer.selected_data:
            nuc_id = index_id[nuc_ix]
            if df.loc[nuc_id, "n_children"] != 0:
                print("one or more selected nuclei are not terminal")
                return

        for nuc_ix in layer.selected_data:
            nuc_id = index_id[nuc_ix]
            action = MarkNucleusAction(nuc_id, Status.END)

            controller.do_action(action)

        controller.refresh_status()

    @points_layer.bind_key("p")
    def mark_as_pole_terminal(layer):
        df = data.df

        for nuc_ix in layer.selected_data:
            nuc_id = index_id[nuc_ix]
            if df.loc[nuc_id, "n_children"] != 0:
                print("one or more selected nuclei are not terminal")
                return

        for nuc_ix in layer.selected_data:
            nuc_id = index_id[nuc_ix]
            action = MarkNucleusAction(nuc_id, Status.POLE_TERMINAL)

            controller.do_action(action)

        controller.refresh_status()

    @points_layer.bind_key("g")
    def mark_as_issue(layer):

        print("marking as issue")
        print(layer.selected_data)
        for nuc_ix in layer.selected_data:
            nuc_id = index_id[nuc_ix]
            action = MarkIssueAction(nuc_id)

            controller.do_action(action)

        controller.refresh_status()

    @points_layer.bind_key("Enter")
    def choose_nucleus(layer):
        nuc_id = controller.next_unmarked()

        nuc_ix = id_index[nuc_id]

        if nuc_id is None:
            return

        controller.clear_all_selections()
        controller.view_nucleus(nuc_id)

        layer.selected_data.add(nuc_ix)

        branch_ids = controller.get_nuc_branch(nuc_id)

        for branch_id in branch_ids:
            branch_ix = id_index[branch_id]
            layer.border_color[branch_ix] = [0.0, 1.0, 1.0, 1.0]

        layer.border_color[nuc_ix] = [0.0, 1.0, 0.0, 1.0]

        if len(branch_ids) < 4:
            mark_as_issue(layer)

        elif controller.data.df.loc[nuc_id, "AP"] > 0.95:
            mark_as_pole_terminal(layer)

        layer.refresh()


    napari.run()


if __name__ == "__main__":
    main()
