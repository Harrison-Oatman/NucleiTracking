import pickle
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from time import perf_counter, time

import colorcet as cc
import napari
import numpy as np
import pandas as pd
import seaborn as sns
from h5py import File
from matplotlib.colors import to_rgba
from napari.utils.notifications import show_info
from scipy.spatial import KDTree
from sklearn.decomposition import PCA

count_palette = sns.color_palette("Spectral", 16)
count_palette.append((0.0, 0.0, 0.0))


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
    Status.POLE_TERMINAL: "#588157",
}

status_colors_rgba = {
    k: np.array((*tuple(int(v.lstrip("#")[i : i + 2], 16) for i in (0, 2, 4)), 255))
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


def lineage_contiguity(
    df: pd.DataFrame,
    k: int = 6,
    max_dist_factor: float = 1.6,
    min_event_frames: int = 10,
) -> tuple[pd.Series, pd.DataFrame]:
    """
    Find lineages whose nuclei are not spatially contiguous.

    For each frame, every nucleus is joined to up to ``k`` nearest neighbours
    that lie within ``max_dist_factor`` times the median nearest-neighbour
    distance of that frame. A lineage (``track_id``) is contiguous if its
    nuclei form one connected piece of this graph using only edges between
    members of the lineage. Nuclei outside their lineage's largest piece are
    *detached*: usually a mistracked link carried them away from their sisters
    and cousins.

    Returns
    -------
    detached : bool Series aligned with df.index
    events : one row per detachment event -- a detached nucleus whose parent was
        not detached -- that stays detached along its descendants for at least
        ``min_event_frames`` frames. Columns: nuc_id, frame, track_id,
        n_frames, gap_um (distance from the detached nucleus to the nearest
        nucleus of the lineage's main piece). Sorted by frame.
    """
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    detached = pd.Series(False, index=df.index)
    gap = pd.Series(np.nan, index=df.index)
    for _frame, idx in df.groupby("frame").groups.items():
        sub = df.loc[idx]
        n = len(sub)
        lineage = sub["track_id"].to_numpy()
        if n < 2 or len(np.unique(lineage)) == n:
            continue  # every nucleus is its own lineage
        pts = sub[["x", "y", "z"]].to_numpy(float)
        dist, nb = KDTree(pts).query(pts, k=min(k + 1, n))
        cap = max_dist_factor * np.median(dist[:, 1])
        rows = np.repeat(np.arange(n), nb.shape[1] - 1)
        cols = nb[:, 1:].ravel()
        keep = (dist[:, 1:].ravel() <= cap) & (lineage[rows] == lineage[cols])
        pairs = np.column_stack([rows[keep], cols[keep]])
        graph = coo_matrix(
            (np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(n, n)
        )
        _, comp = connected_components(graph, directed=False)
        pieces = pd.DataFrame({"lineage": lineage, "comp": comp})
        sizes = pieces.groupby(["lineage", "comp"]).size()
        main = sizes.groupby(level="lineage").idxmax().map(lambda t: t[1])
        is_det = comp != pieces["lineage"].map(main).to_numpy()
        detached.loc[idx] = is_det
        for lin in np.unique(lineage[is_det]):
            in_main = (lineage == lin) & ~is_det
            away = np.where((lineage == lin) & is_det)[0]
            d = KDTree(pts[in_main]).query(pts[away])[0]
            gap.loc[idx[away]] = d

    # detachment events: start of a detached stretch, and how long it lasts
    parent = df["parent_id"]
    parent_detached = parent.map(detached).fillna(False).astype(bool)
    starts = df.index[detached & ~parent_detached]
    length = pd.Series(0, index=df.index)
    for _frame, idx in sorted(df.groupby("frame").groups.items(), reverse=True):
        idx = idx[detached.loc[idx].to_numpy()]
        if len(idx) == 0:
            continue
        length.loc[idx] = 1 + length.loc[idx]
        par = parent.loc[idx]
        par = par[par.isin(df.index)]
        if len(par):
            best = length.loc[par.index].groupby(par.to_numpy()).max()
            length.loc[best.index] = np.maximum(length.loc[best.index], best)
    events = pd.DataFrame(
        {
            "nuc_id": starts,
            "frame": df.loc[starts, "frame"].to_numpy(),
            "track_id": df.loc[starts, "track_id"].to_numpy(),
            "n_frames": length.loc[starts].to_numpy(),
            "gap_um": gap.loc[starts].to_numpy(),
        }
    )
    events = events[events["n_frames"] >= min_event_frames]
    return detached, events.sort_values("frame").reset_index(drop=True)


pal = cc.glasbey


@dataclass
class DataSpace:
    df: pd.DataFrame
    status: pd.Series = field(init=False)
    points: np.ndarray = field(init=False)
    points_center: np.ndarray = field(init=False)
    marked: dict[int, Status] = field(default_factory=dict)
    unmarked: set = field(default_factory=set)

    # Per-instance index lookups (position → DataFrame index, and the reverse).
    # Built from df at init so they stay consistent with this DataSpace's DataFrame.
    index_id: dict[int, int] = field(init=False)
    id_index: dict[int, int] = field(init=False)

    def __post_init__(self):
        self.index_id = dict(enumerate(self.df.index))
        self.id_index = {ix_id: ix for ix, ix_id in enumerate(self.df.index)}

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

    def mark_branch(self, nuc_id: int, status: Status):
        """
        Marks entire branch as status
        """
        parent_id = self.df.loc[nuc_id, "parent_id"]
        self.mark_nucleus(nuc_id, status)

        i = 0

        terminated_at_branch = False

        while parent_id != -1:
            nuc_id = parent_id

            if self.df.loc[nuc_id, "n_children"] == 2:
                terminated_at_branch = True
                break

            self.mark_nucleus(nuc_id, status)

            parent_id = self.df.loc[nuc_id, "parent_id"]

            i += 1

        print(
            f"{i} issues marked (terminated at {'branch' if terminated_at_branch else 'start'})"
        )

    def mark_nucleus(self, nuc_id: int, status: Status):
        self.marked[nuc_id] = status
        if nuc_id in self.unmarked:
            self.unmarked.remove(nuc_id)

        # noinspection PyTypeChecker
        self.df.loc[nuc_id, "status"] = status
        self.status = self.df["status"]


class Action(ABC):
    @abstractmethod
    def execute(self, data: DataSpace):
        pass

    @abstractmethod
    def describe(self, data: DataSpace) -> str:
        """Human-readable summary of the action, for display before execution."""
        pass


class LinkNucleiAction(Action):
    def __init__(self, parent_id: int, child_id: int):
        self.parent_id = parent_id
        self.child_id = child_id

    def describe(self, data: DataSpace) -> str:
        pf = int(data.df.loc[self.parent_id, "frame"])
        cf = int(data.df.loc[self.child_id, "frame"])
        return (
            f"Linked nucleus {self.parent_id} (frame {pf}) "
            f"→ nucleus {self.child_id} (frame {cf})"
        )

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

    def describe(self, data: DataSpace) -> str:
        frame = int(data.df.loc[self.nuc_id, "frame"])
        return f"Marked nucleus {self.nuc_id} (frame {frame}) as {self.status.name}"

    def execute(self, data: DataSpace):
        data.mark_nucleus(self.nuc_id, self.status)


class MarkIssueAction(Action):
    def __init__(self, nuc_id: int):
        self.nuc_id = nuc_id

    def describe(self, data: DataSpace) -> str:
        frame = int(data.df.loc[self.nuc_id, "frame"])
        return f"Flagged branch at nucleus {self.nuc_id} (frame {frame}) as ISSUE"

    def execute(self, data: DataSpace):
        data.mark_branch(self.nuc_id, Status.ISSUE)


class MarkNotIssueAction(Action):
    def __init__(self, nuc_id: int):
        self.nuc_id = nuc_id

    def describe(self, data: DataSpace) -> str:
        frame = int(data.df.loc[self.nuc_id, "frame"])
        return f"Unflagged branch at nucleus {self.nuc_id} (frame {frame}) as ISSUE"

    def execute(self, data: DataSpace):
        data.mark_branch(self.nuc_id, Status.OTHER)


class UnlinkFromParentAction(Action):
    """
    Sever the link between a nucleus and its parent.

    After execution the nucleus has no parent (parent_id = -1, n_parents = 0)
    and the former parent's child count is decremented by one.  If the former
    parent now has no children and is not in the last frame it will appear as a
    new terminal END nucleus after the next refresh_status call.
    """

    def __init__(self, nuc_id: int):
        self.nuc_id = nuc_id

    def describe(self, data: DataSpace) -> str:
        frame = int(data.df.loc[self.nuc_id, "frame"])
        parent_id = int(data.df.loc[self.nuc_id, "parent_id"])
        if parent_id == -1:
            return f"Nucleus {self.nuc_id} (frame {frame}) already has no parent — no change"
        parent_frame = int(data.df.loc[parent_id, "frame"])
        return (
            f"Unlinked nucleus {self.nuc_id} (frame {frame}) "
            f"from parent {parent_id} (frame {parent_frame})"
        )

    def execute(self, data: DataSpace):
        df = data.df
        parent_id = df.loc[self.nuc_id, "parent_id"]

        if parent_id == -1:
            return  # already unlinked, nothing to do

        df.loc[self.nuc_id, "parent_id"] = -1
        df.loc[self.nuc_id, "n_parents"] = 0
        df.loc[parent_id, "n_children"] -= 1

        # If the former parent is now childless and not in the last frame it
        # becomes a new terminal; add it to the unmarked set for review.
        if (
            df.loc[parent_id, "n_children"] == 0
            and df.loc[parent_id, "frame"] < df["frame"].max()
            and parent_id not in data.marked
        ):
            data.unmarked.add(parent_id)


class ActionHistory:
    def __init__(self):
        self._actions: list[Action] = []

    def push(self, action: Action):
        self._actions.append(action)

    def pop(self) -> Action | None:
        """Remove and return the most recent action, or None if history is empty."""
        if not self._actions:
            return None
        return self._actions.pop()

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
    df["n_children"] = (
        df.index.map(df["parent_id"].value_counts()).fillna(0).astype(int)
    )
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
    print(f"status calculated in {end - start:0.3f}.")

    return status


class Controller:
    def __init__(self, viewer, df):
        self.viewer = viewer
        # Keep a clean copy of the original data so undo can replay from scratch.
        self._original_df = df.copy()
        self.data = DataSpace(df)
        self.points_layer = self._add_points_layer()
        self.current_view = "track"
        self.action_history: ActionHistory = ActionHistory()
        self.contiguity_events = pd.DataFrame()
        self.contiguity_index = -1
        self.contiguity_detached = np.zeros(len(self.data.points), bool)

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
        nuc_ix = self.data.id_index[nuc_id]
        _dis, neighbors = tree.query(self.data.tree_points[nuc_ix], k)
        return neighbors

    def _nuc_normal_axis(self, nuc_id, k=25):
        points = self.data.points
        neighbors = self.get_nuc_neighbors(nuc_id, k=k)

        pca = PCA(3)
        pca.fit(points[neighbors][:, 1:])
        normal_axis = pca.components_[-1]

        nuc_ix = self.data.id_index[nuc_id]

        nucleus_loc = points[nuc_ix]
        centered_loc = nucleus_loc[1:] - self.data.points_center
        if np.linalg.norm(centered_loc + normal_axis) > np.linalg.norm(centered_loc):
            normal_axis = -normal_axis

        return normal_axis

    def view_nucleus(self, nuc_id):
        points = self.data.points
        viewer = self.viewer

        nuc_ix = self.data.id_index[nuc_id]

        nucleus_loc = points[nuc_ix]
        normal_axis = self._nuc_normal_axis(nuc_id, k=25)

        viewer.dims.set_current_step(0, nucleus_loc[0])
        viewer.camera.center = nucleus_loc
        viewer.camera.set_view_direction(normal_axis)

    def color_by_status(self):
        self._leave_contiguity()
        status = self.data.status
        colors = np.array([status_colors_rgba[s] for s in status])
        self.points_layer.face_color = colors
        self.points_layer.refresh()
        self.current_view = "status"

    def color_by_track(self):
        self._leave_contiguity()
        df = self.data.df
        df = calculate_tracks(df)
        colors = np.array([pal[tid % 256] for tid in df["track_id"]])
        self.points_layer.face_color = colors
        self.points_layer.refresh()
        self.current_view = "track"

    def color_by_count(self):
        self._leave_contiguity()
        df = self.data.df
        df = calculate_tracks(df)

        df["is_parent"] = df["status"] == Status.PARENT

        track_counts = df.groupby("track_id")["is_parent"].sum() + 1

        colors = [
            count_palette[min(track_counts[tid] - 1, 16)] for tid in df["track_id"]
        ]
        self.points_layer.face_color = colors
        self.points_layer.refresh()
        self.current_view = "count"

    def color_by_contiguity(self):
        """
        Grey out contiguous lineages; show lineages that split into separate
        pieces in their lineage colour, with the detached nuclei outlined red.
        """
        df = calculate_tracks(self.data.df)
        detached, events = lineage_contiguity(df)
        self.contiguity_events = events
        self.contiguity_index = -1

        split = df["track_id"].isin(events["track_id"]).to_numpy()
        colors = np.tile(np.array([0.85, 0.85, 0.85, 0.6]), (len(df), 1))
        lineage_rgb = np.array(
            [to_rgba(pal[tid % 256]) for tid in df["track_id"][split]]
        )
        colors[split] = lineage_rgb
        self.contiguity_detached = detached.to_numpy() & split
        self.points_layer.face_color = colors
        self.current_view = "contiguity"
        self._apply_contiguity_borders()
        show_info(
            f"Contiguity: {len(events)} detachment events in "
            f"{events['track_id'].nunique()} lineages (n / b to step through)"
        )

    def _apply_contiguity_borders(self):
        """Red borders on detached nuclei, black elsewhere."""
        borders = np.tile(np.array([0.0, 0.0, 0.0, 1.0]), (len(self.data.points), 1))
        borders[self.contiguity_detached] = [1.0, 0.0, 0.0, 1.0]
        self.points_layer.border_color = borders
        self.points_layer.refresh()

    def _leave_contiguity(self):
        """Drop the red borders when switching to another colour mode."""
        if self.current_view == "contiguity":
            self.clear_all_selections()

    def step_contiguity_event(self, delta: int):
        """Jump to the next / previous detachment event and select its nucleus."""
        if self.current_view != "contiguity":
            self.color_by_contiguity()
        events = self.contiguity_events
        if events.empty:
            show_info("No detached lineages.")
            return None
        self.contiguity_index = (self.contiguity_index + delta) % len(events)
        ev = events.iloc[self.contiguity_index]
        nuc_id = int(ev["nuc_id"])
        self.clear_all_selections()
        self._apply_contiguity_borders()
        self.view_nucleus(nuc_id)
        nuc_ix = self.data.id_index[nuc_id]
        self.points_layer.selected_data.add(nuc_ix)
        self.points_layer.border_color[nuc_ix] = [0.0, 1.0, 0.0, 1.0]
        self.points_layer.refresh()
        show_info(
            f"Detachment {self.contiguity_index + 1}/{len(events)}: nucleus "
            f"{nuc_id} (frame {int(ev['frame'])}) leaves lineage "
            f"{int(ev['track_id'])} for {int(ev['n_frames'])} frames, "
            f"{ev['gap_um']:.1f} um from the rest of it"
        )
        return nuc_id

    def do_action(self, action: Action):
        msg = action.describe(self.data)
        action.execute(self.data)
        self.action_history.push(action)
        n = len(self.action_history.get_actions())
        show_info(f"{msg}  [{n} action{'s' if n != 1 else ''} in history]")

    def undo(self):
        """
        Remove the most recent action and rebuild state by replaying the rest.
        Because actions modify the DataFrame in place they cannot be reversed
        directly; the only safe approach is to re-execute all remaining actions
        on a fresh copy of the original data.
        """
        action = self.action_history.pop()
        if action is None:
            show_info("Nothing to undo.")
            return

        description = action.describe(self.data)

        # Rebuild DataSpace from the pre-correction original.
        self.data = DataSpace(self._original_df.copy())

        for a in self.action_history.get_actions():
            a.execute(self.data)

        self.data.status = assign_status(self.data.df)
        self.data.df["status"] = self.data.status

        self._refresh_view()

        n = len(self.action_history.get_actions())
        show_info(
            f"Undone: {description}  [{n} action{'s' if n != 1 else ''} remaining]"
        )

    def _refresh_view(self):
        """Re-apply the current color mode after a data rebuild."""
        if self.current_view == "status":
            self.color_by_status()
        elif self.current_view == "track":
            self.color_by_track()
        elif self.current_view == "contiguity":
            self.color_by_contiguity()
        else:
            self.color_by_count()

    def refresh_status(self):
        self.data.df["status"] = assign_status(self.data.df)

        for marked, status in self.data.marked.items():
            self.data.df.loc[marked, "status"] = status

        self.data.status = self.data.df["status"]

        if self.current_view == "status":
            self.color_by_status()
            self.points_layer.refresh()
        elif self.current_view == "contiguity":
            # edits change lineage membership; recompute (a couple of seconds)
            index = self.contiguity_index
            self.color_by_contiguity()
            # resume at the same position: the next event moved up if this one was fixed
            self.contiguity_index = max(min(index, len(self.contiguity_events)) - 1, -1)

    def clear_all_selections(self):
        self.points_layer.selected_data.clear()
        self.points_layer.refresh()
        self.points_layer.border_color = ["k" for _ in range(len(self.data.points))]

    def next_unmarked(self):
        if not self.data.unmarked:
            return None
        return min(
            self.data.unmarked,
            key=lambda nid: self.data.points[self.data.id_index[nid], 0],
        )

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
        return export_table(self.data)

    def get_nuc_branch(self, nuc_id):
        branch_ids = {nuc_id}

        df = self.data.df

        parent_id = df.loc[nuc_id, "parent_id"]

        while parent_id != -1:
            nuc_id = parent_id

            if df.loc[nuc_id, "n_children"] == 2:
                break

            branch_ids.add(nuc_id)
            parent_id = df.loc[nuc_id, "parent_id"]

        return branch_ids


def load_spots(spots_path: Path) -> tuple[str, dict, pd.DataFrame]:
    """
    Load a spots HDF5 file produced by either the old scripts pipeline (key="df")
    or the new pipeline runner (key="lineages").

    Returns (stem, metadata, spots_df).
    """
    for key in ("df", "lineages"):
        try:
            spots_df = pd.read_hdf(spots_path, key=key)
            break
        except KeyError:
            continue
    else:
        msg = (
            f"Could not read spots from {spots_path}. "
            "Expected HDF5 key 'df' or 'lineages'."
        )
        raise ValueError(msg)

    spots_df["source"] = None

    metadata = {}
    with File(spots_path, "r") as f:
        if "metadata" in f:
            metadata.update(f["metadata"].attrs)

    return spots_path.stem, metadata, spots_df


def prepare_spots(spots_df: pd.DataFrame, first_frame: int) -> pd.DataFrame:
    """
    Filter to frames after first_frame, reset frame numbering to start at 0,
    and mark boundary nuclei.
    """
    spots_df = spots_df[spots_df["frame"] > first_frame].copy()

    starting_points = spots_df["frame"] == spots_df["frame"].min()
    spots_df.loc[starting_points, "parent_id"] = -1
    spots_df.loc[starting_points, "n_parents"] = 0

    ending_points = spots_df["frame"] == spots_df["frame"].max()
    spots_df.loc[ending_points, "n_children"] = 0

    spots_df["frame"] = spots_df["frame"] - spots_df["frame"].min()

    return spots_df


def export_table(data: DataSpace) -> pd.DataFrame:
    """Corrected spots with lineage ids remapped to contiguous ints by frequency."""
    data.df = calculate_tracks(data.df)
    export_df = data.df.copy()
    tid_remap = {
        tid: ix for ix, tid in enumerate(data.df["track_id"].value_counts().index)
    }
    export_df["track_id"] = export_df["track_id"].map(tid_remap)
    return export_df


def write_corrected_spots(
    export_df: pd.DataFrame, metadata: dict, export_path: Path, frame_offset: int
):
    """Write corrected spots to HDF5, restoring the original frame numbers."""
    export_df = export_df.copy()
    export_df["status"] = [s.value for s in export_df["status"]]
    export_df["frame"] = export_df["frame"] + frame_offset
    if "source" in export_df.columns:
        del export_df["source"]
    export_df.to_hdf(export_path, key="lineages", format="fixed", mode="w")

    with File(export_path, "a") as f:
        m = f.create_group("metadata")
        for k, v in metadata.items():
            m.attrs[k] = v


def frame_offset_of(raw_df: pd.DataFrame, first_frame: int) -> int:
    """The value prepare_spots subtracts from frame numbers."""
    return int(raw_df.loc[raw_df["frame"] > first_frame, "frame"].min())


def export_latest_corrections(
    spots_path: Path, corrections_dir: Path, export_path: Path, first_frame: int = 0
) -> pd.DataFrame:
    """
    Replay the latest corrections pickle and write the corrected spots, without
    opening the viewer. Same result as opening correct-embryo and pressing Shift+E.
    """
    _stem, metadata, raw_df = load_spots(spots_path)
    data = DataSpace(prepare_spots(raw_df, first_frame))

    latest = sorted(corrections_dir.glob("corrections_*.pkl"))[-1]
    history = ActionHistory()
    history.load(latest)
    print(f"Replaying {len(history.get_actions())} actions from {latest.name}")
    for action in history.get_actions():
        action.execute(data)

    data.df["status"] = assign_status(data.df)
    for marked, status in data.marked.items():
        data.df.loc[marked, "status"] = status

    export_df = export_table(data)
    write_corrected_spots(
        export_df, metadata, export_path, frame_offset_of(raw_df, first_frame)
    )
    return export_df


def main(
    spots_path: Path,
    corrections_dir: Path,
    export_path: Path,
    first_frame: int = 0,
):
    """
    Launch the interactive Napari correction session for a single embryo.

    Parameters
    ----------
    spots_path:
        Path to the tracked embryo HDF5 file (``*_spots.h5``).
    corrections_dir:
        Directory where correction pickle files are read from and saved to.
        Created automatically if it does not exist.
    export_path:
        Path where the corrected HDF5 will be written on Shift+E.
    first_frame:
        Frames at or before this index are excluded before the session starts.
        Matches ``config.local_post.tracking.start_frame`` in the pipeline.
        Defaults to 0 (no frames dropped).
    """
    corrections_dir.mkdir(parents=True, exist_ok=True)

    _stem, metadata, raw_df = load_spots(spots_path)
    print(raw_df.columns)

    frame_offset = frame_offset_of(raw_df, first_frame)
    spots_df = prepare_spots(raw_df, first_frame)

    # ------------------------------------------------------------------ viewer
    viewer = napari.Viewer(ndisplay=3)

    custom_theme = napari.utils.theme.get_theme("dark")
    custom_theme.canvas = "#7A8CA3"
    napari.utils.theme.register_theme("custom", custom_theme, "custom")
    viewer.theme = "custom"

    controller = Controller(viewer, spots_df)

    print(controller.track_id_value_counts())

    # Load most recent corrections if available
    correction_files = sorted(corrections_dir.glob("corrections_*.pkl"))
    if correction_files:
        latest_corrections = correction_files[-1]
        print(f"Loading corrections from {latest_corrections.name}")
        controller.load_corrections(latest_corrections)

    controller.refresh_status()

    print(controller.track_id_value_counts())

    points_layer = controller.points_layer

    # ---------------------------------------------------------- key bindings
    # All closures access controller.data rather than a captured local so that
    # the undo operation (which replaces controller.data) is always reflected.

    @points_layer.bind_key("l")
    def link_nuclei(layer):
        """Link currently selected nuclei as parent-child."""
        selected = layer.selected_data
        points = controller.data.points

        nuc_ids = sorted(selected, key=lambda nid: points[nid, 0])
        frames = points[nuc_ids, 0]

        for frame in frames[1:]:
            if frame <= frames[0]:
                show_info(
                    "Selection is not in strictly increasing frame order — link cancelled."
                )
                return

        parent_ix = nuc_ids[0]

        for child_ix in nuc_ids[1:]:
            parent_id = controller.data.df.index[parent_ix]
            child_id = controller.data.df.index[child_ix]

            action = LinkNucleiAction(parent_id, child_id)
            controller.do_action(action)

        controller.refresh_status()

    @points_layer.bind_key("u")
    def unlink_from_parent(layer):
        """Sever the selected nucleus from its parent."""
        for nuc_ix in layer.selected_data:
            nuc_id = controller.data.index_id[nuc_ix]
            if controller.data.df.loc[nuc_id, "parent_id"] == -1:
                show_info(f"Nucleus {nuc_id} already has no parent — skipped.")
                continue
            action = UnlinkFromParentAction(nuc_id)
            controller.do_action(action)

        controller.refresh_status()

    @points_layer.bind_key("q")
    def switch_view(_layer):
        if controller.current_view == "status":
            controller.color_by_track()
        elif controller.current_view == "track":
            controller.color_by_count()
        elif controller.current_view == "count":
            controller.color_by_contiguity()
        else:
            controller.color_by_status()

    @points_layer.bind_key("n", overwrite=True)
    def next_detachment(_layer):
        controller.step_contiguity_event(1)

    @points_layer.bind_key("b", overwrite=True)
    def previous_detachment(_layer):
        controller.step_contiguity_event(-1)

    @points_layer.bind_key("shift+s")
    def save_corrections(_layer):
        timestamp = int(time())
        save_path = corrections_dir / f"corrections_{timestamp}.pkl"
        controller.save_corrections(save_path)
        show_info(
            f"Saved {len(controller.action_history.get_actions())} corrections to {save_path.name}"
        )

    @points_layer.bind_key("shift+z")
    def undo(_layer):
        controller.undo()

    @points_layer.bind_key("shift+e")
    def export_dataframe(_layer):
        write_corrected_spots(
            controller.export_dataframe(), metadata, export_path, frame_offset
        )

        show_info(f"Exported corrected spots → {export_path.name}")

    @points_layer.bind_key("t")
    def mark_as_terminal(layer):
        df = controller.data.df

        for nuc_ix in layer.selected_data:
            nuc_id = controller.data.index_id[nuc_ix]
            if df.loc[nuc_id, "n_children"] != 0:
                show_info(
                    "One or more selected nuclei are not terminal — mark cancelled."
                )
                return

        for nuc_ix in layer.selected_data:
            nuc_id = controller.data.index_id[nuc_ix]
            action = MarkNucleusAction(nuc_id, Status.END)
            controller.do_action(action)

        controller.refresh_status()

    @points_layer.bind_key("p")
    def mark_as_pole_terminal(layer):
        df = controller.data.df

        for nuc_ix in layer.selected_data:
            nuc_id = controller.data.index_id[nuc_ix]
            if df.loc[nuc_id, "n_children"] != 0:
                show_info(
                    "One or more selected nuclei are not terminal — mark cancelled."
                )
                return

        for nuc_ix in layer.selected_data:
            nuc_id = controller.data.index_id[nuc_ix]
            action = MarkNucleusAction(nuc_id, Status.POLE_TERMINAL)
            controller.do_action(action)

        controller.refresh_status()

    @points_layer.bind_key("g")
    def mark_as_issue(layer):
        for nuc_ix in layer.selected_data:
            nuc_id = controller.data.index_id[nuc_ix]
            action = MarkIssueAction(nuc_id)
            controller.do_action(action)

        controller.refresh_status()

    @points_layer.bind_key("h")
    def mark_as_not_issue(layer):
        for nuc_ix in layer.selected_data:
            nuc_id = controller.data.index_id[nuc_ix]
            action = MarkNotIssueAction(nuc_id)
            controller.do_action(action)

        controller.refresh_status()

    @points_layer.bind_key("Enter")
    def choose_nucleus(layer):
        nuc_id = controller.next_unmarked()

        if nuc_id is None:
            show_info("All terminals have been reviewed.")
            return

        nuc_ix = controller.data.id_index[nuc_id]

        controller.clear_all_selections()
        controller.view_nucleus(nuc_id)

        layer.selected_data.add(nuc_ix)

        branch_ids = controller.get_nuc_branch(nuc_id)

        for branch_id in branch_ids:
            branch_ix = controller.data.id_index[branch_id]
            layer.border_color[branch_ix] = [0.0, 1.0, 1.0, 1.0]

        layer.border_color[nuc_ix] = [0.0, 1.0, 0.0, 1.0]

        n_remaining = len(controller.data.unmarked)
        show_info(
            f"Reviewing nucleus {nuc_id} — branch length {len(branch_ids)}  "
            f"({n_remaining} unmarked terminal{'s' if n_remaining != 1 else ''} remaining)"
        )

        if len(branch_ids) < 4:
            mark_as_issue(layer)
        elif controller.data.df.loc[nuc_id, "AP"] > 0.95:
            mark_as_pole_terminal(layer)

        layer.refresh()

    napari.run()


if __name__ == "__main__":
    # Fallback: run against the hardcoded legacy paths when executed directly.
    # Prefer using `correct-embryo -c config.yml` from the pipeline entry point.
    _spots_path = Path(r"D:\Tracking\NucleiTracking\data\processed\lightsheet\spots")
    _embryo = "lightsheet_20250131_spots.h5"
    _corrections_dir = _spots_path.parent / "corrections"
    _export_path = _spots_path.parent / "lightsheet_20250131_corrected_spots.h5"

    main(
        spots_path=_spots_path / _embryo,
        corrections_dir=_corrections_dir,
        export_path=_export_path,
        first_frame=25,
    )
