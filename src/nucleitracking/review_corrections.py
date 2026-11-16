"""
review_corrections.py — visualise lineage corrections in Napari.

Usage
-----
    review-corrections -c path/to/config.yml [--side-by-side]

Loads the tracked spots exactly as ``correct-embryo`` does, replays a
corrections pickle (the latest one by default), and shows what changed:

Highlight mode (default)
    One 3D view of the corrected embryo, plus one layer per kind of edit:
      - ``added links``    new parent→child links, coloured by edit type
      - ``removed links``  original links that were cut (red)
      - ``edited: <type>`` the nuclei whose parent changed, enlarged
      - ``marked: issue / pole`` nuclei flagged ISSUE / POLE_TERMINAL
    Links are drawn in both the parent's and the child's frame.

Side-by-side mode (``--side-by-side``)
    Original tracking (left) and corrected tracking (right), sharing the same
    camera and time slider.  Nuclei are coloured by lineage, so a rescued
    daughter switches to its mother's colour on the right.

Keys
----
    n / b   jump to the next / previous edit (sorted by frame)
    m       toggle whether ISSUE/POLE marks are included in n / b

Edit types are read from the ``auto_reason`` stored on automatic actions;
edits made by hand in ``correct-embryo`` are shown as ``manual``.
"""

import argparse
import contextlib
import io
import pickle
from dataclasses import dataclass
from pathlib import Path

import colorcet as cc
import napari
import numpy as np
import pandas as pd
from napari.utils.notifications import show_info
from scipy.spatial import KDTree
from sklearn.decomposition import PCA

from .corrections import (
    DataSpace,
    LinkNucleiAction,
    MarkIssueAction,
    MarkNotIssueAction,
    MarkNucleusAction,
    Status,
    UnlinkFromParentAction,
    calculate_tracks,
    load_spots,
    prepare_spots,
)

COORDS = ["frame", "z", "y", "x"]

EDIT_COLORS = {
    "zipper": "#00e5ff",
    "division": "#ff3df5",
    "continuation": "#ffd600",
    "stub/head unlink": "#ff8c00",
    "manual": "#ffffff",
    "issue": "#ebacda",
    "pole": "#588157",
}
REMOVED_COLOR = "#ff1744"


@dataclass
class Edit:
    child: int
    kind: str  # one of EDIT_COLORS
    old_parent: int
    new_parent: int
    reason: str


def _edit_type(action) -> str:
    reason = getattr(action, "auto_reason", None)
    if reason is None:
        return "manual"
    if reason.startswith("zipper"):
        return "zipper"
    if reason.startswith("rescue:link division"):
        return "division"
    if reason.startswith("rescue:link"):
        return "continuation"
    if reason.startswith("rescue:unlink"):
        return "stub/head unlink"
    if reason.startswith("mark:issue"):
        return "issue"
    if reason.startswith("mark:pole"):
        return "pole"
    return "manual"


def replay(spots_df: pd.DataFrame, actions: list) -> tuple[pd.DataFrame, list[Edit]]:
    """
    Replay actions on a copy of spots_df. Returns the corrected DataFrame and
    the net edits: one per nucleus whose parent changed, plus one per mark.
    """
    with contextlib.redirect_stdout(io.StringIO()):
        data = DataSpace(spots_df.copy())
    original_parent = data.df["parent_id"].copy()

    last_type: dict[int, tuple[str, str]] = {}
    marks: dict[int, tuple[str, str]] = {}
    with contextlib.redirect_stdout(io.StringIO()):
        for action in actions:
            kind = _edit_type(action)
            reason = getattr(action, "auto_reason", "manual")
            if isinstance(action, (LinkNucleiAction, UnlinkFromParentAction)):
                child = (
                    action.child_id
                    if isinstance(action, LinkNucleiAction)
                    else action.nuc_id
                )
                # a later relink describes the edit better than the unlink before it
                if isinstance(action, LinkNucleiAction) or child not in last_type:
                    last_type[child] = (kind, reason)
            elif isinstance(action, MarkIssueAction):
                marks[action.nuc_id] = (
                    "issue" if kind != "manual" else "manual",
                    reason,
                )
            elif isinstance(action, MarkNucleusAction):
                if action.status == Status.POLE_TERMINAL:
                    marks[action.nuc_id] = (
                        "pole" if kind != "manual" else "manual",
                        reason,
                    )
                else:
                    marks[action.nuc_id] = ("manual", f"marked {action.status.name}")
            elif isinstance(action, MarkNotIssueAction):
                marks.pop(action.nuc_id, None)
            action.execute(data)

    corrected = data.df
    corrected["status"] = corrected["status"].map(lambda s: s.name)
    changed = corrected.index[corrected["parent_id"] != original_parent]

    edits = []
    for child in changed:
        kind, reason = last_type.get(child, ("manual", "manual"))
        edits.append(
            Edit(
                int(child),
                kind,
                int(original_parent[child]),
                int(corrected.loc[child, "parent_id"]),
                reason,
            )
        )
    for nuc, (kind, reason) in marks.items():
        edits.append(Edit(int(nuc), kind, -1, -1, reason))
    return corrected, edits


def build_tracks(df: pd.DataFrame) -> tuple[np.ndarray, dict, np.ndarray]:
    """
    Split the lineage graph into tracklets (chains between divisions) for a
    napari Tracks layer. Returns (data, graph, lineage_id per vertex).
    """
    parent = df["parent_id"]
    n_children = df.index.map(parent.value_counts()).fillna(0).astype(int)
    n_children = pd.Series(n_children.to_numpy(), index=df.index)
    has_parent = parent != -1
    continues = has_parent & parent.map(n_children).eq(1)

    tracklet = pd.Series(df.index.to_numpy(), index=df.index)
    for _frame, idx in df.groupby("frame").groups.items():
        cont = idx[continues.loc[idx].to_numpy()]
        tracklet.loc[cont] = parent.loc[cont].map(tracklet).to_numpy()

    heads = df.index[has_parent & ~continues]
    graph = {
        int(tracklet[h]): [int(tracklet[parent[h]])]
        for h in heads
        if parent[h] in tracklet.index
    }
    data = np.column_stack([tracklet.to_numpy(), df[COORDS].to_numpy()])
    order = np.lexsort((data[:, 1], data[:, 0]))
    return data[order], graph, df["track_id"].to_numpy()[order]


PALETTE = np.array(
    [[int(h[i : i + 2], 16) / 255 for i in (1, 3, 5)] for h in cc.glasbey[:256]]
)


def _lineage_colors(df: pd.DataFrame) -> np.ndarray:
    return PALETTE[df["track_id"].to_numpy().astype(int) % 256]


def _link_lines(df_from: pd.DataFrame, edits: list[Edit], which: str):
    """Line segments (4D) for each edit's old or new link, in both end frames."""
    lines, kinds = [], []
    for e in edits:
        p = e.old_parent if which == "old" else e.new_parent
        if p == -1:
            continue
        a = df_from.loc[p, COORDS].to_numpy(float)
        b = df_from.loc[e.child, COORDS].to_numpy(float)
        for t in (a[0], b[0]):
            lines.append(np.array([[t, *a[1:]], [t, *b[1:]]]))
            kinds.append(e.kind)
    return lines, kinds


def _add_group(
    viewer: napari.Viewer,
    df: pd.DataFrame,
    label: str,
    edits: list[Edit],
    which: str,
    show_tracks: bool,
):
    """Add nuclei, tracks, link lines and edited nuclei for one panel."""
    viewer.add_points(
        df[COORDS].to_numpy(),
        size=df["radius"].to_numpy() * 2.0,
        face_color=_lineage_colors(df),
        border_color="black",
        border_width=0.1,
        opacity=0.6,
        name=f"{label} nuclei",
    )

    if show_tracks:
        data, graph, _ = build_tracks(df)
        viewer.add_tracks(
            data,
            graph=graph,
            tail_length=8,
            head_length=0,
            tail_width=1.5,
            color_by="track_id",
            colormap="hsv",
            name=f"{label} tracks",
        )
    else:  # keep groups the same size for grid mode
        viewer.add_points(np.empty((0, 4)), name=f"{label} tracks (off)")

    link_edits = [e for e in edits if e.old_parent != -1 or e.new_parent != -1]
    lines, kinds = _link_lines(df, link_edits, which)
    color = (
        [REMOVED_COLOR] * len(lines)
        if which == "old"
        else [EDIT_COLORS[k] for k in kinds]
    )
    viewer.add_shapes(
        lines or None,
        shape_type="line",
        edge_color=color or "white",
        edge_width=0.6,
        ndim=4,
        name=f"{label} {'removed' if which == 'old' else 'added'} links",
    )

    pts = link_edits
    viewer.add_points(
        df.loc[[e.child for e in pts], COORDS].to_numpy() if pts else np.empty((0, 4)),
        size=df.loc[[e.child for e in pts], "radius"].to_numpy() * 3.0 if pts else 1,
        face_color="transparent",
        border_color=[EDIT_COLORS[e.kind] for e in pts] or "white",
        border_width=0.25,
        out_of_slice_display=True,
        properties={"edit": [e.kind for e in pts], "reason": [e.reason for e in pts]},
        name=f"{label} edited nuclei",
    )


def build_viewer(
    viewer: napari.Viewer,
    original: pd.DataFrame,
    corrected: pd.DataFrame,
    edits: list[Edit],
    side_by_side: bool = False,
    show_tracks: bool = True,
):
    if side_by_side:
        _add_group(viewer, original, "original", edits, "old", show_tracks)
        _add_group(viewer, corrected, "corrected", edits, "new", show_tracks)
        viewer.grid.stride = 4
        viewer.grid.shape = (1, 2)
        viewer.grid.enabled = True
        return

    _add_group(viewer, corrected, "corrected", edits, "new", show_tracks)
    lines, _ = _link_lines(original, edits, "old")
    viewer.add_shapes(
        lines or None,
        shape_type="line",
        edge_color=REMOVED_COLOR,
        edge_width=0.4,
        ndim=4,
        name="removed links",
    )
    if show_tracks:
        data, graph, _ = build_tracks(original)
        viewer.add_tracks(
            data,
            graph=graph,
            tail_length=8,
            head_length=0,
            colormap="gray",
            name="original tracks",
            visible=False,
        )
    for kind in ("issue", "pole"):
        marked = [e.child for e in edits if e.kind == kind]
        if marked:
            viewer.add_points(
                corrected.loc[marked, COORDS].to_numpy(),
                size=corrected.loc[marked, "radius"].to_numpy() * 2.5,
                face_color=EDIT_COLORS[kind],
                border_color="black",
                out_of_slice_display=True,
                name=f"marked: {kind}",
                visible=kind == "issue",
            )


class EditNavigator:
    """Step through edits with n / b, centring the camera on each nucleus."""

    def __init__(self, viewer: napari.Viewer, df: pd.DataFrame, edits: list[Edit]):
        self.viewer = viewer
        self.df = df
        self.all_edits = sorted(edits, key=lambda e: df.loc[e.child, "frame"])
        self.include_marks = False
        self.i = -1
        self.center = df[["z", "y", "x"]].to_numpy().mean(axis=0)
        self._trees: dict[int, tuple[KDTree, np.ndarray]] = {}
        self.zoomed = False

    @property
    def edits(self) -> list[Edit]:
        if self.include_marks:
            return self.all_edits
        return [e for e in self.all_edits if e.kind not in ("issue", "pole")]

    def _normal(self, frame: int, pos: np.ndarray) -> np.ndarray:
        if frame not in self._trees:
            pts = self.df.loc[self.df["frame"] == frame, ["z", "y", "x"]].to_numpy()
            self._trees[frame] = (KDTree(pts), pts)
        tree, pts = self._trees[frame]
        _, nb = tree.query(pos, k=min(25, len(pts)))
        normal = PCA(3).fit(pts[nb]).components_[-1]
        out = pos - self.center
        if np.linalg.norm(out + normal) > np.linalg.norm(out):
            normal = -normal
        return normal

    def step(self, delta: int):
        edits = self.edits
        if not edits:
            show_info("No edits to show.")
            return
        self.i = (self.i + delta) % len(edits)
        e = edits[self.i]
        row = self.df.loc[e.child]
        pos = row[["z", "y", "x"]].to_numpy(float)
        self.viewer.dims.set_current_step(0, int(row["frame"]))
        if not self.zoomed:  # first jump: frame ~1/6 of the embryo, then keep user zoom
            self.viewer.reset_view()
            self.viewer.camera.zoom *= 6
            self.zoomed = True
        self.viewer.camera.center = tuple(pos)
        self.viewer.camera.set_view_direction(self._normal(int(row["frame"]), pos))
        link = (
            f"parent {e.old_parent} -> {e.new_parent}"
            if e.kind not in ("issue", "pole")
            else ""
        )
        show_info(
            f"[{self.i + 1}/{len(edits)}] {e.kind}: nucleus {e.child}, "
            f"frame {int(row['frame'])}  {link}  ({e.reason})"
        )

    def toggle_marks(self):
        self.include_marks = not self.include_marks
        self.i = -1
        show_info(
            f"{'Including' if self.include_marks else 'Excluding'} ISSUE/POLE marks "
            f"({len(self.edits)} edits)"
        )


def load(config_path: Path, corrections: Path | None):
    from .pipeline.config import PipelineConfig

    config = PipelineConfig.load(config_path)
    track_dir = config.dataset / f"tracking_{config.param_set_name}"
    _stem, _meta, raw = load_spots(track_dir / f"{config.param_set_name}_spots.h5")
    spots = prepare_spots(raw, config.local_post.tracking.start_frame)

    if corrections is None:
        files = sorted((track_dir / "corrections").glob("corrections_*.pkl"))
        if not files:
            msg = f"No corrections found in {track_dir / 'corrections'}"
            raise FileNotFoundError(msg)
        corrections = files[-1]
    with corrections.open("rb") as f:
        actions = pickle.load(f)
    print(f"Replaying {len(actions)} actions from {corrections.name}")

    corrected, edits = replay(spots, actions)
    original, _ = replay(spots, [])  # same derived columns (n_parents, status)
    with contextlib.redirect_stdout(io.StringIO()):
        original = calculate_tracks(original)
        corrected = calculate_tracks(corrected)
    counts = pd.Series([e.kind for e in edits]).value_counts()
    print("Edits:", counts.to_dict())
    return config.param_set_name, original, corrected, edits


def main():
    parser = argparse.ArgumentParser(
        description="Show lineage corrections in Napari.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("-c", "--config", required=True, metavar="CONFIG")
    parser.add_argument(
        "--corrections",
        type=Path,
        default=None,
        help="Corrections pickle to show (default: latest in corrections/).",
    )
    parser.add_argument(
        "--side-by-side",
        action="store_true",
        help="Original (left) and corrected (right) in a shared-camera grid.",
    )
    parser.add_argument(
        "--no-tracks", action="store_true", help="Skip track tails (faster)."
    )
    args = parser.parse_args()

    name, original, corrected, edits = load(Path(args.config), args.corrections)

    viewer = napari.Viewer(ndisplay=3, title=f"Corrections — {name}")
    build_viewer(
        viewer,
        original,
        corrected,
        edits,
        side_by_side=args.side_by_side,
        show_tracks=not args.no_tracks,
    )

    nav = EditNavigator(viewer, corrected, edits)
    viewer.bind_key("n", lambda _v: nav.step(1), overwrite=True)
    viewer.bind_key("b", lambda _v: nav.step(-1), overwrite=True)
    viewer.bind_key("m", lambda _v: nav.toggle_marks(), overwrite=True)
    show_info("n / b: next / previous edit,  m: include ISSUE/POLE marks")

    napari.run()


if __name__ == "__main__":
    main()
