"""
Motion-model LAP tracker.

Drop-in alternative to ``lap_tracking.run_lap_tracking`` for movies where
nuclei move a large fraction of their spacing between frames (e.g. the
post-mitotic spreading after NC13), where nearest-neighbour assignment shifts
whole rows of nuclei by one neighbour.

Frame-to-frame linking predicts where each nucleus will be before assigning:

    predicted = position + dt * (w * own_velocity + (1 - w) * local_flow)

- ``own_velocity`` is an exponentially smoothed velocity of the nucleus's own
  track (px per median frame interval);
- ``local_flow`` is the median velocity of its nearest tracked neighbours,
  which also covers nuclei without a history (new tracks, daughters);
- ``dt`` is the frame interval relative to the median, so acquisition pauses
  scale the prediction instead of breaking tracks.

Each frame pair is solved twice: the second pass replaces the neighbour flow
with the median displacement actually measured from the first pass's links,
which adapts to sudden accelerations.  Gap closing uses the same predictions,
extrapolated over the gap.

Assignments are solved as sparse LAPs (Jaqaman et al. 2008 block structure)
with ``scipy.sparse.csgraph.min_weight_full_bipartite_matching``, so memory
scales with the number of candidate links rather than spots squared.
"""

import itertools

import networkx as nx
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import min_weight_full_bipartite_matching
from scipy.spatial import KDTree
from tqdm import tqdm

AXES = ["px_x", "px_y", "px_z"]


def solve_sparse_lap(
    rows: np.ndarray,
    cols: np.ndarray,
    costs: np.ndarray,
    n_a: int,
    n_b: int,
    no_link_cost: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Minimum-cost partial assignment between sets A (n_a) and B (n_b) given
    candidate links (rows[k] -> cols[k], costs[k]).  Leaving an element of A or
    B unassigned costs ``no_link_cost`` each.  Returns the linked (row, col)
    pairs.
    """
    if len(rows) == 0:
        return np.empty(0, int), np.empty(0, int)
    eps = 1e-9  # explicit zeros would be dropped from the sparse matrix
    a, b = np.arange(n_a), np.arange(n_b)
    r = np.concatenate([rows, a, n_a + b, n_a + cols])
    c = np.concatenate([cols, n_b + a, b, n_b + rows])
    w = np.concatenate(
        [
            costs + eps,
            np.full(n_a, no_link_cost),
            np.full(n_b, no_link_cost),
            np.full(len(rows), eps),
        ]
    )
    m = coo_matrix((w, (r, c)), shape=(n_a + n_b, n_b + n_a)).tocsr()
    ri, ci = min_weight_full_bipartite_matching(m)
    keep = (ri < n_a) & (ci < n_b)
    return ri[keep], ci[keep]


def _local_median(
    tree: KDTree | None, values: np.ndarray, points: np.ndarray, k: int
) -> np.ndarray:
    """Median of ``values`` over the k nearest tree points to each point."""
    if tree is None or len(values) == 0:
        return np.zeros_like(points)
    k = min(k, len(values))
    _, nb = tree.query(points, k=k)
    nb = nb.reshape(len(points), k)
    return np.median(values[nb], axis=1)


def _candidates(pred: np.ndarray, targets: np.ndarray, radius: float, k: int):
    """Candidate links within radius of each prediction (up to k per row)."""
    tree = KDTree(targets)
    k = min(k, len(targets))
    d, j = tree.query(pred, k=k, distance_upper_bound=radius)
    d, j = d.reshape(len(pred), k), j.reshape(len(pred), k)
    ok = np.isfinite(d)
    rows = np.broadcast_to(np.arange(len(pred))[:, None], d.shape)[ok]
    return rows, j[ok], d[ok]


def run_motion_tracking(
    spots_df: pd.DataFrame,
    max_distance: float = 11.0,
    max_gap_frames: int = 3,
    frame_times: dict[int, float] | None = None,
    velocity_weight: float = 0.5,
    velocity_smoothing: float = 0.5,
    n_neighbors: int = 12,
    refine: bool = True,
) -> tuple[pd.DataFrame, nx.DiGraph]:
    """
    Link 3D nuclei with a motion-predicting LAP tracker.

    Args:
        spots_df: spots with 'frame', 'px_x', 'px_y', 'px_z' and a unique
            integer index (the spot id).
        max_distance: gating radius in px around each predicted position.
        max_gap_frames: maximum frame gap bridged by gap closing.
        frame_times: optional {frame: time} used to scale predictions by the
            actual frame interval (any time unit).
        velocity_weight: weight of a nucleus's own velocity vs. the local flow
            of its neighbours in the prediction (0 = flow only).
        velocity_smoothing: weight of the newest displacement when updating a
            track's velocity (1 = no smoothing).
        n_neighbors: neighbours used for local flow estimates.
        refine: re-solve each frame pair using the flow measured from the
            first pass.

    Returns:
        (spots_df with 'graph_key' and 'linear_track_id', tracking graph) in
        the same format as ``run_lap_tracking``.
    """
    spots_df = spots_df.copy()
    if "graph_key" not in spots_df.columns:
        spots_df["graph_key"] = spots_df.index
    spots_df["graph_key"] = spots_df["graph_key"].astype(int)

    keys = spots_df["graph_key"].to_numpy()
    pos = spots_df[AXES].to_numpy(float)
    frame = spots_df["frame"].to_numpy()
    frames = np.sort(np.unique(frame))
    rows_at = {f: np.where(frame == f)[0] for f in frames}

    # frame interval relative to the median, from frame f to the next frame
    if frame_times:
        t = np.array([frame_times.get(int(f), np.nan) for f in frames], float)
        dts = np.diff(t)
        med = np.nanmedian(dts) / np.median(np.diff(frames))
        rel = np.where(np.isfinite(dts) & (dts > 0), dts / med, np.diff(frames))
    else:
        rel = np.diff(frames).astype(float)
    dt_next = dict(zip(frames[:-1].tolist(), rel.tolist(), strict=True))

    n = len(spots_df)
    velocity = np.full((n, 3), np.nan)  # px per median frame interval
    successor = np.full(n, -1)
    predecessor = np.full(n, -1)
    no_link = 0.5 * max_distance * 1.05  # any gated link beats two non-links

    # ---------------------------------------------------------------- linking
    for f, f_next in tqdm(list(itertools.pairwise(frames)), desc="Motion linking"):
        if f_next - f > 1:  # skipped frames are bridged by gap closing
            continue
        cur, nxt = rows_at[f], rows_at[f_next]
        if len(cur) == 0 or len(nxt) == 0:
            continue
        dt = dt_next[f]
        p = pos[cur]

        v_own = velocity[cur]
        known = np.isfinite(v_own[:, 0])
        tree = KDTree(p[known]) if known.sum() >= n_neighbors else None
        flow = _local_median(tree, v_own[known], p, n_neighbors)
        v = np.where(
            known[:, None], velocity_weight * v_own + (1 - velocity_weight) * flow, flow
        )
        pred = p + dt * v

        ri, ci = _link(pred, pos[nxt], max_distance, no_link)

        if refine and len(ri) >= n_neighbors:
            disp = pos[nxt[ci]] - p[ri]
            measured = _local_median(KDTree(p[ri]), disp, p, n_neighbors)
            v2 = np.where(
                known[:, None],
                velocity_weight * v_own * dt + (1 - velocity_weight) * measured,
                measured,
            )
            ri, ci = _link(p + v2, pos[nxt], max_distance, no_link)

        a, b = cur[ri], nxt[ci]
        successor[a] = b
        predecessor[b] = a
        new_v = (pos[b] - pos[a]) / dt
        old_v = velocity[a]
        has_old = np.isfinite(old_v[:, 0])
        velocity[b] = np.where(
            has_old[:, None],
            velocity_smoothing * new_v + (1 - velocity_smoothing) * old_v,
            new_v,
        )

    # ------------------------------------------------------------ gap closing
    if max_gap_frames > 1:
        ends = np.where((successor == -1) & (frame < frames[-1]))[0]
        starts = np.where((predecessor == -1) & (frame > frames[0]))[0]
        if len(ends) and len(starts):
            frame_index = {int(f): i for i, f in enumerate(frames)}
            cum_dt = np.concatenate([[0.0], np.cumsum(rel)])
            # flow at each end from tracked neighbours in the same frame
            flow_end = np.zeros((len(ends), 3))
            for f in np.unique(frame[ends]):
                rows = rows_at[f]
                known = rows[np.isfinite(velocity[rows, 0])]
                m = frame[ends] == f
                if len(known) >= n_neighbors:
                    flow_end[m] = _local_median(
                        KDTree(pos[known]), velocity[known], pos[ends[m]], n_neighbors
                    )
            v_end = np.where(
                np.isfinite(velocity[ends, 0])[:, None],
                velocity_weight * np.nan_to_num(velocity[ends])
                + (1 - velocity_weight) * flow_end,
                flow_end,
            )

            start_tree = KDTree(pos[starts])
            rows_l, cols_l, costs_l = [], [], []
            for i, e in enumerate(ends):
                fe = frame_index[int(frame[e])]
                hits = start_tree.query_ball_point(
                    pos[e], max_distance * max_gap_frames
                )
                for j in hits:
                    s = starts[j]
                    gap = frame[s] - frame[e]
                    if gap < 1 or gap > max_gap_frames:
                        continue
                    fs = frame_index.get(int(frame[s]))
                    if fs is None:
                        continue
                    pred = pos[e] + v_end[i] * (cum_dt[fs] - cum_dt[fe])
                    d = np.linalg.norm(pos[s] - pred)
                    if d < max_distance:
                        rows_l.append(i)
                        cols_l.append(j)
                        costs_l.append(d)
            ri, ci = solve_sparse_lap(
                np.array(rows_l, int),
                np.array(cols_l, int),
                np.array(costs_l, float),
                len(ends),
                len(starts),
                no_link,
            )
            successor[ends[ri]] = starts[ci]
            predecessor[starts[ci]] = ends[ri]

    # ------------------------------------------------------------- output
    track = np.full(n, -1)
    next_id = 1
    for r in np.where(predecessor == -1)[0]:
        cur = r
        while cur != -1:
            track[cur] = next_id
            cur = successor[cur]
        next_id += 1
    spots_df["linear_track_id"] = track

    graph = nx.DiGraph()
    graph.add_nodes_from(keys.tolist())
    linked = np.where(successor != -1)[0]
    for a in linked:
        b = successor[a]
        graph.add_edge(
            int(keys[a]),
            int(keys[b]),
            time=int(frame[b] - frame[a]),
            track_id=int(track[a]),
        )
    print(f"Tracking complete. Found {next_id - 1} tracks.")
    return spots_df, graph


def _link(pred: np.ndarray, targets: np.ndarray, radius: float, no_link: float):
    rows, cols, costs = _candidates(pred, targets, radius, k=10)
    return solve_sparse_lap(rows, cols, costs, len(pred), len(targets), no_link)
