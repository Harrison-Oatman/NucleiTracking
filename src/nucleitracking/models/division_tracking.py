"""
Division mapping: link daughter tracklets to mother tracklets in each mitosis
window, given linear (division-free) tracks with interpolated gaps.

Two methods:

``map_divisions`` (LAP)
    One LAP per window. Mothers are tracklets spanning the window, daughters
    are tracklets starting inside it and outliving it; each mother takes at
    most one new daughter, at the frame before the daughter appears.

``map_divisions_hypotheses``
    Same backward-interpolated midpoint cost, but over explicit hypotheses
    solved jointly with an integer program per window:

    - one new daughter: mother track M continues to sibling S, new track D
      appears; cost = |midpoint(back(D), back(S)) - M|
    - two new daughters: M ends inside the window (e.g. its anaphase
      segmentation was missed, or tracking broke there) and two new tracks
      D1, D2 appear within ``max_mother_gap`` frames; cost =
      |midpoint(back(D1), back(D2)) - M|

    ``back(X)`` extrapolates a daughter's first real position back to the
    mother's frame with its own initial velocity. Mother positions are the last
    *real* (non-interpolated) detection before the daughter appears, up to
    ``max_mother_gap`` frames earlier, so a missed detection in anaphase does
    not hide the mother or bias it with interpolated positions. Mothers are any
    tracklet established before the window (no need to span it unbroken), and
    daughters any tracklet starting in it with ``min_daughter_len`` real spots.
    Each daughter and each mother is used at most once per window, which also
    keeps every lineage to one division per cycle.
"""

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, linear_sum_assignment, milp
from scipy.sparse import coo_matrix
from scipy.spatial import KDTree

AXES = ["px_x", "px_y", "px_z"]


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


def map_divisions(
    spots_df: pd.DataFrame,
    windows: list[tuple[int, int]],
    new_track_cost: float = 20,
) -> pd.DataFrame:
    """
    One LAP per mitosis window [start, end]: mothers exist before the window and
    span it, daughters start inside it and outlive it.
    """
    spots_df = spots_df.copy()
    spots_df["status"] = 0

    for start, end in windows:
        print(f"mapping divisions in window {start}-{end}")
        cost_matrix, parent_ids, child_ids, spots_df = get_division_costs(
            spots_df, start - 1, end
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


def map_divisions_hypotheses(
    spots_df: pd.DataFrame,
    windows: list[tuple[int, int]],
    max_cost: float = 5.0,
    max_mother_gap: int = 3,
    min_daughter_len: int = 3,
    search_radius: float = 15.0,
    min_sibling_dist: float = 3.0,
    pair_penalty: float = 0.5,
    allow_cut: bool = False,
    cut_penalty: float = 1.0,
) -> pd.DataFrame:
    """
    Assign daughters to mothers by choosing the best non-conflicting set of
    one- and two-new-daughter hypotheses in each window (see module docstring).
    Distances are in px. Links may skip frames when the mother's last real
    detection is earlier than the frame before the daughter; re-interpolate
    afterwards.

    With ``allow_cut``, a mother that currently continues into S may instead
    take two new daughters, cutting S loose (its continuation was wrong, e.g.
    carried onto a neighbour's daughter). A freed S becomes a daughter
    candidate for other mothers in the same integer program; ``cut_penalty``
    keeps existing continuations unless the cut is clearly better.
    """
    df = spots_df.copy()
    ids = df.index.to_numpy()
    frame = df["frame"].to_numpy().astype(int)
    pos = df[AXES].to_numpy(float)
    tracklet = df["tracklet_id"].to_numpy()
    real = (
        ~df["is_interpolated"].fillna(False).astype(bool).to_numpy()
        if "is_interpolated" in df.columns
        else np.ones(len(df), bool)
    )

    # real spots of each tracklet in frame order, and the next real spot
    order = np.lexsort((frame, tracklet))
    order = order[real[order]]
    next_real = np.full(len(df), -1)
    same = tracklet[order[1:]] == tracklet[order[:-1]]
    next_real[order[:-1][same]] = order[1:][same]
    t_first = pd.Series(frame).groupby(tracklet).min()
    t_nreal = pd.Series(real.astype(int)).groupby(tracklet).sum()
    t_head = (
        pd.Series(np.arange(len(df)))
        .groupby(tracklet)
        .agg(lambda r: r.iloc[np.argmin(frame[r.to_numpy()])])
    )

    def velocity(r: int) -> np.ndarray:
        n = next_real[r]
        if n == -1:
            return np.zeros(3)
        return (pos[n] - pos[r]) / (frame[n] - frame[r])

    def back(r: int, f: int) -> np.ndarray:
        return pos[r] - (frame[r] - f) * velocity(r)

    parent = df["parent_id"].to_numpy().copy()

    # real spots remaining in the tracklet from each real spot onwards
    remaining = np.zeros(len(df), int)
    remaining[order] = (
        pd.Series(tracklet[order]).groupby(tracklet[order]).cumcount(ascending=False)
        + 1
    ).to_numpy()

    def pair_cost(d1: int, d2: int, m: int) -> float | None:
        if np.linalg.norm(pos[d1] - pos[d2]) < min_sibling_dist:
            return None
        f = frame[m]
        cost = float(np.linalg.norm((back(d1, f) + back(d2, f)) / 2 - pos[m]))
        return cost if cost <= max_cost else None

    for ws, we in windows:
        daughters = [
            int(t_head[t])
            for t in t_first.index[(t_first >= ws) & (t_first <= we)]
            if t_nreal[t] >= min_daughter_len and real[t_head[t]]
        ]
        mother_tracklets = set(t_first.index[t_first < ws])
        mother_rows = np.where(
            real
            & (frame >= ws - max_mother_gap)
            & (frame <= we)
            & np.isin(tracklet, list(mother_tracklets))
        )[0]
        by_frame = {
            f: mother_rows[frame[mother_rows] == f]
            for f in np.unique(frame[mother_rows])
        }
        trees = {f: KDTree(pos[r]) for f, r in by_frame.items()}

        # hypothesis: (cost term, mother tracklet, mother spot, daughters, freed S)
        hyps: list[tuple[float, int, int, tuple[int, ...], int]] = []
        ends: dict[tuple[int, int], set[int]] = {}
        conts: dict[tuple[int, int], set[int]] = {}

        def add_candidates(d: int, exclude_tracklet: int = -1) -> None:
            fd = frame[d]
            for g in range(1, max_mother_gap + 1):
                f = fd - g
                if f not in trees:
                    continue
                for k in trees[f].query_ball_point(pos[d], search_radius):
                    m = by_frame[f][k]
                    if tracklet[m] == exclude_tracklet:
                        continue
                    nx = next_real[m]
                    if nx != -1 and frame[nx] < fd:
                        continue  # a later detection of this mother exists
                    key = (int(tracklet[m]), int(m))
                    if nx != -1:  # mother continues: one new daughter
                        mid = (back(d, f) + back(nx, f)) / 2
                        cost = float(np.linalg.norm(mid - pos[m]))
                        if cost <= max_cost:
                            hyps.append((cost - max_cost, *key, (d,), -1))
                        if exclude_tracklet == -1:
                            conts.setdefault(key, set()).add(d)
                    else:  # mother ends here: candidate for two new daughters
                        ends.setdefault(key, set()).add(d)

        for d in daughters:
            add_candidates(d)

        for (t, m), cands in ends.items():
            cands = sorted(cands)
            for i, d1 in enumerate(cands):
                for d2 in cands[i + 1 :]:
                    cost = pair_cost(d1, d2, m)
                    if cost is not None:
                        hyps.append(
                            (2 * (cost - max_cost) + pair_penalty, t, m, (d1, d2), -1)
                        )

        freed: set[int] = set()
        if allow_cut:
            for (t, m), cands in conts.items():
                s_row = int(next_real[m])
                if not ws <= frame[s_row] <= we:
                    continue
                cands = sorted(cands)
                for i, d1 in enumerate(cands):
                    for d2 in cands[i + 1 :]:
                        cost = pair_cost(d1, d2, m)
                        if cost is not None:
                            hyps.append(
                                (
                                    2 * (cost - max_cost) + cut_penalty,
                                    t,
                                    m,
                                    (d1, d2),
                                    s_row,
                                )
                            )
                            freed.add(s_row)
            # a freed continuation may become another mother's daughter
            for s_row in sorted(freed):
                if remaining[s_row] >= min_daughter_len:
                    add_candidates(s_row, exclude_tracklet=int(tracklet[s_row]))

        if not hyps:
            print(f"window {ws}-{we}: no division hypotheses")
            continue

        # integer program. Rows: each daughter used at most once; a freed S is
        # usable only if its mother's cut is chosen; each mother tracklet once.
        targets = daughters + sorted(freed - set(daughters))
        d_index = {d: i for i, d in enumerate(targets)}
        t_index = {t: i for i, t in enumerate({h[1] for h in hyps})}
        n_d = len(targets)
        rows, cols, vals = [], [], []
        c = np.array([h[0] for h in hyps])
        for j, (_c, t, _m, ds, s_row) in enumerate(hyps):
            for d in ds:
                rows.append(d_index[d])
                cols.append(j)
                vals.append(1.0)
            if s_row != -1:
                rows.append(d_index[s_row])
                cols.append(j)
                vals.append(-1.0)
            rows.append(n_d + t_index[t])
            cols.append(j)
            vals.append(1.0)
        upper = np.ones(n_d + len(t_index))
        upper[[d_index[s] for s in freed if s not in set(daughters)]] = 0.0
        a = coo_matrix(
            (vals, (rows, cols)), shape=(n_d + len(t_index), len(hyps))
        ).tocsr()
        res = milp(
            c,
            constraints=LinearConstraint(a, -np.inf, upper),
            integrality=np.ones(len(hyps)),
            bounds=Bounds(0, 1),
        )
        chosen = np.where(res.x > 0.5)[0] if res.x is not None else []

        counts = {"one": 0, "two": 0, "cut": 0}
        for j in chosen:  # cuts first, so a freed S can be re-linked below
            if hyps[j][4] != -1:
                parent[hyps[j][4]] = -1
        for j in chosen:
            _c, _t, m, ds, s_row = hyps[j]
            for d in ds:
                parent[d] = ids[m]
            kind = "cut" if s_row != -1 else ("one" if len(ds) == 1 else "two")
            counts[kind] += 1
        print(
            f"window {ws}-{we}: {len(daughters)} daughter candidates, "
            f"{len(hyps)} hypotheses -> {counts['one']} one-new-daughter, "
            f"{counts['two']} two-new-daughter, {counts['cut']} cut-continuation"
        )

    df["parent_id"] = parent
    return df
